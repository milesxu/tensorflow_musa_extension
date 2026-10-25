#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "../utils_op.h"
#include "musa_pad_params.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

#ifndef MAX_DIM
#define MAX_DIM 4  // NHWC 4d support only
#endif

namespace tensorflow {
namespace musa {
extern "C" {
void nd_pad_kernel_launcher_float(const float *, float *, MusaPadParams,
                                  const float,
                                  const musaStream_t);
void nd_pad_kernel_launcher_double(const double *, double *, MusaPadParams,
                                   const double,
                                   const musaStream_t);
void nd_pad_kernel_launcher_int32(const int32_t *, int32_t *, MusaPadParams,
                                  const int32_t,
                                  const musaStream_t);
void nd_pad_kernel_launcher_int64(const int64_t *, int64_t *, MusaPadParams,
                                  const int64_t,
                                  const musaStream_t);
void nd_pad_kernel_launcher_uint8(const uint8_t *, uint8_t *, MusaPadParams,
                                  const uint8_t,
                                  const musaStream_t);
void nd_pad_kernel_launcher_uint16(const uint16_t *, uint16_t *, MusaPadParams,
                                   const uint16_t,
                                   const musaStream_t);
}

namespace {
struct FloatTag {};
struct DoubleTag {};
struct Int32Tag {};
struct Int64Tag {};
struct Uint8Tag {};
struct HalfTag {};
struct BFloat16Tag {};
struct UnsupportedTag {};

template <typename T>
struct TypeTag {
  using type = UnsupportedTag;
};

template <>
struct TypeTag<float> {
  using type = FloatTag;
};
template <>
struct TypeTag<double> {
  using type = DoubleTag;
};
template <>
struct TypeTag<int32_t> {
  using type = Int32Tag;
};
template <>
struct TypeTag<int64_t> {
  using type = Int64Tag;
};
template <>
struct TypeTag<uint8_t> {
  using type = Uint8Tag;
};
template <>
struct TypeTag<Eigen::half> {
  using type = HalfTag;
};
template <>
struct TypeTag<bfloat16> {
  using type = BFloat16Tag;
};

#define DEFINE_PAD_LAUNCHER_IMPL(T, TAG, SUFFIX)                              \
  void CallPadLauncherImpl(                                                   \
      const T *input_data, T *output_data, const MusaPadParams &params,       \
      const T pad_value, const musaStream_t stream, TAG) {                    \
    nd_pad_kernel_launcher_##SUFFIX(input_data, output_data, params,          \
                                    pad_value, stream);                        \
  }

DEFINE_PAD_LAUNCHER_IMPL(float, FloatTag, float)
DEFINE_PAD_LAUNCHER_IMPL(double, DoubleTag, double)
DEFINE_PAD_LAUNCHER_IMPL(int32_t, Int32Tag, int32)
DEFINE_PAD_LAUNCHER_IMPL(int64_t, Int64Tag, int64)
DEFINE_PAD_LAUNCHER_IMPL(uint8_t, Uint8Tag, uint8)

#undef DEFINE_PAD_LAUNCHER_IMPL

// half and bfloat16 dispatched via uint16 kernel (zero = 0x0000 for both)
void CallPadLauncherImpl(const Eigen::half *input_data,
                         Eigen::half *output_data,
                         const MusaPadParams &params,
                         const Eigen::half pad_value,
                         const musaStream_t stream, HalfTag) {
  uint16_t pad_bits;
  memcpy(&pad_bits, &pad_value, sizeof(uint16_t));
  nd_pad_kernel_launcher_uint16(reinterpret_cast<const uint16_t *>(input_data),
                                reinterpret_cast<uint16_t *>(output_data), params,
                                pad_bits, stream);
}

void CallPadLauncherImpl(const bfloat16 *input_data, bfloat16 *output_data,
                         const MusaPadParams &params, const bfloat16 pad_value,
                         const musaStream_t stream, BFloat16Tag) {
  uint16_t pad_bits;
  memcpy(&pad_bits, &pad_value, sizeof(uint16_t));
  nd_pad_kernel_launcher_uint16(reinterpret_cast<const uint16_t *>(input_data),
                                reinterpret_cast<uint16_t *>(output_data), params,
                                pad_bits, stream);
}

template <typename T>
void CallPadLauncherImpl(const T *, T *, const MusaPadParams &, const T,
                         const musaStream_t, UnsupportedTag) {
  throw std::invalid_argument("Pad not support current type");
}

template <typename T>
void CallPadLauncher(const T *input_data, T *output_data,
                     const MusaPadParams &params, const T pad_value,
                     const musaStream_t stream) {
  CallPadLauncherImpl(input_data, output_data, params, pad_value, stream,
                      typename TypeTag<T>::type());
}
}  // namespace

template <typename T, typename Tpadding>
class MusaPadOp : public OpKernel {
 public:
  explicit MusaPadOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext *ctx) override {
    const Tensor &input = ctx->input(0);
    const Tensor &paddings = ctx->input(1);
    const int dims = input.dims();

    static constexpr int kMaxDims = 4;
    OP_REQUIRES(ctx, dims >= 1 && dims <= kMaxDims,
                errors::Unimplemented(
                    "MusaPadGeneric supports 1-4D input, got ", dims, "D"));
    OP_REQUIRES(
        ctx,
        paddings.dims() == 2 && paddings.dim_size(0) == dims &&
            paddings.dim_size(1) == 2,
        errors::InvalidArgument("Paddings must be [", dims, ", 2], got ",
                                paddings.shape().DebugString()));

    T pad_value = T(0);
    if (ctx->num_inputs() == 3) {
      const Tensor &constant_values = ctx->input(2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(constant_values.shape()),
                  errors::InvalidArgument("constant_values must be scalar"));
      pad_value = constant_values.scalar<T>()();
    }

    typename TTypes<Tpadding>::ConstMatrix paddings_mat =
        paddings.matrix<Tpadding>();
    MusaPadParams params{};
    params.dims = dims;
    for (int d = 0; d < dims; ++d) {
      params.pad_before[d] = static_cast<int64_t>(paddings_mat(d, 0));
      params.pad_after[d] = static_cast<int64_t>(paddings_mat(d, 1));
      OP_REQUIRES(ctx, params.pad_before[d] >= 0 && params.pad_after[d] >= 0,
                  errors::InvalidArgument("Paddings must be non-negative"));
    }

    TensorShape out_shape;
    int64_t out_size = 1;
    for (int d = 0; d < dims; ++d) {
      int64_t out_dim = input.dim_size(d) + params.pad_before[d] +
                        params.pad_after[d];
      params.in_dims[d] = input.dim_size(d);
      params.out_dims[d] = out_dim;
      out_shape.AddDim(out_dim);
      out_size *= out_dim;
    }
    params.total_out_elements = out_size;

    Tensor *output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if (out_size == 0) return;

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("MUSA stream is unavailable for Pad"));
    CallPadLauncher<T>(input.flat<T>().data(), output->flat<T>().data(), params,
                       pad_value, stream);

    musaError_t err = musaGetLastError();
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MUSA Pad kernel launch failed: ",
                                 musaGetErrorString(err)));
  }
};

#define REGISTER_MUSA_PAD_TYPE(TYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("Pad")                             \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MusaPadOp<TYPE, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Pad")                             \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int64>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MusaPadOp<TYPE, int64>);                \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                           \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings")             \
                              .HostMemory("constant_values"),     \
                          MusaPadOp<TYPE, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                           \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int64>("Tpaddings") \
                              .HostMemory("paddings")             \
                              .HostMemory("constant_values"),     \
                          MusaPadOp<TYPE, int64>);

REGISTER_MUSA_PAD_TYPE(float);
REGISTER_MUSA_PAD_TYPE(int32);
REGISTER_MUSA_PAD_TYPE(int64);
REGISTER_MUSA_PAD_TYPE(double);
REGISTER_MUSA_PAD_TYPE(uint8);
REGISTER_MUSA_PAD_TYPE(Eigen::half);
REGISTER_MUSA_PAD_TYPE(bfloat16);

#undef REGISTER_MUSA_PAD_TYPE
}  // namespace musa
}  // namespace tensorflow
