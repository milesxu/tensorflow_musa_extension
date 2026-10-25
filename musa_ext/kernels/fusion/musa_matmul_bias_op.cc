#include <mublas.h>

#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

#include "../utils_op.h"
#include "musa_bias_add_double.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace musa {

namespace {

inline bool ResolveTF32Enabled() {
  const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
  if (tf32_env == nullptr) {
    return true;
  }
  return std::atoi(tf32_env) != 0;
}

}  // namespace

Status RunDoubleMatMulBiasAdd(OpKernelContext* ctx, const Tensor& a,
                              const Tensor& b, const Tensor& bias, int64_t m,
                              int64_t k, int64_t n, bool transpose_a,
                              bool transpose_b, Tensor* output) {
  if (m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max()) {
    return errors::InvalidArgument(
        "MUSA double MatMulBiasAdd dimensions exceed int range");
  }

  auto runtime = QueryMusaKernelRuntimeView(ctx);
  if (runtime.mublas_handle == nullptr) {
    return errors::Internal("MUSA mublas handle is unavailable");
  }

  const int lda = static_cast<int>(transpose_a ? m : k);
  const int ldb = static_cast<int>(transpose_b ? k : n);
  const int ldc = static_cast<int>(n);
  const mublasOperation_t op_first =
      transpose_b ? MUBLAS_OP_T : MUBLAS_OP_N;
  const mublasOperation_t op_second =
      transpose_a ? MUBLAS_OP_T : MUBLAS_OP_N;
  const double alpha = 1.0;
  const double beta = 0.0;

  mublasStatus_t status = mublasDgemm(
      runtime.mublas_handle, op_first, op_second, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(k), &alpha,
      b.flat<double>().data(), ldb, a.flat<double>().data(), lda, &beta,
      output->flat<double>().data(), ldc);
  if (status != MUBLAS_STATUS_SUCCESS) {
    return errors::Internal("MUSA mublasDgemm for MatMulBiasAdd failed. "
                            "Status: ",
                            static_cast<int>(status));
  }

  const double* bias_ptr = bias.flat<double>().data();
  double* output_ptr = output->flat<double>().data();
  const musaError_t launch_status =
      LaunchMusaBiasAddDouble(output_ptr, bias_ptr, m, n, runtime.stream);
  if (launch_status != musaSuccess) {
    return errors::Internal("MUSA BiasAddDouble kernel launch failed. "
                            "Status: ",
                            static_cast<int>(launch_status));
  }
  return OkStatus();
}

template <typename T>
class MusaMatMulBiasAddOp : public MusaOpKernel {
 public:
  explicit MusaMatMulBiasAddOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

    static const bool tf32_enabled_global = ResolveTF32Enabled();
    tf32_enabled_ = tf32_enabled_global;
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& bias = ctx->input(2);

    OP_REQUIRES(ctx, a.dims() == 2,
                errors::InvalidArgument(
                    "MatMulBiasAdd requires input a to be 2D, got shape ",
                    a.shape().DebugString()));
    OP_REQUIRES(ctx, b.dims() == 2,
                errors::InvalidArgument(
                    "MatMulBiasAdd requires input b to be 2D, got shape ",
                    b.shape().DebugString()));
    OP_REQUIRES(
        ctx, bias.dims() == 1 || (bias.dims() == 2 && bias.dim_size(0) == 1),
        errors::InvalidArgument(
            "MatMulBiasAdd requires bias to be 1D or [1, N], got shape ",
            bias.shape().DebugString()));

    if (a.NumElements() == 0 || b.NumElements() == 0 ||
        bias.NumElements() == 0) {
      TensorShape out_shape;
      OP_REQUIRES_OK(ctx, ComputeOutputShape(a, b, &out_shape));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
      return;
    }

    const int64_t a_rows = a.dim_size(0);
    const int64_t a_cols = a.dim_size(1);
    const int64_t b_rows = b.dim_size(0);
    const int64_t b_cols = b.dim_size(1);

    const int64_t m = transpose_a_ ? a_cols : a_rows;
    const int64_t k_a = transpose_a_ ? a_rows : a_cols;
    const int64_t k_b = transpose_b_ ? b_cols : b_rows;
    const int64_t n = transpose_b_ ? b_rows : b_cols;

    OP_REQUIRES(ctx, k_a == k_b,
                errors::InvalidArgument("Matrix size-incompatible: a shape ",
                                        a.shape().DebugString(), ", b shape ",
                                        b.shape().DebugString(),
                                        ", transpose_a=", transpose_a_,
                                        ", transpose_b=", transpose_b_));

    OP_REQUIRES(
        ctx, IsSupportedBiasShape(bias, n),
        errors::InvalidArgument("Bias dimension mismatch: bias shape ",
                                bias.shape().DebugString(), ", expected [", n,
                                "] or [1, ", n, "]"));

    TensorShape out_shape({m, n});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    if constexpr (std::is_same<T, double>::value) {
      OP_REQUIRES_OK(ctx, RunDoubleMatMulBiasAdd(
                              ctx, a, b, bias, m, k_a, n, transpose_a_,
                              transpose_b_, output));
      return;
    }

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    mTensor mt_a = CreateMTensor(a, format_);
    mTensor mt_b = CreateMTensor(b, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    if (bias.dims() == 2) {
      std::vector<int64_t> bias_dims = {BiasSize(bias)};
      std::vector<int64_t> bias_strides = {1};
      auto status = mt_bias.SetNdInfo(1, bias_dims.data(), bias_strides.data());
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("muDNN bias SetNdInfo failed, status=",
                                   static_cast<int>(status)));
    }

    ::musa::dnn::MatMul op;
    auto status = op.SetTranspose(transpose_a_, transpose_b_);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetTranspose failed, status=",
                                 static_cast<int>(status)));

    status = op.SetAlpha(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetAlpha failed, status=",
                                 static_cast<int>(status)));

    status = op.SetBeta(0.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetBeta failed, status=",
                                 static_cast<int>(status)));

    status = op.SetGamma(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetGamma failed, status=",
                                 static_cast<int>(status)));

    status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMulBiasAdd failed, status=",
                                 static_cast<int>(status)));
  }

 private:
  bool IsSupportedBiasShape(const Tensor& bias, int64_t n) const {
    if (bias.dims() == 1) {
      return bias.dim_size(0) == n;
    }
    return bias.dims() == 2 && bias.dim_size(0) == 1 && bias.dim_size(1) == n;
  }

  int64_t BiasSize(const Tensor& bias) const {
    return bias.dims() == 1 ? bias.dim_size(0) : bias.dim_size(1);
  }

  Status ComputeOutputShape(const Tensor& a, const Tensor& b,
                            TensorShape* out_shape) {
    const int64_t a_rows = a.dim_size(0);
    const int64_t a_cols = a.dim_size(1);
    const int64_t b_rows = b.dim_size(0);
    const int64_t b_cols = b.dim_size(1);

    const int64_t m = transpose_a_ ? a_cols : a_rows;
    const int64_t k_a = transpose_a_ ? a_rows : a_cols;
    const int64_t k_b = transpose_b_ ? b_cols : b_rows;
    const int64_t n = transpose_b_ ? b_rows : b_cols;

    if (k_a != k_b) {
      return errors::InvalidArgument(
          "Matrix size-incompatible: a shape ", a.shape().DebugString(),
          ", b shape ", b.shape().DebugString(), ", transpose_a=", transpose_a_,
          ", transpose_b=", transpose_b_);
    }

    *out_shape = TensorShape({m, n});
    return OkStatus();
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool tf32_enabled_ = false;
};

#define REGISTER_MUSA_MATMUL_BIASADD(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MusaMatMulBiasAdd").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMatMulBiasAddOp<TYPE>);

REGISTER_MUSA_MATMUL_BIASADD(float);
REGISTER_MUSA_MATMUL_BIASADD(double);
REGISTER_MUSA_MATMUL_BIASADD(Eigen::half);
REGISTER_MUSA_MATMUL_BIASADD(bfloat16);

#undef REGISTER_MUSA_MATMUL_BIASADD

}  // namespace musa

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("MusaMatMulBiasAdd")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);
}  // namespace tensorflow
