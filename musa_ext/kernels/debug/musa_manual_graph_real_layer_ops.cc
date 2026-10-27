#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <mublas.h>
#include <musa_runtime.h>

#include <cstdio>
#include <vector>

extern "C" void* MusaGraphCounterTickKernelAddress();
extern "C" void* MusaManualGraphBiasTanhFloatKernelAddress();

namespace tensorflow {
namespace musa {
namespace {

constexpr int kM = 1536;
constexpr int kK = 1600;
constexpr int kN = 240;
constexpr int kElements = kM * kN;

Status CheckMusa(musaError_t status, const char* call) {
  if (status == musaSuccess) {
    return OkStatus();
  }
  return errors::Internal(call, " failed with MUSA error ",
                          static_cast<int>(status));
}

Status CheckMublas(mublasStatus_t status, const char* call) {
  if (status == MUBLAS_STATUS_SUCCESS) {
    return OkStatus();
  }
  return errors::Internal(call, " failed with muBLAS status ",
                          static_cast<int>(status));
}

class MusaManualGraphRealLayerOp : public OpKernel {
 public:
  explicit MusaManualGraphRealLayerOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& matrix = ctx->input(1);
    const Tensor& bias = ctx->input(2);
    OP_REQUIRES(ctx, input.dtype() == DT_FLOAT &&
                         matrix.dtype() == DT_FLOAT &&
                         bias.dtype() == DT_FLOAT,
                errors::InvalidArgument("real layer inputs must be float"));
    OP_REQUIRES(ctx, input.shape() == TensorShape({kM, kK}),
                errors::InvalidArgument("input shape mismatch"));
    OP_REQUIRES(ctx, matrix.shape() == TensorShape({kK, kN}),
                errors::InvalidArgument("matrix shape mismatch"));
    OP_REQUIRES(ctx, bias.shape() == TensorShape({kN}),
                errors::InvalidArgument("bias shape mismatch"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({4, kM, kN}), &output));
    musaStream_t stream = GetMusaStreamByCtx(ctx);

    float* input_dev = nullptr;
    float* matrix_dev = nullptr;
    float* bias_dev = nullptr;
    float* gemm_dev = nullptr;
    float* pre_dev = nullptr;
    float* output_dev = nullptr;
    unsigned int* counter = nullptr;
    mublasHandle_t handle = nullptr;
    musaGraph_t graph = nullptr;
    musaGraphExec_t exec = nullptr;

    auto cleanup = [&]() {
      if (stream != nullptr) {
        musaStreamCaptureStatus capture_status = musaStreamCaptureStatusNone;
        if (musaStreamIsCapturing(stream, &capture_status) == musaSuccess &&
            capture_status == musaStreamCaptureStatusActive) {
          musaGraph_t aborted = nullptr;
          musaStreamEndCapture(stream, &aborted);
          if (aborted != nullptr) {
            musaGraphDestroy(aborted);
          }
        }
        musaStreamSynchronize(stream);
      }
      if (exec != nullptr) {
        musaGraphExecDestroy(exec);
        exec = nullptr;
      }
      if (graph != nullptr) {
        musaGraphDestroy(graph);
        graph = nullptr;
      }
      if (handle != nullptr) {
        mublasDestroy(handle);
        handle = nullptr;
      }
      auto free_buffer = [](void** ptr) {
        if (*ptr != nullptr) {
          musaFree(*ptr);
          *ptr = nullptr;
        }
      };
      free_buffer(reinterpret_cast<void**>(&input_dev));
      free_buffer(reinterpret_cast<void**>(&matrix_dev));
      free_buffer(reinterpret_cast<void**>(&bias_dev));
      free_buffer(reinterpret_cast<void**>(&gemm_dev));
      free_buffer(reinterpret_cast<void**>(&pre_dev));
      free_buffer(reinterpret_cast<void**>(&output_dev));
      free_buffer(reinterpret_cast<void**>(&counter));
    };
    auto fail = [&](const Status& status) {
      cleanup();
      ctx->CtxFailure(status);
    };
    auto alloc = [&](float** ptr, size_t bytes, const char* name) {
      return CheckMusa(musaMalloc(reinterpret_cast<void**>(ptr), bytes), name);
    };

    Status status =
        alloc(&input_dev, input.TotalBytes(), "musaMalloc(input)");
    if (!status.ok()) return fail(status);
    status = alloc(&matrix_dev, matrix.TotalBytes(), "musaMalloc(matrix)");
    if (!status.ok()) return fail(status);
    status = alloc(&bias_dev, bias.TotalBytes(), "musaMalloc(bias)");
    if (!status.ok()) return fail(status);
    status = alloc(&gemm_dev, sizeof(float) * kElements,
                   "musaMalloc(gemm)");
    if (!status.ok()) return fail(status);
    status = alloc(&pre_dev, sizeof(float) * kElements,
                   "musaMalloc(pre)");
    if (!status.ok()) return fail(status);
    status = alloc(&output_dev, sizeof(float) * kElements,
                   "musaMalloc(output)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaMalloc(reinterpret_cast<void**>(&counter),
                                  sizeof(*counter)),
                       "musaMalloc(counter)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaMemset(counter, 0, sizeof(*counter)),
                       "musaMemset(counter)");
    if (!status.ok()) return fail(status);

    status = CheckMublas(mublasCreate(&handle), "mublasCreate");
    if (!status.ok()) return fail(status);
    status = CheckMublas(mublasSetStream(handle, stream), "mublasSetStream");
    if (!status.ok()) return fail(status);

    status = CheckMusa(
        musaMemcpyAsync(input_dev, input.flat<float>().data(),
                        input.TotalBytes(), musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(input)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(matrix_dev, matrix.flat<float>().data(),
                        matrix.TotalBytes(), musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(matrix)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(bias_dev, bias.flat<float>().data(), bias.TotalBytes(),
                        musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(bias)");
    if (!status.ok()) return fail(status);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto run_gemm = [&]() {
      return CheckMublas(
          mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, kN, kM, kK, &alpha,
                      matrix_dev, kN, input_dev, kK, &beta, gemm_dev, kN),
          "mublasSgemm(real_layer)");
    };
    status = run_gemm();
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(warmup)");
    if (!status.ok()) return fail(status);

    status = CheckMusa(
        musaStreamBeginCapture(stream, musaStreamCaptureModeRelaxed),
        "musaStreamBeginCapture(real_layer)");
    if (!status.ok()) return fail(status);
    status = run_gemm();
    if (!status.ok()) {
      musaGraph_t aborted = nullptr;
      musaStreamEndCapture(stream, &aborted);
      if (aborted != nullptr) musaGraphDestroy(aborted);
      return fail(status);
    }
    status = CheckMusa(musaStreamEndCapture(stream, &graph),
                       "musaStreamEndCapture(real_layer)");
    if (!status.ok()) return fail(status);

    size_t captured_nodes = 0;
    status = CheckMusa(musaGraphGetNodes(graph, nullptr, &captured_nodes),
                       "musaGraphGetNodes(captured)");
    if (!status.ok()) return fail(status);
    std::vector<musaGraphNode_t> captured(captured_nodes);
    size_t captured_fetched = captured_nodes;
    status = CheckMusa(
        musaGraphGetNodes(graph, captured.data(), &captured_fetched),
        "musaGraphGetNodes(captured,data)");
    if (!status.ok()) return fail(status);
    std::vector<musaGraphNode_t> gemm_kernels;
    for (musaGraphNode_t node : captured) {
      musaGraphNodeType type;
      status = CheckMusa(musaGraphNodeGetType(node, &type),
                         "musaGraphNodeGetType(captured)");
      if (!status.ok()) return fail(status);
      if (type == musaGraphNodeTypeKernel) {
        gemm_kernels.push_back(node);
      }
    }
    if (gemm_kernels.empty()) {
      return fail(errors::Internal("real layer GEMM produced no kernels"));
    }

    void* bias_tanh_kernel = MusaManualGraphBiasTanhFloatKernelAddress();
    void* tick_kernel = MusaGraphCounterTickKernelAddress();
    if (bias_tanh_kernel == nullptr || tick_kernel == nullptr) {
      return fail(errors::Internal("real layer kernel address is null"));
    }
    const float* bt_input = gemm_dev;
    const float* bt_bias = bias_dev;
    float* bt_pre = pre_dev;
    float* bt_output = output_dev;
    int elements = kElements;
    int width = kN;
    void* bt_args[] = {
        &bt_input, &bt_bias, &bt_pre, &bt_output, &elements, &width,
    };
    musaKernelNodeParams bt_params{};
    bt_params.func = bias_tanh_kernel;
    bt_params.gridDim = dim3((kElements + 255) / 256, 1, 1);
    bt_params.blockDim = dim3(256, 1, 1);
    bt_params.sharedMemBytes = 0;
    bt_params.kernelParams = bt_args;
    bt_params.extra = nullptr;
    musaGraphNode_t bt_node = nullptr;
    status = CheckMusa(
        musaGraphAddKernelNode(&bt_node, graph, gemm_kernels.data(),
                               gemm_kernels.size(), &bt_params),
        "musaGraphAddKernelNode(bias_tanh)");
    if (!status.ok()) return fail(status);

    unsigned int* tick_arg = counter;
    void* tick_args[] = {&tick_arg};
    musaKernelNodeParams tick_params{};
    tick_params.func = tick_kernel;
    tick_params.gridDim = dim3(1, 1, 1);
    tick_params.blockDim = dim3(1, 1, 1);
    tick_params.sharedMemBytes = 0;
    tick_params.kernelParams = tick_args;
    tick_params.extra = nullptr;
    musaGraphNode_t tick_node = nullptr;
    musaGraphNode_t tick_deps[] = {bt_node};
    status = CheckMusa(
        musaGraphAddKernelNode(&tick_node, graph, tick_deps, 1, &tick_params),
        "musaGraphAddKernelNode(tick)");
    if (!status.ok()) return fail(status);

    status = CheckMusa(musaGraphInstantiate(&exec, graph, 0),
                       "musaGraphInstantiate(real_layer)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<float>().data(), pre_dev,
                        sizeof(float) * kElements, musaMemcpyDeviceToDevice,
                        stream),
        "musaMemcpyAsync(pre1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<float>().data() + kElements, output_dev,
                        sizeof(float) * kElements, musaMemcpyDeviceToDevice,
                        stream),
        "musaMemcpyAsync(output1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<float>().data() + 2 * kElements, pre_dev,
                        sizeof(float) * kElements, musaMemcpyDeviceToDevice,
                        stream),
        "musaMemcpyAsync(pre2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<float>().data() + 3 * kElements,
                        output_dev,
                        sizeof(float) * kElements, musaMemcpyDeviceToDevice,
                        stream),
        "musaMemcpyAsync(real_layer_output)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(output)");
    if (!status.ok()) return fail(status);

    unsigned int tick_count = 0;
    status = CheckMusa(musaMemcpy(&tick_count, counter, sizeof(tick_count),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter)");
    if (!status.ok()) return fail(status);
    if (tick_count != 2) {
      return fail(errors::Internal("real layer replay count mismatch: ",
                                   tick_count));
    }
    std::fprintf(stderr,
                 "MANUAL_REAL_LAYER_PROBE captured_nodes=%zu gemm_kernels=%zu "
                 "added_nodes=2 tick=2 shape=%dx%d\n",
                 captured_nodes, gemm_kernels.size(), kM, kN);
    cleanup();
  }
};

}  // namespace

REGISTER_OP("MusaManualGraphRealLayer")
    .Input("input: float")
    .Input("matrix: float")
    .Input("bias: float")
    .Output("output: float")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({4, kM, kN}));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphRealLayer").Device("MUSA"),
    MusaManualGraphRealLayerOp);

}  // namespace musa
}  // namespace tensorflow
