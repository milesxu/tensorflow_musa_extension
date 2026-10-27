#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <mublas.h>
#include <musa_runtime.h>

#include <cstdio>
#include <vector>

extern "C" void* MusaGraphCounterTickKernelAddress();
extern "C" void* MusaManualGraphReshapeKernelAddress();
extern "C" void* MusaManualGraphTanhKernelAddress();
extern "C" void* MusaManualGraphSumKernelAddress();

namespace tensorflow {
namespace musa {
namespace {

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

class MusaManualGraphFittingOp : public OpKernel {
 public:
  explicit MusaManualGraphFittingOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    (void)ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({2, 17}), &output));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    double* a_dev = nullptr;
    double* b_dev = nullptr;
    double* gemm_dev = nullptr;
    double* reshape_dev = nullptr;
    double* tanh_dev = nullptr;
    double* sum_dev = nullptr;
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
      free_buffer(reinterpret_cast<void**>(&counter));
      free_buffer(reinterpret_cast<void**>(&a_dev));
      free_buffer(reinterpret_cast<void**>(&b_dev));
      free_buffer(reinterpret_cast<void**>(&gemm_dev));
      free_buffer(reinterpret_cast<void**>(&reshape_dev));
      free_buffer(reinterpret_cast<void**>(&tanh_dev));
      free_buffer(reinterpret_cast<void**>(&sum_dev));
    };
    auto fail = [&](const Status& status) {
      cleanup();
      ctx->CtxFailure(status);
    };
    auto allocate = [&](double** ptr, size_t bytes, const char* name) {
      return CheckMusa(
          musaMalloc(reinterpret_cast<void**>(ptr), bytes), name);
    };

    const double a1[16] = {
        0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
        0.9, -1.0, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7,
    };
    const double b1[16] = {
        0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9,
        -0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
    };
    const double a2[16] = {
        -0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -0.1,
        0.2, -0.4, 0.5, -0.6, 0.8, -0.9, 0.1, -0.2,
    };
    const double b2[16] = {
        0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1, -0.2,
        0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -0.1,
    };

    Status status = allocate(&a_dev, sizeof(a1), "musaMalloc(a)");
    if (!status.ok()) return fail(status);
    status = allocate(&b_dev, sizeof(b1), "musaMalloc(b)");
    if (!status.ok()) return fail(status);
    status = allocate(&gemm_dev, sizeof(double) * 16, "musaMalloc(gemm)");
    if (!status.ok()) return fail(status);
    status = allocate(&reshape_dev, sizeof(double) * 16, "musaMalloc(reshape)");
    if (!status.ok()) return fail(status);
    status = allocate(&tanh_dev, sizeof(double) * 16, "musaMalloc(tanh)");
    if (!status.ok()) return fail(status);
    status = allocate(&sum_dev, sizeof(double), "musaMalloc(sum)");
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
        musaMemcpyAsync(a_dev, a1, sizeof(a1), musaMemcpyHostToDevice, stream),
        "musaMemcpyAsync(a1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(b_dev, b1, sizeof(b1), musaMemcpyHostToDevice, stream),
        "musaMemcpyAsync(b1)");
    if (!status.ok()) return fail(status);

    const double alpha = 1.0;
    const double beta = 0.0;
    status = CheckMublas(
        mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, 4, 4, 4, &alpha,
                    b_dev, 4, a_dev, 4, &beta, gemm_dev, 4),
        "mublasDgemm(warmup)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(warmup)");
    if (!status.ok()) return fail(status);

    status = CheckMusa(
        musaStreamBeginCapture(stream, musaStreamCaptureModeRelaxed),
        "musaStreamBeginCapture(fitting)");
    if (!status.ok()) return fail(status);
    status = CheckMublas(
        mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, 4, 4, 4, &alpha,
                    b_dev, 4, a_dev, 4, &beta, gemm_dev, 4),
        "mublasDgemm(fitting)");
    if (!status.ok()) {
      musaGraph_t aborted = nullptr;
      musaStreamEndCapture(stream, &aborted);
      if (aborted != nullptr) musaGraphDestroy(aborted);
      return fail(status);
    }
    status = CheckMusa(musaStreamEndCapture(stream, &graph),
                       "musaStreamEndCapture(fitting)");
    if (!status.ok()) return fail(status);

    size_t node_count = 0;
    status = CheckMusa(musaGraphGetNodes(graph, nullptr, &node_count),
                       "musaGraphGetNodes(fitting)");
    if (!status.ok()) return fail(status);
    if (node_count != 1) {
      return fail(errors::Internal("unexpected fitting GEMM node count: ",
                                   node_count));
    }
    std::vector<musaGraphNode_t> nodes(node_count);
    size_t fetched = node_count;
    status = CheckMusa(
        musaGraphGetNodes(graph, nodes.data(), &fetched),
        "musaGraphGetNodes(fitting,data)");
    if (!status.ok()) return fail(status);
    musaGraphNode_t gemm_node = nodes[0];
    musaGraphNodeType gemm_type;
    status = CheckMusa(musaGraphNodeGetType(gemm_node, &gemm_type),
                       "musaGraphNodeGetType(gemm)");
    if (!status.ok()) return fail(status);
    if (gemm_type != musaGraphNodeTypeKernel) {
      return fail(errors::Internal("captured fitting node is not a kernel"));
    }

    const double* reshape_in = gemm_dev;
    double* reshape_out = reshape_dev;
    void* reshape_args[] = {&reshape_in, &reshape_out};
    musaKernelNodeParams reshape_params{};
    reshape_params.func = MusaManualGraphReshapeKernelAddress();
    if (reshape_params.func == nullptr ||
        MusaManualGraphTanhKernelAddress() == nullptr ||
        MusaManualGraphSumKernelAddress() == nullptr ||
        MusaGraphCounterTickKernelAddress() == nullptr) {
      return fail(errors::Internal("manual fitting kernel address is null"));
    }
    reshape_params.gridDim = dim3(1, 1, 1);
    reshape_params.blockDim = dim3(16, 1, 1);
    reshape_params.sharedMemBytes = 0;
    reshape_params.kernelParams = reshape_args;
    reshape_params.extra = nullptr;
    musaGraphNode_t reshape_node = nullptr;
    musaGraphNode_t reshape_deps[] = {gemm_node};
    status = CheckMusa(
        musaGraphAddKernelNode(&reshape_node, graph, reshape_deps, 1,
                               &reshape_params),
        "musaGraphAddKernelNode(reshape)");
    if (!status.ok()) return fail(status);

    const double* tanh_in = reshape_dev;
    double* tanh_out = tanh_dev;
    void* tanh_args[] = {&tanh_in, &tanh_out};
    musaKernelNodeParams tanh_params{};
    tanh_params.func = MusaManualGraphTanhKernelAddress();
    tanh_params.gridDim = dim3(1, 1, 1);
    tanh_params.blockDim = dim3(16, 1, 1);
    tanh_params.sharedMemBytes = 0;
    tanh_params.kernelParams = tanh_args;
    tanh_params.extra = nullptr;
    musaGraphNode_t tanh_node = nullptr;
    musaGraphNode_t tanh_deps[] = {reshape_node};
    status = CheckMusa(
        musaGraphAddKernelNode(&tanh_node, graph, tanh_deps, 1, &tanh_params),
        "musaGraphAddKernelNode(tanh)");
    if (!status.ok()) return fail(status);

    const double* sum_in = tanh_dev;
    double* sum_out = sum_dev;
    void* sum_args[] = {&sum_in, &sum_out};
    musaKernelNodeParams sum_params{};
    sum_params.func = MusaManualGraphSumKernelAddress();
    sum_params.gridDim = dim3(1, 1, 1);
    sum_params.blockDim = dim3(1, 1, 1);
    sum_params.sharedMemBytes = 0;
    sum_params.kernelParams = sum_args;
    sum_params.extra = nullptr;
    musaGraphNode_t sum_node = nullptr;
    musaGraphNode_t sum_deps[] = {tanh_node};
    status = CheckMusa(
        musaGraphAddKernelNode(&sum_node, graph, sum_deps, 1, &sum_params),
        "musaGraphAddKernelNode(sum)");
    if (!status.ok()) return fail(status);

    unsigned int* tick_arg = counter;
    void* tick_args[] = {&tick_arg};
    musaKernelNodeParams tick_params{};
    tick_params.func = MusaGraphCounterTickKernelAddress();
    tick_params.gridDim = dim3(1, 1, 1);
    tick_params.blockDim = dim3(1, 1, 1);
    tick_params.sharedMemBytes = 0;
    tick_params.kernelParams = tick_args;
    tick_params.extra = nullptr;
    musaGraphNode_t tick_node = nullptr;
    musaGraphNode_t tick_deps[] = {sum_node};
    status = CheckMusa(
        musaGraphAddKernelNode(&tick_node, graph, tick_deps, 1, &tick_params),
        "musaGraphAddKernelNode(tick)");
    if (!status.ok()) return fail(status);

    size_t final_node_count = 0;
    status = CheckMusa(musaGraphGetNodes(graph, nullptr, &final_node_count),
                       "musaGraphGetNodes(final)");
    if (!status.ok()) return fail(status);
    std::vector<musaGraphNode_t> final_nodes(final_node_count);
    size_t final_fetched = final_node_count;
    status = CheckMusa(
        musaGraphGetNodes(graph, final_nodes.data(), &final_fetched),
        "musaGraphGetNodes(final,data)");
    if (!status.ok()) return fail(status);
    size_t final_kernel_count = 0;
    for (musaGraphNode_t node : final_nodes) {
      musaGraphNodeType type;
      status = CheckMusa(musaGraphNodeGetType(node, &type),
                         "musaGraphNodeGetType(final)");
      if (!status.ok()) return fail(status);
      if (type == musaGraphNodeTypeKernel) {
        ++final_kernel_count;
      }
    }
    if (final_node_count != 5 || final_kernel_count != 5) {
      return fail(errors::Internal(
          "unexpected fitting graph node count: nodes=", final_node_count,
          " kernels=", final_kernel_count));
    }

    status = CheckMusa(musaGraphInstantiate(&exec, graph, 0),
                       "musaGraphInstantiate(fitting)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay1)");
    if (!status.ok()) return fail(status);
    unsigned int tick_after_first = 0;
    status = CheckMusa(musaMemcpy(&tick_after_first, counter,
                                  sizeof(tick_after_first),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_after_first)");
    if (!status.ok()) return fail(status);
    if (tick_after_first != 1) {
      return fail(errors::Internal("first fitting replay count mismatch: ",
                                   tick_after_first));
    }
    status = CheckMusa(
        musaMemcpyAsync(output->flat<double>().data(), tanh_dev,
                        sizeof(double) * 16, musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(output_tanh1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<double>().data() + 16, sum_dev,
                        sizeof(double), musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(output_sum1)");
    if (!status.ok()) return fail(status);

    status = CheckMusa(
        musaMemcpyAsync(a_dev, a2, sizeof(a2), musaMemcpyHostToDevice, stream),
        "musaMemcpyAsync(a2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(b_dev, b2, sizeof(b2), musaMemcpyHostToDevice, stream),
        "musaMemcpyAsync(b2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<double>().data() + 17, tanh_dev,
                        sizeof(double) * 16, musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(output_tanh2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<double>().data() + 33, sum_dev,
                        sizeof(double), musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(output_sum2)");
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
      return fail(errors::Internal("fitting replay count mismatch: ",
                                   tick_count));
    }
    std::fprintf(stderr,
                 "MANUAL_FITTING_PROBE nodes=%zu kernels=%zu gemm_nodes=1 "
                 "added_nodes=4 tick=2 tanh=16 sum=1\n",
                 final_node_count, final_kernel_count);
    cleanup();
  }
};

}  // namespace

REGISTER_OP("MusaManualGraphFitting")
    .Input("like: T")
    .Attr("T: {float, double}")
    .Output("output: double")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({2, 17}));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphFitting")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaManualGraphFittingOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphFitting")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaManualGraphFittingOp);

}  // namespace musa
}  // namespace tensorflow
