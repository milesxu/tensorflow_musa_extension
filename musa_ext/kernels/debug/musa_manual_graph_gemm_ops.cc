#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <mublas.h>
#include <musa_runtime.h>

#include <array>
#include <cstdio>
#include <vector>

extern "C" void* MusaGraphCounterTickKernelAddress();

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

class MusaManualGraphGemmOp : public OpKernel {
 public:
  explicit MusaManualGraphGemmOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    (void)ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({2, 4, 4}), &output));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    double* a_dev = nullptr;
    double* b_dev = nullptr;
    double* c_dev = nullptr;
    unsigned int* counter = nullptr;
    mublasHandle_t handle = nullptr;
    musaGraph_t graph = nullptr;
    musaGraphExec_t exec = nullptr;

    auto cleanup = [&]() {
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
      if (counter != nullptr) {
        musaFree(counter);
        counter = nullptr;
      }
      if (a_dev != nullptr) {
        musaFree(a_dev);
        a_dev = nullptr;
      }
      if (b_dev != nullptr) {
        musaFree(b_dev);
        b_dev = nullptr;
      }
      if (c_dev != nullptr) {
        musaFree(c_dev);
        c_dev = nullptr;
      }
    };
    auto fail = [&](const Status& status) {
      cleanup();
      ctx->CtxFailure(status);
    };

    const double a1[16] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    };
    const double b1[16] = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    };
    const double a2[16] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    };
    const double b2[16] = {
        3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3,
    };

    Status status =
        CheckMusa(musaMalloc(reinterpret_cast<void**>(&a_dev), sizeof(a1)),
                  "musaMalloc(a)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMalloc(reinterpret_cast<void**>(&b_dev), sizeof(b1)),
        "musaMalloc(b)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMalloc(reinterpret_cast<void**>(&c_dev), sizeof(double) * 16),
        "musaMalloc(c)");
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

    status = CheckMusa(musaMemcpy(a_dev, a1, sizeof(a1), musaMemcpyHostToDevice),
                       "musaMemcpy(a1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaMemcpy(b_dev, b1, sizeof(b1), musaMemcpyHostToDevice),
                       "musaMemcpy(b1)");
    if (!status.ok()) return fail(status);

    const double alpha = 1.0;
    const double beta = 0.0;
    status = CheckMublas(
        mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, 4, 4, 4, &alpha,
                    b_dev, 4, a_dev, 4, &beta, c_dev, 4),
        "mublasDgemm(warmup)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(warmup)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaStreamBeginCapture(stream, musaStreamCaptureModeRelaxed),
        "musaStreamBeginCapture(manual_gemm)");
    if (!status.ok()) return fail(status);
    status = CheckMublas(
        mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, 4, 4, 4, &alpha,
                    b_dev, 4, a_dev, 4, &beta, c_dev, 4),
        "mublasDgemm(captured)");
    if (!status.ok()) {
      musaGraph_t aborted = nullptr;
      musaStreamEndCapture(stream, &aborted);
      if (aborted != nullptr) musaGraphDestroy(aborted);
      return fail(status);
    }
    status = CheckMusa(musaStreamEndCapture(stream, &graph),
                       "musaStreamEndCapture(manual_gemm)");
    if (!status.ok()) return fail(status);

    size_t node_count = 0;
    status = CheckMusa(musaGraphGetNodes(graph, nullptr, &node_count),
                       "musaGraphGetNodes(gemm)");
    if (!status.ok()) return fail(status);
    std::vector<musaGraphNode_t> nodes(node_count);
    size_t fetched = node_count;
    status = CheckMusa(
        musaGraphGetNodes(graph, nodes.data(), &fetched),
        "musaGraphGetNodes(gemm,data)");
    if (!status.ok()) return fail(status);

    std::vector<musaGraphNode_t> kernel_nodes;
    for (musaGraphNode_t node : nodes) {
      musaGraphNodeType type;
      status = CheckMusa(musaGraphNodeGetType(node, &type),
                         "musaGraphNodeGetType(gemm)");
      if (!status.ok()) return fail(status);
      if (type == musaGraphNodeTypeKernel) {
        kernel_nodes.push_back(node);
      }
    }
    if (kernel_nodes.empty()) {
      return fail(errors::Internal("muBLAS capture produced no kernel node"));
    }

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
    status = CheckMusa(
        musaGraphAddKernelNode(&tick_node, graph, kernel_nodes.data(),
                               kernel_nodes.size(), &tick_params),
        "musaGraphAddKernelNode(tick)");
    if (!status.ok()) return fail(status);

    status = CheckMusa(musaGraphInstantiate(&exec, graph, 0),
                       "musaGraphInstantiate(gemm)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay1)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<double>().data(), c_dev,
                        sizeof(double) * 16, musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(output1)");
    if (!status.ok()) return fail(status);

    status = CheckMusa(musaMemcpy(a_dev, a2, sizeof(a2), musaMemcpyHostToDevice),
                       "musaMemcpy(a2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaMemcpy(b_dev, b2, sizeof(b2), musaMemcpyHostToDevice),
                       "musaMemcpy(b2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay2)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<double>().data() + 16, c_dev,
                        sizeof(double) * 16, musaMemcpyDeviceToDevice, stream),
        "musaMemcpyAsync(output2)");
    if (!status.ok()) return fail(status);

    unsigned int tick_count = 0;
    status = CheckMusa(musaMemcpy(&tick_count, counter, sizeof(tick_count),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(output)");
    if (!status.ok()) return fail(status);
    if (tick_count != 2) {
      return fail(errors::Internal("manual GEMM replay count mismatch: ",
                                   tick_count));
    }

    std::fprintf(stderr,
                 "MANUAL_GEMM_PROBE nodes=%zu gemm_kernels=%zu tick=2\n",
                 node_count, kernel_nodes.size());
    cleanup();
  }
};

}  // namespace

REGISTER_OP("MusaManualGraphGemm")
    .Input("like: T")
    .Attr("T: {float, double}")
    .Output("output: double")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({2, 4, 4}));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphGemm")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaManualGraphGemmOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphGemm")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaManualGraphGemmOp);

}  // namespace musa
}  // namespace tensorflow
