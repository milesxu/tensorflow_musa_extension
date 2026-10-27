#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

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

template <typename T>
class MusaManualGraphProbeOp : public OpKernel {
 public:
  explicit MusaManualGraphProbeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    (void)ctx->input(0);
    Tensor* diagnostics = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({5}), &diagnostics));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    unsigned int* counter1 = nullptr;
    unsigned int* counter2 = nullptr;
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
      if (counter1 != nullptr) {
        musaFree(counter1);
        counter1 = nullptr;
      }
      if (counter2 != nullptr) {
        musaFree(counter2);
        counter2 = nullptr;
      }
    };
    auto fail = [&](const Status& status) {
      cleanup();
      ctx->CtxFailure(status);
    };

    Status status =
        CheckMusa(musaMalloc(reinterpret_cast<void**>(&counter1),
                             sizeof(*counter1)),
                  "musaMalloc(counter1)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaMalloc(reinterpret_cast<void**>(&counter2),
                                  sizeof(*counter2)),
                       "musaMalloc(counter2)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaMemset(counter1, 0, sizeof(*counter1)),
                       "musaMemset(counter1)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaMemset(counter2, 0, sizeof(*counter2)),
                       "musaMemset(counter2)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(init)");
    if (!status.ok()) {
      fail(status);
      return;
    }

    status = CheckMusa(musaGraphCreate(&graph, 0), "musaGraphCreate");
    if (!status.ok()) {
      fail(status);
      return;
    }

    void* tick_kernel = MusaGraphCounterTickKernelAddress();
    unsigned int* arg1 = counter1;
    void* kernel_args1[] = {&arg1};
    musaKernelNodeParams params1{};
    params1.func = tick_kernel;
    params1.gridDim = dim3(1, 1, 1);
    params1.blockDim = dim3(1, 1, 1);
    params1.sharedMemBytes = 0;
    params1.kernelParams = kernel_args1;
    params1.extra = nullptr;

    musaGraphNode_t node1 = nullptr;
    status = CheckMusa(
        musaGraphAddKernelNode(&node1, graph, nullptr, 0, &params1),
        "musaGraphAddKernelNode(node1)");
    if (!status.ok()) {
      fail(status);
      return;
    }

    unsigned int* arg2 = counter2;
    void* kernel_args2[] = {&arg2};
    musaKernelNodeParams params2 = params1;
    params2.kernelParams = kernel_args2;
    musaGraphNode_t node2 = nullptr;
    musaGraphNode_t dependencies[] = {node1};
    status = CheckMusa(
        musaGraphAddKernelNode(&node2, graph, dependencies, 1, &params2),
        "musaGraphAddKernelNode(node2)");
    if (!status.ok()) {
      fail(status);
      return;
    }

    size_t node_count = 0;
    status = CheckMusa(musaGraphGetNodes(graph, nullptr, &node_count),
                       "musaGraphGetNodes");
    if (!status.ok()) {
      fail(status);
      return;
    }
    std::vector<musaGraphNode_t> nodes(node_count);
    size_t fetched = node_count;
    status = CheckMusa(
        musaGraphGetNodes(graph, nodes.data(), &fetched),
        "musaGraphGetNodes(data)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    size_t kernel_count = 0;
    for (musaGraphNode_t node : nodes) {
      musaGraphNodeType type;
      status = CheckMusa(musaGraphNodeGetType(node, &type),
                         "musaGraphNodeGetType");
      if (!status.ok()) {
        fail(status);
        return;
      }
      if (type == musaGraphNodeTypeKernel) {
        ++kernel_count;
      }
    }

    status = CheckMusa(musaGraphInstantiate(&exec, graph, 0),
                       "musaGraphInstantiate");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(1)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay1)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch(2)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(replay2)");
    if (!status.ok()) {
      fail(status);
      return;
    }

    unsigned int value1 = 0;
    unsigned int value2 = 0;
    status = CheckMusa(musaMemcpy(&value1, counter1, sizeof(value1),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter1)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaMemcpy(&value2, counter2, sizeof(value2),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter2)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    if (value1 != 2 || value2 != 2) {
      fail(errors::Internal("manual graph replay mismatch: counter1=",
                            value1, " counter2=", value2));
      return;
    }
    if (node_count != 2 || kernel_count != 2) {
      fail(errors::Internal("manual graph node mismatch: nodes=",
                            node_count, " kernels=", kernel_count));
      return;
    }

    std::fprintf(stderr,
                 "MANUAL_GRAPH_PROBE nodes=%zu kernels=%zu replay=2 "
                 "counter1=%u counter2=%u\n",
                 node_count, kernel_count, value1, value2);
    std::array<int32, 5> values = {
        static_cast<int32>(node_count),
        static_cast<int32>(kernel_count),
        static_cast<int32>(value1),
        static_cast<int32>(value2),
        2,
    };
    status = CheckMusa(
        musaMemcpy(diagnostics->flat<int32>().data(), values.data(),
                   sizeof(values), musaMemcpyHostToDevice),
        "musaMemcpy(diagnostics)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaStreamSynchronize(stream),
                       "musaStreamSynchronize(diagnostics)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    cleanup();
  }
};

}  // namespace

REGISTER_OP("MusaManualGraphProbe")
    .Input("like: T")
    .Attr("T: {float, double}")
    .Output("diagnostics: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(5));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphProbe")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaManualGraphProbeOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphProbe")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaManualGraphProbeOp<double>);

}  // namespace musa
}  // namespace tensorflow
