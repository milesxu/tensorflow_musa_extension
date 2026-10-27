#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <musa_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" void LaunchMusaGraphCounterTick(unsigned int* counter,
                                           musaStream_t stream);

namespace tensorflow {
namespace musa {
namespace {

struct CapturedGraph {
  musaGraph_t graph = nullptr;
  musaGraphExec_t exec = nullptr;
  unsigned int* counter = nullptr;
  size_t node_count = 0;
  size_t kernel_nodes = 0;
};

std::mutex& CapturedGraphMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<int64_t, CapturedGraph>& CapturedGraphs() {
  static std::unordered_map<int64_t, CapturedGraph> graphs;
  return graphs;
}

Status CheckMusa(musaError_t status, const char* call) {
  if (status == musaSuccess) {
    return OkStatus();
  }
  return errors::Internal(call, " failed with MUSA error ",
                          static_cast<int>(status));
}

double ProbeTimestamp() {
  static const auto start = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
      .count();
}

Status InspectGraph(musaGraph_t graph, size_t* node_count,
                    size_t* kernel_nodes) {
  size_t count = 0;
  TF_RETURN_IF_ERROR(
      CheckMusa(musaGraphGetNodes(graph, nullptr, &count), "musaGraphGetNodes"));
  if (count == 0) {
    return errors::Internal("captured graph has no nodes");
  }

  std::vector<musaGraphNode_t> nodes(count);
  size_t fetched = count;
  TF_RETURN_IF_ERROR(CheckMusa(musaGraphGetNodes(graph, nodes.data(), &fetched),
                               "musaGraphGetNodes(data)"));
  if (fetched != count) {
    return errors::Internal("musaGraphGetNodes returned ", fetched,
                            " nodes, expected ", count);
  }

  size_t kernels = 0;
  for (musaGraphNode_t node : nodes) {
    musaGraphNodeType type;
    TF_RETURN_IF_ERROR(CheckMusa(musaGraphNodeGetType(node, &type),
                                 "musaGraphNodeGetType"));
    if (type == musaGraphNodeTypeKernel) {
      ++kernels;
    }
  }

  *node_count = count;
  *kernel_nodes = kernels;
  return OkStatus();
}

class MusaGraphCaptureBeginOp : public OpKernel {
 public:
  explicit MusaGraphCaptureBeginOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING begin_enter=%.6f\n",
                 ProbeTimestamp());
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    {
      std::lock_guard<std::mutex> lock(CapturedGraphMutex());
      OP_REQUIRES(
          ctx, CapturedGraphs().find(handle_) == CapturedGraphs().end(),
          errors::AlreadyExists("graph handle already allocated: ", handle_));
    }

    unsigned int* counter = nullptr;
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMalloc(reinterpret_cast<void**>(&counter),
                                  sizeof(*counter)),
                       "musaMalloc(counter)"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaMemset(counter, 0, sizeof(*counter)),
                                  "musaMemset(counter)"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(stream),
                                  "musaStreamSynchronize(counter)"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING begin_counter_ready=%.6f\n",
                 ProbeTimestamp());

    {
      std::lock_guard<std::mutex> lock(CapturedGraphMutex());
      CapturedGraph state;
      state.counter = counter;
      CapturedGraphs().emplace(handle_, state);
    }

    OP_REQUIRES_OK(
        ctx, CheckMusa(musaStreamBeginCapture(
                          stream, musaStreamCaptureModeRelaxed),
                       "musaStreamBeginCapture"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING begin_capture=%.6f\n",
                 ProbeTimestamp());
  }

 private:
  int64_t handle_;
};

class MusaGraphCaptureEndOp : public OpKernel {
 public:
  explicit MusaGraphCaptureEndOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING end_enter=%.6f\n",
                 ProbeTimestamp());
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));
    (void)ctx->input(0);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaGraph_t graph = nullptr;
    musaGraphExec_t exec = nullptr;
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamEndCapture(stream, &graph),
                                  "musaStreamEndCapture"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING end_capture=%.6f\n",
                 ProbeTimestamp());
    size_t node_count = 0;
    size_t kernel_nodes = 0;
    OP_REQUIRES_OK(ctx, InspectGraph(graph, &node_count, &kernel_nodes));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING end_inspect=%.6f\n",
                 ProbeTimestamp());
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaGraphInstantiate(&exec, graph, 0),
                       "musaGraphInstantiate"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING end_instantiate=%.6f\n",
                 ProbeTimestamp());

    unsigned int counter_after_capture = 0;
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(stream),
                                  "musaStreamSynchronize(capture)"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING end_sync=%.6f\n",
                 ProbeTimestamp());
    {
      std::lock_guard<std::mutex> lock(CapturedGraphMutex());
      auto& graphs = CapturedGraphs();
      auto it = graphs.find(handle_);
      OP_REQUIRES(ctx, it != graphs.end(),
                  errors::NotFound("graph handle not found: ", handle_));
      OP_REQUIRES(ctx, it->second.counter != nullptr,
                  errors::FailedPrecondition("graph counter is null"));
      OP_REQUIRES_OK(
          ctx, CheckMusa(musaMemcpy(&counter_after_capture, it->second.counter,
                                    sizeof(counter_after_capture),
                                    musaMemcpyDeviceToHost),
                         "musaMemcpy(counter_after_capture)"));
      it->second.graph = graph;
      it->second.exec = exec;
      it->second.node_count = node_count;
      it->second.kernel_nodes = kernel_nodes;
    }
    std::fprintf(stderr,
                 "GRAPH_CAPTURE_DIAG capture_nodes=%zu capture_kernels=%zu "
                 "counter_after_capture=%u\n",
                 node_count, kernel_nodes, counter_after_capture);
    OP_REQUIRES(ctx, counter_after_capture == 0,
                errors::Internal("captured counter kernel executed during "
                                 "capture: counter=",
                                 counter_after_capture));
  }

 private:
  int64_t handle_;
};

class MusaGraphCounterTickOp : public OpKernel {
 public:
  explicit MusaGraphCounterTickOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING tick_enter=%.6f\n",
                 ProbeTimestamp());
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    unsigned int* counter = nullptr;
    {
      std::lock_guard<std::mutex> lock(CapturedGraphMutex());
      auto& graphs = CapturedGraphs();
      auto it = graphs.find(handle_);
      OP_REQUIRES(ctx, it != graphs.end(),
                  errors::NotFound("graph handle not found: ", handle_));
      counter = it->second.counter;
    }
    OP_REQUIRES(ctx, counter != nullptr,
                errors::FailedPrecondition("graph counter is null"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchMusaGraphCounterTick(counter, stream);
    OP_REQUIRES_OK(ctx, CheckMusa(musaGetLastError(),
                                  "LaunchMusaGraphCounterTick"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING tick_exit=%.6f\n",
                 ProbeTimestamp());
  }

 private:
  int64_t handle_;
};

class MusaGraphReplayOp : public OpKernel {
 public:
  explicit MusaGraphReplayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING replay_enter=%.6f\n",
                 ProbeTimestamp());
    Tensor* diagnostics = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({4}), &diagnostics));

    musaGraphExec_t exec = nullptr;
    unsigned int* counter = nullptr;
    size_t node_count = 0;
    size_t kernel_nodes = 0;
    {
      std::lock_guard<std::mutex> lock(CapturedGraphMutex());
      auto& graphs = CapturedGraphs();
      auto it = graphs.find(handle_);
      OP_REQUIRES(ctx, it != graphs.end(),
                  errors::NotFound("graph handle not found: ", handle_));
      exec = it->second.exec;
      counter = it->second.counter;
      node_count = it->second.node_count;
      kernel_nodes = it->second.kernel_nodes;
    }
    OP_REQUIRES(ctx, exec != nullptr && counter != nullptr,
                errors::FailedPrecondition("graph is not fully captured"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(stream),
                                  "musaStreamSynchronize(before_replay)"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING replay_presync=%.6f\n",
                 ProbeTimestamp());
    unsigned int counter_before_replay = 0;
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpy(&counter_before_replay, counter,
                                  sizeof(counter_before_replay),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_before_replay)"));
    OP_REQUIRES(ctx, counter_before_replay == 0,
                errors::Internal("captured counter kernel executed before "
                                 "replay: counter=",
                                 counter_before_replay));

    OP_REQUIRES_OK(
        ctx, CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(stream),
                                  "musaStreamSynchronize(after_replay)"));
    std::fprintf(stderr, "GRAPH_CAPTURE_TIMING replay_postsync=%.6f\n",
                 ProbeTimestamp());
    unsigned int counter_after_replay = 0;
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpy(&counter_after_replay, counter,
                                  sizeof(counter_after_replay),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_after_replay)"));
    OP_REQUIRES(ctx, counter_after_replay == 1,
                errors::Internal("captured counter kernel did not execute "
                                 "exactly once during replay: counter=",
                                 counter_after_replay));

    std::fprintf(stderr,
                 "GRAPH_CAPTURE_DIAG replay_nodes=%zu replay_kernels=%zu "
                 "counter_before=%u counter_after=%u\n",
                 node_count, kernel_nodes, counter_before_replay,
                 counter_after_replay);
    auto values = diagnostics->flat<int32>();
    values(0) = static_cast<int32>(counter_before_replay);
    values(1) = static_cast<int32>(counter_after_replay);
    values(2) = static_cast<int32>(node_count);
    values(3) = static_cast<int32>(kernel_nodes);
  }

 private:
  int64_t handle_;
};

}  // namespace

REGISTER_OP("MusaGraphCaptureBegin")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaGraphCaptureEnd")
    .Input("x: T")
    .Attr("T: {float, double}")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaGraphReplay")
    .Attr("handle: int = 1")
    .Output("diagnostics: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(4));
      return OkStatus();
    });

REGISTER_OP("MusaGraphCounterTick")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaGraphCaptureBegin").Device("MUSA"),
    MusaGraphCaptureBeginOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaGraphCaptureEnd").Device("MUSA").TypeConstraint<float>("T"),
    MusaGraphCaptureEndOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaGraphCaptureEnd").Device("MUSA").TypeConstraint<double>("T"),
    MusaGraphCaptureEndOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaGraphReplay").Device("MUSA"),
    MusaGraphReplayOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaGraphCounterTick").Device("MUSA"),
    MusaGraphCounterTickOp);

}  // namespace musa
}  // namespace tensorflow
