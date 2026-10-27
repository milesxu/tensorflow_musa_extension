#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <musa_runtime.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace tensorflow {
namespace musa {
namespace {

struct CapturedGraph {
  musaGraph_t graph;
  musaGraphExec_t exec;
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

class MusaGraphCaptureBeginOp : public OpKernel {
 public:
  explicit MusaGraphCaptureBeginOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaStreamBeginCapture(
                          stream, musaStreamCaptureModeRelaxed),
                       "musaStreamBeginCapture"));
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
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));
    (void)ctx->input(0);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaGraph_t graph = nullptr;
    musaGraphExec_t exec = nullptr;
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamEndCapture(stream, &graph),
                                  "musaStreamEndCapture"));
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaGraphInstantiate(&exec, graph, 0),
                       "musaGraphInstantiate"));

    std::lock_guard<std::mutex> lock(CapturedGraphMutex());
    auto& graphs = CapturedGraphs();
    OP_REQUIRES(ctx, graphs.find(handle_) == graphs.end(),
                errors::AlreadyExists("graph handle already captured: ",
                                      handle_));
    graphs.emplace(handle_, CapturedGraph{graph, exec});
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
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    musaGraphExec_t exec = nullptr;
    {
      std::lock_guard<std::mutex> lock(CapturedGraphMutex());
      auto& graphs = CapturedGraphs();
      auto it = graphs.find(handle_);
      OP_REQUIRES(ctx, it != graphs.end(),
                  errors::NotFound("graph handle not found: ", handle_));
      exec = it->second.exec;
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch"));
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

}  // namespace musa
}  // namespace tensorflow
