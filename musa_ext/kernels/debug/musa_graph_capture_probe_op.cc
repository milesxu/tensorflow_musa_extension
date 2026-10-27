#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <musa_runtime.h>

extern "C" void LaunchMusaGraphCaptureProbe(const float* x, float* y, int n,
                                            musaStream_t stream);

namespace tensorflow {
namespace musa {

class MusaGraphCaptureProbeOp : public OpKernel {
 public:
  explicit MusaGraphCaptureProbeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dtype() == DT_FLOAT,
                errors::InvalidArgument("MusaGraphCaptureProbe expects float32"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    const int n = static_cast<int>(input.NumElements());
    if (n == 0) {
      return;
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    auto check = [](musaError_t status, const char* call) -> Status {
      if (status == musaSuccess) {
        return OkStatus();
      }
      return errors::Internal(call, " failed with MUSA error ",
                              static_cast<int>(status));
    };

    OP_REQUIRES_OK(
        ctx, check(musaMemsetAsync(output->flat<float>().data(), 0,
                                   n * sizeof(float), stream),
                   "musaMemsetAsync"));
    OP_REQUIRES_OK(ctx, check(musaStreamBeginCapture(
                             stream, musaStreamCaptureModeRelaxed),
                             "musaStreamBeginCapture"));

    LaunchMusaGraphCaptureProbe(input.flat<float>().data(),
                                output->flat<float>().data(), n, stream);

    musaGraph_t graph = nullptr;
    musaGraphExec_t graph_exec = nullptr;
    OP_REQUIRES_OK(ctx, check(musaStreamEndCapture(stream, &graph),
                              "musaStreamEndCapture"));
    OP_REQUIRES_OK(
        ctx, check(musaGraphInstantiate(&graph_exec, graph, 0),
                   "musaGraphInstantiate"));
    OP_REQUIRES_OK(
        ctx, check(musaGraphLaunch(graph_exec, stream), "musaGraphLaunch#1"));
    OP_REQUIRES_OK(
        ctx, check(musaGraphLaunch(graph_exec, stream), "musaGraphLaunch#2"));
    OP_REQUIRES_OK(
        ctx, check(musaStreamSynchronize(stream), "musaStreamSynchronize"));

    musaGraphExecDestroy(graph_exec);
    musaGraphDestroy(graph);
  }
};

REGISTER_OP("MusaGraphCaptureProbe")
    .Input("x: float")
    .Output("y: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaGraphCaptureProbe").Device("MUSA"),
    MusaGraphCaptureProbeOp);

}  // namespace musa
}  // namespace tensorflow
