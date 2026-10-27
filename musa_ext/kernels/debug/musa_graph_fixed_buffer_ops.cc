#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <musa_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" void LaunchMusaGraphCounterTick(unsigned int* counter,
                                           musaStream_t stream);
extern "C" void* MusaGraphCounterTickKernelAddress();

namespace tensorflow {
namespace musa {
namespace {

struct FixedGraphState {
  std::mutex mutex;
  int64_t handle = 0;
  int device_id = -1;
  musaStream_t stream = nullptr;
  unsigned long long capture_id = 0;
  musaGraph_t graph = nullptr;
  musaGraphExec_t exec = nullptr;
  unsigned int* counter = nullptr;
  Tensor a;
  Tensor b;
  Tensor c;
  void* captured_a = nullptr;
  void* captured_b = nullptr;
  void* captured_c = nullptr;
  size_t node_count = 0;
  size_t kernel_nodes = 0;
  size_t tick_kernel_nodes = 0;
  int capture_count = 0;
  int retain_calls = 0;
  int replay_count = 0;
  bool capture_active = false;
};

using FixedGraphStatePtr = std::shared_ptr<FixedGraphState>;

std::mutex& FixedGraphMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<int64_t, FixedGraphStatePtr>& FixedGraphs() {
  static std::unordered_map<int64_t, FixedGraphStatePtr> graphs;
  return graphs;
}

FixedGraphStatePtr FindFixedGraph(int64_t handle) {
  std::lock_guard<std::mutex> lock(FixedGraphMutex());
  auto& graphs = FixedGraphs();
  auto it = graphs.find(handle);
  return it == graphs.end() ? nullptr : it->second;
}

void RemoveFixedGraph(int64_t handle, const FixedGraphStatePtr& expected) {
  std::lock_guard<std::mutex> lock(FixedGraphMutex());
  auto& graphs = FixedGraphs();
  auto it = graphs.find(handle);
  if (it != graphs.end() && it->second == expected) {
    graphs.erase(it);
  }
}

Status CheckMusa(musaError_t status, const char* call) {
  if (status == musaSuccess) {
    return OkStatus();
  }
  return errors::Internal(call, " failed with MUSA error ",
                          static_cast<int>(status));
}

Status GetActiveCaptureInfo(musaStream_t stream,
                            unsigned long long* capture_id) {
  musaStreamCaptureStatus status = musaStreamCaptureStatusNone;
  unsigned long long id = 0;
  TF_RETURN_IF_ERROR(CheckMusa(
      musaStreamGetCaptureInfo(stream, &status, &id), "musaStreamGetCaptureInfo"));
  if (status != musaStreamCaptureStatusActive) {
    return errors::FailedPrecondition("stream is not actively capturing");
  }
  *capture_id = id;
  return OkStatus();
}

Status InspectGraph(musaGraph_t graph, void* tick_kernel,
                    size_t* node_count, size_t* kernel_nodes,
                    size_t* tick_kernel_nodes) {
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
  size_t tick_kernels = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    musaGraphNodeType type;
    TF_RETURN_IF_ERROR(CheckMusa(musaGraphNodeGetType(nodes[i], &type),
                                 "musaGraphNodeGetType"));
    if (type != musaGraphNodeTypeKernel) {
      continue;
    }
    ++kernels;
    musaKernelNodeParams params{};
    TF_RETURN_IF_ERROR(CheckMusa(
        musaGraphKernelNodeGetParams(nodes[i], &params),
        "musaGraphKernelNodeGetParams"));
    const bool is_tick = params.func == tick_kernel;
    if (is_tick) {
      ++tick_kernels;
    }
    std::fprintf(
        stderr,
        "FIXED_GRAPH_KERNEL index=%zu func=%p tick=%d grid=(%u,%u,%u) "
        "block=(%u,%u,%u) shared=%u\n",
        i, params.func, is_tick ? 1 : 0, params.gridDim.x, params.gridDim.y,
        params.gridDim.z, params.blockDim.x, params.blockDim.y,
        params.blockDim.z, params.sharedMemBytes);
  }

  *node_count = count;
  *kernel_nodes = kernels;
  *tick_kernel_nodes = tick_kernels;
  return OkStatus();
}

class MusaFixedGraphCaptureBeginOp : public OpKernel {
 public:
  explicit MusaFixedGraphCaptureBeginOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr, "FIXED_GRAPH_BEGIN_ENTER handle=%lld\n",
                 static_cast<long long>(handle_));
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = std::make_shared<FixedGraphState>();
    state->handle = handle_;
    state->device_id = GetMusaDeviceIdByCtx(ctx);
    state->stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES(ctx, state->device_id >= 0 && state->stream != nullptr,
                errors::FailedPrecondition("invalid MUSA device or stream"));

    {
      std::lock_guard<std::mutex> lock(FixedGraphMutex());
      auto& graphs = FixedGraphs();
      OP_REQUIRES(ctx, graphs.find(handle_) == graphs.end(),
                  errors::AlreadyExists("fixed graph handle already exists: ",
                                        handle_));
      graphs.emplace(handle_, state);
    }
    std::fprintf(stderr, "FIXED_GRAPH_BEGIN_STATE handle=%lld\n",
                 static_cast<long long>(handle_));

    unsigned int* counter = nullptr;
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMalloc(reinterpret_cast<void**>(&counter),
                                  sizeof(*counter)),
                       "musaMalloc(counter)"));
    std::fprintf(stderr, "FIXED_GRAPH_BEGIN_MALLOC handle=%lld counter=%p\n",
                 static_cast<long long>(handle_), counter);
    OP_REQUIRES_OK(ctx, CheckMusa(musaMemset(counter, 0, sizeof(*counter)),
                                  "musaMemset(counter)"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(state->stream),
                                  "musaStreamSynchronize(counter)"));

    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->counter = counter;
    }

    OP_REQUIRES_OK(ctx, CheckMusa(
                            musaStreamBeginCapture(
                                state->stream, musaStreamCaptureModeRelaxed),
                            "musaStreamBeginCapture"));
    unsigned long long capture_id = 0;
    OP_REQUIRES_OK(ctx, GetActiveCaptureInfo(state->stream, &capture_id));
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->capture_id = capture_id;
      state->capture_active = true;
    }
    std::fprintf(stderr,
                 "FIXED_GRAPH_BEGIN handle=%lld device=%d capture_id=%llu\n",
                 static_cast<long long>(handle_), state->device_id, capture_id);
  }

 private:
  int64_t handle_;
};

template <typename T>
class MusaFixedGraphRetainTensorsOp : public OpKernel {
 public:
  explicit MusaFixedGraphRetainTensorsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr, "FIXED_GRAPH_RETAIN_ENTER handle=%lld\n",
                 static_cast<long long>(handle_));
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("retain device mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx, state->capture_active && state->capture_count == 0,
                errors::FailedPrecondition("retain outside active capture"));
    state->a = ctx->input(0);
    state->b = ctx->input(1);
    state->c = ctx->input(2);
    state->captured_a = state->a.data();
    state->captured_b = state->b.data();
    state->captured_c = state->c.data();
    ++state->retain_calls;
    std::fprintf(
        stderr,
        "FIXED_GRAPH_RETAIN handle=%lld calls=%d a=%p b=%p c=%p bytes=%lld\n",
        static_cast<long long>(handle_), state->retain_calls, state->a.data(),
        state->b.data(), state->c.data(),
        static_cast<long long>(state->c.TotalBytes()));
  }

 private:
  int64_t handle_;
};

class MusaFixedGraphCounterTickOp : public OpKernel {
 public:
  explicit MusaFixedGraphCounterTickOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("tick device mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx, state->capture_active && state->counter != nullptr,
                errors::FailedPrecondition("tick outside active capture"));
    LaunchMusaGraphCounterTick(state->counter, GetMusaStreamByCtx(ctx));
    OP_REQUIRES_OK(ctx, CheckMusa(musaGetLastError(),
                                  "LaunchMusaGraphCounterTick"));
  }

 private:
  int64_t handle_;
};

class MusaFixedGraphCaptureEndOp : public OpKernel {
 public:
  explicit MusaFixedGraphCaptureEndOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));
    (void)ctx->input(0);

    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("capture end device mismatch"));
    OP_REQUIRES(ctx, GetMusaStreamByCtx(ctx) == state->stream,
                errors::FailedPrecondition("capture end stream mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx, state->capture_active && state->capture_count == 0,
                errors::FailedPrecondition("capture is not active"));
    OP_REQUIRES(ctx,
                state->a.IsInitialized() && state->b.IsInitialized() &&
                    state->c.IsInitialized(),
                errors::FailedPrecondition("capture tensors were not retained"));

    musaGraph_t graph = nullptr;
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamEndCapture(state->stream, &graph),
                                  "musaStreamEndCapture"));
    state->graph = graph;
    state->capture_active = false;

    OP_REQUIRES_OK(
        ctx, InspectGraph(graph, MusaGraphCounterTickKernelAddress(),
                          &state->node_count, &state->kernel_nodes,
                          &state->tick_kernel_nodes));
    OP_REQUIRES(ctx, state->kernel_nodes >= 2,
                errors::Internal("captured graph has fewer than 2 kernels"));
    OP_REQUIRES(ctx, state->tick_kernel_nodes == 1,
                errors::Internal("captured graph tick kernel count is ",
                                 state->tick_kernel_nodes));

    OP_REQUIRES_OK(ctx, CheckMusa(musaGraphInstantiate(&state->exec, graph, 0),
                                  "musaGraphInstantiate"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(state->stream),
                                  "musaStreamSynchronize(capture)"));

    unsigned int counter_after_capture = 0;
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpy(&counter_after_capture, state->counter,
                                  sizeof(counter_after_capture),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_after_capture)"));
    OP_REQUIRES(ctx, counter_after_capture == 0,
                errors::Internal("captured tick executed during capture: ",
                                 counter_after_capture));

    state->capture_count = 1;
    std::fprintf(
        stderr,
        "FIXED_GRAPH_CAPTURE handle=%lld capture_id=%llu nodes=%zu kernels=%zu "
        "tick_kernels=%zu retain_calls=%d counter=%u a=%p b=%p c=%p\n",
        static_cast<long long>(handle_), state->capture_id, state->node_count,
        state->kernel_nodes, state->tick_kernel_nodes, state->retain_calls,
        counter_after_capture, state->a.data(), state->b.data(),
        state->c.data());
  }

 private:
  int64_t handle_;
};

class MusaFixedGraphReplayOp : public OpKernel {
 public:
  explicit MusaFixedGraphReplayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* diagnostics = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({7}), &diagnostics));

    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("replay device mismatch"));
    OP_REQUIRES(ctx, GetMusaStreamByCtx(ctx) == state->stream,
                errors::FailedPrecondition("replay stream mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx, state->capture_count == 1 && state->exec != nullptr &&
                         state->counter != nullptr,
                errors::FailedPrecondition("fixed graph is not ready"));

    unsigned int counter_before = 0;
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(state->stream),
                                  "musaStreamSynchronize(before_replay)"));
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpy(&counter_before, state->counter,
                                  sizeof(counter_before),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_before)"));
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaGraphLaunch(state->exec, state->stream),
                       "musaGraphLaunch"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(state->stream),
                                  "musaStreamSynchronize(after_replay)"));
    unsigned int counter_after = 0;
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpy(&counter_after, state->counter,
                                  sizeof(counter_after),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_after)"));
    OP_REQUIRES(ctx, counter_after == counter_before + 1,
                errors::Internal("replay tick count mismatch: before=",
                                 counter_before, " after=", counter_after));

    ++state->replay_count;
    std::fprintf(
        stderr,
        "FIXED_GRAPH_REPLAY handle=%lld count=%d counter=%u->%u nodes=%zu "
        "kernels=%zu tick_kernels=%zu retain_calls=%d\n",
        static_cast<long long>(handle_), state->replay_count, counter_before,
        counter_after, state->node_count, state->kernel_nodes,
        state->tick_kernel_nodes, state->retain_calls);

    std::array<int32, 7> values = {
        static_cast<int32>(counter_before),
        static_cast<int32>(counter_after),
        static_cast<int32>(state->node_count),
        static_cast<int32>(state->kernel_nodes),
        static_cast<int32>(state->tick_kernel_nodes),
        state->retain_calls,
        state->replay_count,
    };
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpy(diagnostics->flat<int32>().data(),
                                  values.data(), sizeof(values),
                                  musaMemcpyHostToDevice),
                       "musaMemcpy(diagnostics)"));
  }

 private:
  int64_t handle_;
};

class MusaFixedGraphZeroOutputOp : public OpKernel {
 public:
  explicit MusaFixedGraphZeroOutputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("zero device mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx, state->capture_count == 1 && state->c.IsInitialized(),
                errors::FailedPrecondition("fixed graph is not ready"));
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemsetAsync(state->c.data(), 0, state->c.TotalBytes(),
                                       GetMusaStreamByCtx(ctx)),
                       "musaMemsetAsync(output)"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(GetMusaStreamByCtx(ctx)),
                                  "musaStreamSynchronize(zero)"));
  }

 private:
  int64_t handle_;
};

template <typename T>
class MusaFixedGraphWriteInputOp : public OpKernel {
 public:
  explicit MusaFixedGraphWriteInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("write input device mismatch"));
    OP_REQUIRES(ctx, GetMusaStreamByCtx(ctx) == state->stream,
                errors::FailedPrecondition("write input stream mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx,
                state->capture_count == 1 && state->a.IsInitialized() &&
                    state->b.IsInitialized(),
                errors::FailedPrecondition("fixed graph is not ready"));
    const Tensor& input_a = ctx->input(0);
    const Tensor& input_b = ctx->input(1);
    OP_REQUIRES(ctx,
                input_a.dtype() == DataTypeToEnum<T>::value &&
                    input_b.dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument("write input dtype mismatch"));
    OP_REQUIRES(ctx,
                input_a.shape() == state->a.shape() &&
                    input_b.shape() == state->b.shape(),
                errors::InvalidArgument("write input shape mismatch"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpyAsync(state->a.data(), input_a.data(),
                                       state->a.TotalBytes(),
                                       musaMemcpyDeviceToDevice, stream),
                       "musaMemcpyAsync(write_a)"));
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpyAsync(state->b.data(), input_b.data(),
                                       state->b.TotalBytes(),
                                       musaMemcpyDeviceToDevice, stream),
                       "musaMemcpyAsync(write_b)"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(stream),
                                  "musaStreamSynchronize(write_input)"));
    std::fprintf(
        stderr,
        "FIXED_GRAPH_WRITE_INPUT handle=%lld a=%p<-%p b=%p<-%p bytes=%lld\n",
        static_cast<long long>(handle_), state->a.data(), input_a.data(),
        state->b.data(), input_b.data(),
        static_cast<long long>(state->a.TotalBytes()));
  }

 private:
  int64_t handle_;
};

template <typename T>
class MusaFixedGraphReadOutputOp : public OpKernel {
 public:
  explicit MusaFixedGraphReadOutputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("read device mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx,
                state->capture_count == 1 && state->c.IsInitialized() &&
                    state->c.dtype() == DataTypeToEnum<T>::value,
                errors::FailedPrecondition("fixed graph output is not ready"));
    OP_REQUIRES(ctx, ctx->input(0).shape() == state->c.shape(),
                errors::InvalidArgument("read output shape mismatch"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, state->c.shape(), &output));
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpyAsync(output->flat<T>().data(),
                                       state->c.flat<T>().data(),
                                       state->c.TotalBytes(),
                                       musaMemcpyDeviceToDevice, stream),
                       "musaMemcpyAsync(read_output)"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(stream),
                                  "musaStreamSynchronize(read_output)"));
  }

 private:
  int64_t handle_;
};

template <typename T>
class MusaFixedGraphReadInputOp : public OpKernel {
 public:
  explicit MusaFixedGraphReadInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("which", &which_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("read inputs device mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx,
                state->capture_count == 1 && state->a.IsInitialized() &&
                    state->b.IsInitialized() &&
                    state->a.dtype() == DataTypeToEnum<T>::value &&
                    state->b.dtype() == DataTypeToEnum<T>::value,
                errors::FailedPrecondition("fixed graph inputs are not ready"));

    OP_REQUIRES(ctx, which_ == 0 || which_ == 1,
                errors::InvalidArgument("which must be 0 or 1, got ", which_));
    const Tensor& source = which_ == 0 ? state->a : state->b;
    OP_REQUIRES(ctx, ctx->input(0).shape() == source.shape(),
                errors::InvalidArgument("read input shape mismatch"));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, source.shape(), &output));
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpyAsync(output->flat<T>().data(),
                                       source.flat<T>().data(),
                                       source.TotalBytes(),
                                       musaMemcpyDeviceToDevice, stream),
                       "musaMemcpyAsync(read_input)"));
    OP_REQUIRES_OK(ctx, CheckMusa(musaStreamSynchronize(stream),
                                  "musaStreamSynchronize(read_input)"));
  }

 private:
  int64_t handle_;
  int64_t which_ = 0;
};

template <typename T>
class MusaFixedGraphVerifyAddressesOp : public OpKernel {
 public:
  explicit MusaFixedGraphVerifyAddressesOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));
    OP_REQUIRES(ctx, GetMusaDeviceIdByCtx(ctx) == state->device_id,
                errors::FailedPrecondition("verify device mismatch"));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx,
                state->capture_count == 1 && state->a.IsInitialized() &&
                    state->b.IsInitialized() && state->c.IsInitialized(),
                errors::FailedPrecondition("fixed graph tensors are not ready"));
    void* current_a = const_cast<T*>(ctx->input(0).flat<T>().data());
    void* current_b = const_cast<T*>(ctx->input(1).flat<T>().data());
    std::fprintf(
        stderr,
        "FIXED_GRAPH_ADDRESS_CHECK handle=%lld a=%p/%p b=%p/%p c=%p/%p\n",
        static_cast<long long>(handle_), state->captured_a, current_a,
        state->captured_b, current_b, state->captured_c, state->c.data());
    OP_REQUIRES(ctx, current_a == state->captured_a,
                errors::FailedPrecondition("captured A address changed"));
    OP_REQUIRES(ctx, current_b == state->captured_b,
                errors::FailedPrecondition("captured B address changed"));
    OP_REQUIRES(ctx, state->c.data() == state->captured_c,
                errors::FailedPrecondition("captured C address changed"));
  }

 private:
  int64_t handle_;
};

class MusaFixedGraphReleaseOp : public OpKernel {
 public:
  explicit MusaFixedGraphReleaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    FixedGraphStatePtr state;
    {
      std::lock_guard<std::mutex> lock(FixedGraphMutex());
      auto& graphs = FixedGraphs();
      auto it = graphs.find(handle_);
      if (it == graphs.end()) {
        return;
      }
      state = it->second;
      graphs.erase(it);
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES(ctx, !state->capture_active,
                errors::FailedPrecondition("cannot release active capture"));
    if (state->device_id >= 0) {
      musaSetDevice(state->device_id);
    }
    if (state->exec != nullptr) {
      OP_REQUIRES_OK(ctx, CheckMusa(musaGraphExecDestroy(state->exec),
                                    "musaGraphExecDestroy"));
      state->exec = nullptr;
    }
    if (state->graph != nullptr) {
      OP_REQUIRES_OK(ctx, CheckMusa(musaGraphDestroy(state->graph),
                                    "musaGraphDestroy"));
      state->graph = nullptr;
    }
    if (state->counter != nullptr) {
      OP_REQUIRES_OK(ctx, CheckMusa(musaFree(state->counter), "musaFree"));
      state->counter = nullptr;
    }
    state->a = Tensor();
    state->b = Tensor();
    state->c = Tensor();
    std::fprintf(stderr, "FIXED_GRAPH_RELEASE handle=%lld\n",
                 static_cast<long long>(handle_));
  }

 private:
  int64_t handle_;
};

}  // namespace

REGISTER_OP("MusaFixedGraphCaptureBegin")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphRetainTensors")
    .Input("a: T")
    .Input("b: T")
    .Input("c: T")
    .Attr("T: {float, double}")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphCaptureEnd")
    .Input("trigger: int32")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphCounterTick")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphReplay")
    .Attr("handle: int = 1")
    .Output("diagnostics: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(7));
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphZeroOutput")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphWriteInput")
    .Input("a: T")
    .Input("b: T")
    .Attr("T: {float, double}")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphReadOutput")
    .Input("like: T")
    .Attr("T: {float, double}")
    .Attr("handle: int = 1")
    .Output("output: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphReadInput")
    .Input("like: T")
    .Attr("T: {float, double}")
    .Attr("handle: int = 1")
    .Attr("which: int = 0")
    .Output("output: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphVerifyAddresses")
    .Input("a: T")
    .Input("b: T")
    .Attr("T: {float, double}")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphRelease")
    .Attr("handle: int = 1")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphCaptureBegin").Device("MUSA"),
    MusaFixedGraphCaptureBeginOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphRetainTensors")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaFixedGraphRetainTensorsOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphRetainTensors")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaFixedGraphRetainTensorsOp<double>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphCaptureEnd").Device("MUSA"),
    MusaFixedGraphCaptureEndOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphCounterTick").Device("MUSA"),
    MusaFixedGraphCounterTickOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphReplay").Device("MUSA"),
    MusaFixedGraphReplayOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphZeroOutput").Device("MUSA"),
    MusaFixedGraphZeroOutputOp);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphWriteInput")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaFixedGraphWriteInputOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphWriteInput")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaFixedGraphWriteInputOp<double>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphReadOutput")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaFixedGraphReadOutputOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphReadOutput")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaFixedGraphReadOutputOp<double>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphReadInput")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaFixedGraphReadInputOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphReadInput")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaFixedGraphReadInputOp<double>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphVerifyAddresses")
        .Device("MUSA")
        .TypeConstraint<float>("T"),
    MusaFixedGraphVerifyAddressesOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphVerifyAddresses")
        .Device("MUSA")
        .TypeConstraint<double>("T"),
    MusaFixedGraphVerifyAddressesOp<double>);
REGISTER_KERNEL_BUILDER(
    Name("MusaFixedGraphRelease").Device("MUSA"),
    MusaFixedGraphReleaseOp);

}  // namespace musa
}  // namespace tensorflow
