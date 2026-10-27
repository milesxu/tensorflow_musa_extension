#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <musa_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "mu/device/musa_capture_event_registry.h"

extern "C" void LaunchMusaGraphCounterTick(unsigned int* counter,
                                           musaStream_t stream);
extern "C" void* MusaGraphCounterTickKernelAddress();

namespace tensorflow {
namespace musa {
namespace {

enum class FixedGraphPhase {
  kCapturing,
  kReady,
  kFailed,
  kReplaying,
};

struct FixedGraphState {
  std::mutex mutex;
  int64_t handle = 0;
  int64_t generation = 0;
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
  size_t capture_event_count = 0;
  std::vector<musaEvent_t> capture_events;
  FixedGraphPhase phase = FixedGraphPhase::kCapturing;
};

using FixedGraphStatePtr = std::shared_ptr<FixedGraphState>;

std::mutex& FixedGraphMutex() {
  static std::mutex mutex;
  return mutex;
}

using FixedGraphKey = std::pair<int64_t, int64_t>;

std::map<FixedGraphKey, FixedGraphStatePtr>& FixedGraphs() {
  static std::map<FixedGraphKey, FixedGraphStatePtr> graphs;
  return graphs;
}

FixedGraphStatePtr FindFixedGraph(int64_t handle, int64_t generation) {
  std::lock_guard<std::mutex> lock(FixedGraphMutex());
  auto& graphs = FixedGraphs();
  auto it = graphs.find(FixedGraphKey(handle, generation));
  if (it == graphs.end()) {
    return nullptr;
  }
  return it->second;
}

void RemoveFixedGraph(int64_t handle, int64_t generation,
                      const FixedGraphStatePtr& expected) {
  std::lock_guard<std::mutex> lock(FixedGraphMutex());
  auto& graphs = FixedGraphs();
  auto it = graphs.find(FixedGraphKey(handle, generation));
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

Status ValidateFixedGraphContext(OpKernelContext* ctx,
                                 const FixedGraphStatePtr& state,
                                 FixedGraphPhase required_phase,
                                 const char* operation) {
  if (GetMusaDeviceIdByCtx(ctx) != state->device_id) {
    return errors::FailedPrecondition(operation, " device mismatch");
  }
  if (GetMusaStreamByCtx(ctx) != state->stream) {
    return errors::FailedPrecondition(operation, " stream mismatch");
  }
  if (state->phase != required_phase) {
    return errors::FailedPrecondition(operation, " invalid graph phase");
  }
  return OkStatus();
}

Status DestroyCaptureEvents(std::vector<musaEvent_t>* events, int device_id) {
  if (events == nullptr || events->empty()) {
    return OkStatus();
  }
  if (device_id >= 0) {
    musaSetDevice(device_id);
  }
  Status first_error = OkStatus();
  std::vector<musaEvent_t> remaining;
  size_t destroyed = 0;
  for (musaEvent_t event : *events) {
    const musaError_t err = musaEventDestroy(event);
    if (err == musaSuccess) {
      ++destroyed;
    } else {
      remaining.push_back(event);
      if (first_error.ok()) {
        first_error = errors::Internal("musaEventDestroy failed: ",
                                       static_cast<int>(err));
      }
    }
  }
  events->swap(remaining);
  std::fprintf(stderr,
               "FIXED_GRAPH_EVENT_DESTROY destroyed=%zu remaining=%zu "
               "status=%s\n",
               destroyed, events->size(),
               first_error.ok() ? "OK" : "ERROR");
  return first_error;
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr,
                 "FIXED_GRAPH_BEGIN_ENTER handle=%lld generation=%lld\n",
                 static_cast<long long>(handle_),
                 static_cast<long long>(generation_));
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = std::make_shared<FixedGraphState>();
    state->handle = handle_;
    state->generation = generation_;
    state->device_id = GetMusaDeviceIdByCtx(ctx);
    state->stream = GetMusaStreamByCtx(ctx);
    if (state->device_id < 0 || state->stream == nullptr) {
      ctx->CtxFailure(
          errors::FailedPrecondition("invalid MUSA device or stream"));
      return;
    }

    unsigned int* counter = nullptr;
    auto fail = [&](const Status& status) {
      if (counter != nullptr) {
        std::fprintf(stderr,
                     "FIXED_GRAPH_COUNTER_FREE handle=%lld generation=%lld "
                     "counter=%p reason=begin_failure\n",
                     static_cast<long long>(handle_),
                     static_cast<long long>(generation_), counter);
        musaFree(counter);
        counter = nullptr;
      }
      ctx->CtxFailure(status);
    };

    Status status =
        CheckMusa(musaMalloc(reinterpret_cast<void**>(&counter),
                             sizeof(*counter)),
                  "musaMalloc(counter)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    std::fprintf(stderr,
                 "FIXED_GRAPH_COUNTER_ALLOC handle=%lld generation=%lld "
                 "counter=%p\n",
                 static_cast<long long>(handle_),
                 static_cast<long long>(generation_), counter);
    status = CheckMusa(musaMemset(counter, 0, sizeof(*counter)),
                       "musaMemset(counter)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaStreamSynchronize(state->stream),
                       "musaStreamSynchronize(counter)");
    if (!status.ok()) {
      fail(status);
      return;
    }

    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->counter = counter;
    }

    status = CheckMusa(
        musaStreamBeginCapture(state->stream, musaStreamCaptureModeRelaxed),
        "musaStreamBeginCapture");
    if (!status.ok()) {
      fail(status);
      return;
    }
    unsigned long long capture_id = 0;
    status = GetActiveCaptureInfo(state->stream, &capture_id);
    if (!status.ok()) {
      musaGraph_t aborted_graph = nullptr;
      musaStreamEndCapture(state->stream, &aborted_graph);
      if (aborted_graph != nullptr) {
        musaGraphDestroy(aborted_graph);
      }
      fail(status);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->capture_id = capture_id;
    }
    {
      std::lock_guard<std::mutex> lock(FixedGraphMutex());
      auto& graphs = FixedGraphs();
      const FixedGraphKey key(handle_, generation_);
      if (graphs.find(key) != graphs.end()) {
        musaGraph_t aborted_graph = nullptr;
        musaStreamEndCapture(state->stream, &aborted_graph);
        if (aborted_graph != nullptr) {
          musaGraphDestroy(aborted_graph);
        }
        fail(errors::AlreadyExists(
            "fixed graph handle/generation already exists: ", handle_, "/",
            generation_));
        return;
      }
      graphs.emplace(key, state);
    }
    std::fprintf(stderr,
                 "FIXED_GRAPH_BEGIN handle=%lld generation=%lld device=%d "
                 "capture_id=%llu counter=%p\n",
                 static_cast<long long>(handle_),
                 static_cast<long long>(generation_), state->device_id,
                 capture_id, counter);
  }

 private:
  int64_t handle_;
  int64_t generation_ = 0;
};

template <typename T>
class MusaFixedGraphRetainTensorsOp : public OpKernel {
 public:
  explicit MusaFixedGraphRetainTensorsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::fprintf(stderr, "FIXED_GRAPH_RETAIN_ENTER handle=%lld\n",
                 static_cast<long long>(handle_));
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kCapturing, "retain"));
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
  int64_t generation_ = 0;
};

class MusaFixedGraphCounterTickOp : public OpKernel {
 public:
  explicit MusaFixedGraphCounterTickOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kCapturing, "tick"));
    OP_REQUIRES(ctx, state->counter != nullptr,
                errors::FailedPrecondition("tick counter is null"));
    LaunchMusaGraphCounterTick(state->counter, GetMusaStreamByCtx(ctx));
    OP_REQUIRES_OK(ctx, CheckMusa(musaGetLastError(),
                                  "LaunchMusaGraphCounterTick"));
  }

 private:
  int64_t handle_;
  int64_t generation_ = 0;
};

class MusaFixedGraphCaptureEndOp : public OpKernel {
 public:
  explicit MusaFixedGraphCaptureEndOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));
    (void)ctx->input(0);

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    Status status = ValidateFixedGraphContext(
        ctx, state, FixedGraphPhase::kCapturing, "capture_end");
    auto fail = [&](const Status& failure) {
      state->phase = FixedGraphPhase::kFailed;
      ctx->CtxFailure(failure);
    };
    if (!status.ok()) {
      fail(status);
      return;
    }
    if (!state->a.IsInitialized() || !state->b.IsInitialized() ||
        !state->c.IsInitialized()) {
      fail(errors::FailedPrecondition(
          "capture tensors were not retained"));
      return;
    }

    musaGraph_t graph = nullptr;
    status = CheckMusa(musaStreamEndCapture(state->stream, &graph),
                       "musaStreamEndCapture");
    if (!status.ok()) {
      fail(status);
      return;
    }
    state->graph = graph;
    state->capture_events = TakeCaptureDependencyEvents(
        state->device_id, state->capture_id);
    state->capture_event_count = state->capture_events.size();

    status = InspectGraph(graph, MusaGraphCounterTickKernelAddress(),
                          &state->node_count, &state->kernel_nodes,
                          &state->tick_kernel_nodes);
    if (!status.ok()) {
      fail(status);
      return;
    }
    if (state->kernel_nodes < 2) {
      fail(errors::Internal("captured graph has fewer than 2 kernels"));
      return;
    }
    if (state->tick_kernel_nodes != 1) {
      fail(errors::Internal("captured graph tick kernel count is ",
                            state->tick_kernel_nodes));
      return;
    }

    status = CheckMusa(musaGraphInstantiate(&state->exec, graph, 0),
                       "musaGraphInstantiate");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaStreamSynchronize(state->stream),
                       "musaStreamSynchronize(capture)");
    if (!status.ok()) {
      fail(status);
      return;
    }

    unsigned int counter_after_capture = 0;
    status = CheckMusa(musaMemcpy(&counter_after_capture, state->counter,
                                  sizeof(counter_after_capture),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_after_capture)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    if (counter_after_capture != 0) {
      fail(errors::Internal("captured tick executed during capture: ",
                            counter_after_capture));
      return;
    }

    state->phase = FixedGraphPhase::kReady;
    state->capture_count = 1;
    std::fprintf(
        stderr,
        "FIXED_GRAPH_CAPTURE handle=%lld generation=%lld capture_id=%llu "
        "nodes=%zu kernels=%zu tick_kernels=%zu retain_calls=%d events=%zu "
        "counter=%u a=%p b=%p c=%p\n",
        static_cast<long long>(handle_),
        static_cast<long long>(generation_), state->capture_id,
        state->node_count, state->kernel_nodes, state->tick_kernel_nodes,
        state->retain_calls, state->capture_event_count, counter_after_capture,
        state->a.data(), state->b.data(), state->c.data());
  }

 private:
  int64_t handle_;
  int64_t generation_ = 0;
};

class MusaFixedGraphReplayOp : public OpKernel {
 public:
  explicit MusaFixedGraphReplayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* diagnostics = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({9}), &diagnostics));

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kReady, "replay"));
    OP_REQUIRES(ctx, state->exec != nullptr && state->counter != nullptr,
                errors::FailedPrecondition("fixed graph is not ready"));
    state->phase = FixedGraphPhase::kReplaying;
    auto fail = [&](const Status& failure) {
      state->phase = FixedGraphPhase::kFailed;
      ctx->CtxFailure(failure);
    };

    unsigned int counter_before = 0;
    Status status = CheckMusa(musaStreamSynchronize(state->stream),
                              "musaStreamSynchronize(before_replay)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaMemcpy(&counter_before, state->counter,
                                  sizeof(counter_before),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_before)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaGraphLaunch(state->exec, state->stream),
                       "musaGraphLaunch");
    if (!status.ok()) {
      fail(status);
      return;
    }
    status = CheckMusa(musaStreamSynchronize(state->stream),
                       "musaStreamSynchronize(after_replay)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    unsigned int counter_after = 0;
    status = CheckMusa(musaMemcpy(&counter_after, state->counter,
                                  sizeof(counter_after),
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(counter_after)");
    if (!status.ok()) {
      fail(status);
      return;
    }
    if (counter_after != counter_before + 1) {
      fail(errors::Internal("replay tick count mismatch: before=",
                            counter_before, " after=", counter_after));
      return;
    }

    ++state->replay_count;
    state->phase = FixedGraphPhase::kReady;
    std::fprintf(
        stderr,
        "FIXED_GRAPH_REPLAY handle=%lld generation=%lld count=%d "
        "counter=%u->%u nodes=%zu kernels=%zu tick_kernels=%zu "
        "retain_calls=%d events=%zu\n",
        static_cast<long long>(handle_),
        static_cast<long long>(generation_), state->replay_count,
        counter_before, counter_after, state->node_count, state->kernel_nodes,
        state->tick_kernel_nodes, state->retain_calls,
        state->capture_event_count);

    std::array<int32, 9> values = {
        static_cast<int32>(counter_before),
        static_cast<int32>(counter_after),
        static_cast<int32>(state->node_count),
        static_cast<int32>(state->kernel_nodes),
        static_cast<int32>(state->tick_kernel_nodes),
        state->retain_calls,
        state->replay_count,
        static_cast<int32>(state->capture_event_count),
        static_cast<int32>(state->generation),
    };
    OP_REQUIRES_OK(
        ctx, CheckMusa(musaMemcpy(diagnostics->flat<int32>().data(),
                                  values.data(), sizeof(values),
                                  musaMemcpyHostToDevice),
                       "musaMemcpy(diagnostics)"));
  }

 private:
  int64_t handle_;
  int64_t generation_ = 0;
};

class MusaFixedGraphZeroOutputOp : public OpKernel {
 public:
  explicit MusaFixedGraphZeroOutputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kReady, "zero"));
    OP_REQUIRES(ctx, state->c.IsInitialized(),
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
  int64_t generation_ = 0;
};

template <typename T>
class MusaFixedGraphWriteInputOp : public OpKernel {
 public:
  explicit MusaFixedGraphWriteInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kReady, "write_input"));
    OP_REQUIRES(ctx, state->a.IsInitialized() && state->b.IsInitialized(),
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
  int64_t generation_ = 0;
};

template <typename T>
class MusaFixedGraphReadOutputOp : public OpKernel {
 public:
  explicit MusaFixedGraphReadOutputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kReady, "read_output"));
    OP_REQUIRES(ctx, state->c.IsInitialized() &&
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
  int64_t generation_ = 0;
};

template <typename T>
class MusaFixedGraphReadInputOp : public OpKernel {
 public:
  explicit MusaFixedGraphReadInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("which", &which_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kReady, "read_input"));
    OP_REQUIRES(ctx, state->a.IsInitialized() && state->b.IsInitialized() &&
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
  int64_t generation_ = 0;
  int64_t which_ = 0;
};

template <typename T>
class MusaFixedGraphVerifyAddressesOp : public OpKernel {
 public:
  explicit MusaFixedGraphVerifyAddressesOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle not found: ", handle_));

    std::lock_guard<std::mutex> lock(state->mutex);
    OP_REQUIRES_OK(ctx, ValidateFixedGraphContext(
                            ctx, state, FixedGraphPhase::kReady, "verify"));
    OP_REQUIRES(ctx, state->a.IsInitialized() && state->b.IsInitialized() &&
                         state->c.IsInitialized(),
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
  int64_t generation_ = 0;
};

class MusaFixedGraphReleaseOp : public OpKernel {
 public:
  explicit MusaFixedGraphReleaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("handle", &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("generation", &generation_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* token = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &token));

    auto state = FindFixedGraph(handle_, generation_);
    OP_REQUIRES(ctx, state != nullptr,
                errors::NotFound("fixed graph handle/generation not found: ",
                                 handle_, "/", generation_));

    std::lock_guard<std::mutex> lock(state->mutex);
    if (GetMusaDeviceIdByCtx(ctx) != state->device_id ||
        GetMusaStreamByCtx(ctx) != state->stream) {
      ctx->CtxFailure(
          errors::FailedPrecondition("release context mismatch"));
      return;
    }
    if (state->phase == FixedGraphPhase::kReplaying) {
      ctx->CtxFailure(
          errors::FailedPrecondition("cannot release while replaying"));
      return;
    }
    if (state->device_id >= 0) {
      musaSetDevice(state->device_id);
    }

    if (state->phase == FixedGraphPhase::kCapturing) {
      musaStreamCaptureStatus capture_status = musaStreamCaptureStatusNone;
      Status status = CheckMusa(
          musaStreamIsCapturing(state->stream, &capture_status),
          "musaStreamIsCapturing(release)");
      if (!status.ok()) {
        state->phase = FixedGraphPhase::kFailed;
        ctx->CtxFailure(status);
        return;
      }
      if (capture_status == musaStreamCaptureStatusActive) {
        musaGraph_t aborted_graph = nullptr;
        status = CheckMusa(musaStreamEndCapture(state->stream, &aborted_graph),
                           "musaStreamEndCapture(abort)");
        if (!status.ok()) {
          state->phase = FixedGraphPhase::kFailed;
          ctx->CtxFailure(status);
          return;
        }
        if (state->graph == nullptr) {
          state->graph = aborted_graph;
        } else if (aborted_graph != nullptr) {
          musaGraphDestroy(aborted_graph);
        }
      }
    }

    if (state->capture_id != 0) {
      std::vector<musaEvent_t> pending =
          TakeCaptureDependencyEvents(state->device_id, state->capture_id);
      state->capture_events.insert(state->capture_events.end(),
                                   pending.begin(), pending.end());
      state->capture_event_count =
          std::max(state->capture_event_count, state->capture_events.size());
    }

    Status sync_status =
        CheckMusa(musaStreamSynchronize(state->stream),
                  "musaStreamSynchronize(release)");
    if (!sync_status.ok()) {
      state->phase = FixedGraphPhase::kFailed;
      ctx->CtxFailure(sync_status);
      return;
    }

    auto fail_cleanup = [&](const Status& status) {
      state->phase = FixedGraphPhase::kFailed;
      ctx->CtxFailure(status);
    };
    if (state->exec != nullptr) {
      const Status status = CheckMusa(musaGraphExecDestroy(state->exec),
                                      "musaGraphExecDestroy");
      if (!status.ok()) {
        fail_cleanup(status);
        return;
      }
      state->exec = nullptr;
    }
    if (state->graph != nullptr) {
      const Status status =
          CheckMusa(musaGraphDestroy(state->graph), "musaGraphDestroy");
      if (!status.ok()) {
        fail_cleanup(status);
        return;
      }
      state->graph = nullptr;
    }
    {
      const Status status = DestroyCaptureEvents(&state->capture_events,
                                                 state->device_id);
      if (!status.ok()) {
        fail_cleanup(status);
        return;
      }
    }
    if (state->counter != nullptr) {
      std::fprintf(stderr,
                   "FIXED_GRAPH_COUNTER_FREE handle=%lld generation=%lld "
                   "counter=%p reason=release\n",
                   static_cast<long long>(handle_),
                   static_cast<long long>(generation_), state->counter);
      const Status status = CheckMusa(musaFree(state->counter), "musaFree");
      if (!status.ok()) {
        fail_cleanup(status);
        return;
      }
      state->counter = nullptr;
    }
    if (state->capture_id != 0 &&
        PendingCaptureDependencyEventCount(state->device_id,
                                           state->capture_id) != 0) {
      fail_cleanup(errors::Internal(
          "capture dependency events remain after release"));
      return;
    }
    if (state->capture_id != 0) {
      ReleaseCaptureDependencyGate(state->device_id, state->capture_id);
    }
    state->a = Tensor();
    state->b = Tensor();
    state->c = Tensor();
    std::fprintf(stderr,
                 "FIXED_GRAPH_RELEASE handle=%lld generation=%lld "
                 "capture_id=%llu events=%zu status=OK\n",
                 static_cast<long long>(handle_),
                 static_cast<long long>(generation_), state->capture_id,
                 state->capture_event_count);
    RemoveFixedGraph(handle_, generation_, state);
  }

 private:
  int64_t handle_;
  int64_t generation_ = 0;
};

}  // namespace

REGISTER_OP("MusaFixedGraphCaptureBegin")
    .Attr("handle: int = 1")
    .Attr("generation: int = 0")
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
    .Attr("generation: int = 0")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphCaptureEnd")
    .Input("trigger: int32")
    .Attr("handle: int = 1")
    .Attr("generation: int = 0")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphCounterTick")
    .Attr("handle: int = 1")
    .Attr("generation: int = 0")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphReplay")
    .Attr("handle: int = 1")
    .Attr("generation: int = 0")
    .Output("diagnostics: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(9));
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphZeroOutput")
    .Attr("handle: int = 1")
    .Attr("generation: int = 0")
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
    .Attr("generation: int = 0")
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
    .Attr("generation: int = 0")
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
    .Attr("generation: int = 0")
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
    .Attr("generation: int = 0")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("MusaFixedGraphRelease")
    .Attr("handle: int = 1")
    .Attr("generation: int = 0")
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
