#include "../utils_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <mublas.h>
#include <musa_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

extern "C" void LaunchMusaGraphCounterTick(unsigned int* counter,
                                           musaStream_t stream);
extern "C" void LaunchMusaManualGraphBiasTanhFloat(
    const float* input, const float* bias, float* pre, float* output,
    int elements, int width, musaStream_t stream);
extern "C" void LaunchMusaManualGraphBiasFloat(
    const float* input, const float* bias, float* output, int elements,
    int width, musaStream_t stream);
extern "C" void LaunchMusaManualGraphResidualFloat(
    const float* previous, const float* current, const float* idt,
    float* output, int elements, int width, musaStream_t stream);
extern "C" void LaunchMusaManualGraphPoisonFloat(
    float* data, int elements, float value, musaStream_t stream);
extern "C" void* MusaGraphCounterTickKernelAddress();
extern "C" void* MusaManualGraphBiasTanhFloatKernelAddress();
extern "C" void* MusaManualGraphBiasFloatKernelAddress();
extern "C" void* MusaManualGraphResidualFloatKernelAddress();

namespace tensorflow {
namespace musa {
namespace {

constexpr int kM = 1536;
constexpr int kK = 1600;
constexpr int kH = 240;
constexpr int kActive = kM * kH;
constexpr float kPoisonValue = -123456.0f;

Status CheckMusa(musaError_t status, const char* call) {
  if (status == musaSuccess) return OkStatus();
  return errors::Internal(call, " failed with MUSA error ",
                          static_cast<int>(status));
}

Status CheckMublas(mublasStatus_t status, const char* call) {
  if (status == MUBLAS_STATUS_SUCCESS) return OkStatus();
  return errors::Internal(call, " failed with muBLAS status ",
                          static_cast<int>(status));
}

struct FullFittingCache {
  musaStream_t stream = nullptr;
  int device_id = -1;
  mublasHandle_t handle = nullptr;
  musaGraph_t graph = nullptr;
  musaGraphExec_t exec = nullptr;
  float alpha = 1.0f;
  float beta = 0.0f;
  float* x_dev = nullptr;
  float* w0_dev = nullptr;
  float* b0_dev = nullptr;
  float* w1_dev = nullptr;
  float* b1_dev = nullptr;
  float* w2_dev = nullptr;
  float* b2_dev = nullptr;
  float* wf_dev = nullptr;
  float* bf_dev = nullptr;
  float* idt1_dev = nullptr;
  float* idt2_dev = nullptr;
  float* h0g_dev = nullptr;
  float* h0_dev = nullptr;
  float* h1g_dev = nullptr;
  float* h1_dev = nullptr;
  float* h2g_dev = nullptr;
  float* h2_dev = nullptr;
  float* add2_dev = nullptr;
  float* add3_dev = nullptr;
  float* fg_dev = nullptr;
  float* final_dev = nullptr;
  unsigned int* counter = nullptr;
  size_t node_count = 0;
  size_t kernel_count = 0;
  size_t mublas_kernel_count = 0;
  size_t bias_tanh_kernel_count = 0;
  size_t bias_kernel_count = 0;
  size_t residual_kernel_count = 0;
  int launch_count = 0;
};

void CleanupCache(FullFittingCache* cache) {
  if (cache == nullptr) return;
  if (cache->stream != nullptr) {
    musaStreamCaptureStatus capture_status = musaStreamCaptureStatusNone;
    if (musaStreamIsCapturing(cache->stream, &capture_status) == musaSuccess &&
        capture_status != musaStreamCaptureStatusNone) {
      musaGraph_t aborted = nullptr;
      const musaError_t end_status =
          musaStreamEndCapture(cache->stream, &aborted);
      if (end_status != musaSuccess) {
        std::fprintf(stderr,
                     "FULL_FITTING_CLEANUP EndCapture failed err=%d\n",
                     static_cast<int>(end_status));
      }
      if (aborted != nullptr) {
        const musaError_t destroy_status = musaGraphDestroy(aborted);
        if (destroy_status != musaSuccess) {
          std::fprintf(stderr,
                       "FULL_FITTING_CLEANUP aborted destroy failed err=%d\n",
                       static_cast<int>(destroy_status));
        }
      }
    }
    const musaError_t sync_status = musaStreamSynchronize(cache->stream);
    if (sync_status != musaSuccess) {
      std::fprintf(stderr,
                   "FULL_FITTING_CLEANUP sync failed err=%d; "
                   "skip device teardown\n",
                   static_cast<int>(sync_status));
      return;
    }
  }
  if (cache->exec != nullptr) {
    const musaError_t destroy_status = musaGraphExecDestroy(cache->exec);
    if (destroy_status != musaSuccess) {
      std::fprintf(stderr,
                   "FULL_FITTING_CLEANUP exec destroy failed err=%d\n",
                   static_cast<int>(destroy_status));
    }
    cache->exec = nullptr;
  }
  if (cache->graph != nullptr) {
    const musaError_t destroy_status = musaGraphDestroy(cache->graph);
    if (destroy_status != musaSuccess) {
      std::fprintf(stderr,
                   "FULL_FITTING_CLEANUP graph destroy failed err=%d\n",
                   static_cast<int>(destroy_status));
    }
    cache->graph = nullptr;
  }
  if (cache->handle != nullptr) {
    const mublasStatus_t destroy_status = mublasDestroy(cache->handle);
    if (destroy_status != MUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr,
                   "FULL_FITTING_CLEANUP mublas destroy failed status=%d\n",
                   static_cast<int>(destroy_status));
    }
    cache->handle = nullptr;
  }
  auto free_buffer = [](void** ptr) {
    if (*ptr != nullptr) {
      const musaError_t free_status = musaFree(*ptr);
      if (free_status != musaSuccess) {
        std::fprintf(stderr,
                     "FULL_FITTING_CLEANUP free failed err=%d ptr=%p\n",
                     static_cast<int>(free_status), *ptr);
      }
      *ptr = nullptr;
    }
  };
  free_buffer(reinterpret_cast<void**>(&cache->x_dev));
  free_buffer(reinterpret_cast<void**>(&cache->w0_dev));
  free_buffer(reinterpret_cast<void**>(&cache->b0_dev));
  free_buffer(reinterpret_cast<void**>(&cache->w1_dev));
  free_buffer(reinterpret_cast<void**>(&cache->b1_dev));
  free_buffer(reinterpret_cast<void**>(&cache->w2_dev));
  free_buffer(reinterpret_cast<void**>(&cache->b2_dev));
  free_buffer(reinterpret_cast<void**>(&cache->wf_dev));
  free_buffer(reinterpret_cast<void**>(&cache->bf_dev));
  free_buffer(reinterpret_cast<void**>(&cache->idt1_dev));
  free_buffer(reinterpret_cast<void**>(&cache->idt2_dev));
  free_buffer(reinterpret_cast<void**>(&cache->h0g_dev));
  free_buffer(reinterpret_cast<void**>(&cache->h0_dev));
  free_buffer(reinterpret_cast<void**>(&cache->h1g_dev));
  free_buffer(reinterpret_cast<void**>(&cache->h1_dev));
  free_buffer(reinterpret_cast<void**>(&cache->h2g_dev));
  free_buffer(reinterpret_cast<void**>(&cache->h2_dev));
  free_buffer(reinterpret_cast<void**>(&cache->add2_dev));
  free_buffer(reinterpret_cast<void**>(&cache->add3_dev));
  free_buffer(reinterpret_cast<void**>(&cache->fg_dev));
  free_buffer(reinterpret_cast<void**>(&cache->final_dev));
  free_buffer(reinterpret_cast<void**>(&cache->counter));
}

class MusaManualGraphFullFittingOp : public OpKernel {
 public:
  explicit MusaManualGraphFullFittingOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  ~MusaManualGraphFullFittingOp() override {
    std::lock_guard<std::mutex> lock(mu_);
    CleanupCache(cache_.get());
    cache_.reset();
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& w0 = ctx->input(1);
    const Tensor& b0 = ctx->input(2);
    const Tensor& w1 = ctx->input(3);
    const Tensor& b1 = ctx->input(4);
    const Tensor& w2 = ctx->input(5);
    const Tensor& b2 = ctx->input(6);
    const Tensor& wf = ctx->input(7);
    const Tensor& bf = ctx->input(8);
    const Tensor& idt1 = ctx->input(9);
    const Tensor& idt2 = ctx->input(10);
    OP_REQUIRES(
        ctx,
        x.dtype() == DT_FLOAT && w0.dtype() == DT_FLOAT &&
            b0.dtype() == DT_FLOAT && w1.dtype() == DT_FLOAT &&
            b1.dtype() == DT_FLOAT && w2.dtype() == DT_FLOAT &&
            b2.dtype() == DT_FLOAT && wf.dtype() == DT_FLOAT &&
            bf.dtype() == DT_FLOAT && idt1.dtype() == DT_FLOAT &&
            idt2.dtype() == DT_FLOAT,
        errors::InvalidArgument("full fitting inputs must be float"));
    OP_REQUIRES(ctx, x.shape() == TensorShape({kM, kK}),
                errors::InvalidArgument("x shape mismatch"));
    OP_REQUIRES(ctx, w0.shape() == TensorShape({kK, kH}) &&
                         b0.shape() == TensorShape({kH}),
                errors::InvalidArgument("layer0 shape mismatch"));
    OP_REQUIRES(ctx, w1.shape() == TensorShape({kH, kH}) &&
                         b1.shape() == TensorShape({kH}),
                errors::InvalidArgument("layer1 shape mismatch"));
    OP_REQUIRES(ctx, w2.shape() == TensorShape({kH, kH}) &&
                         b2.shape() == TensorShape({kH}),
                errors::InvalidArgument("layer2 shape mismatch"));
    OP_REQUIRES(ctx, wf.shape() == TensorShape({kH, 1}) &&
                         bf.shape() == TensorShape({1}),
                errors::InvalidArgument("final layer shape mismatch"));
    OP_REQUIRES(ctx, idt1.shape() == TensorShape({kH}) &&
                         idt2.shape() == TensorShape({kH}),
                errors::InvalidArgument("residual idt shape mismatch"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({kM, 1}), &output));
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    const int device_id = GetMusaDeviceIdByCtx(ctx);
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("full fitting stream is null"));

    std::lock_guard<std::mutex> lock(mu_);
    const bool cache_miss = cache_ == nullptr;
    std::unique_ptr<FullFittingCache> pending_cache;
    FullFittingCache* cache = cache_.get();
    if (cache == nullptr) {
      pending_cache = std::make_unique<FullFittingCache>();
      cache = pending_cache.get();
    } else {
      OP_REQUIRES(
          ctx,
          cache->stream == stream && cache->device_id == device_id,
          errors::Internal(
              "full fitting cache stream/device mismatch; refusing reuse"));
    }
    cache->stream = stream;
    cache->device_id = device_id;
    mublasHandle_t& handle = cache->handle;
    musaGraph_t& graph = cache->graph;
    musaGraphExec_t& exec = cache->exec;
    float& alpha = cache->alpha;
    float& beta = cache->beta;
    float*& x_dev = cache->x_dev;
    float*& w0_dev = cache->w0_dev;
    float*& b0_dev = cache->b0_dev;
    float*& w1_dev = cache->w1_dev;
    float*& b1_dev = cache->b1_dev;
    float*& w2_dev = cache->w2_dev;
    float*& b2_dev = cache->b2_dev;
    float*& wf_dev = cache->wf_dev;
    float*& bf_dev = cache->bf_dev;
    float*& idt1_dev = cache->idt1_dev;
    float*& idt2_dev = cache->idt2_dev;
    float*& h0g_dev = cache->h0g_dev;
    float*& h0_dev = cache->h0_dev;
    float*& h1g_dev = cache->h1g_dev;
    float*& h1_dev = cache->h1_dev;
    float*& h2g_dev = cache->h2g_dev;
    float*& h2_dev = cache->h2_dev;
    float*& add2_dev = cache->add2_dev;
    float*& add3_dev = cache->add3_dev;
    float*& fg_dev = cache->fg_dev;
    float*& final_dev = cache->final_dev;
    unsigned int*& counter = cache->counter;

    auto alloc = [&](float** ptr, size_t bytes, const char* name) {
      return CheckMusa(musaMalloc(reinterpret_cast<void**>(ptr), bytes), name);
    };
    auto fail = [&](const Status& status) {
      CleanupCache(cache);
      if (cache == cache_.get()) {
        cache_.reset();
      } else {
        pending_cache.reset();
      }
      ctx->CtxFailure(status);
    };

    Status status;
    if (cache_miss) {
    status = alloc(&x_dev, sizeof(float) * kM * kK, "musaMalloc(x)");
    if (!status.ok()) return fail(status);
    status = alloc(&w0_dev, sizeof(float) * kK * kH, "musaMalloc(w0)");
    if (!status.ok()) return fail(status);
    status = alloc(&b0_dev, sizeof(float) * kH, "musaMalloc(b0)");
    if (!status.ok()) return fail(status);
    status = alloc(&w1_dev, sizeof(float) * kH * kH, "musaMalloc(w1)");
    if (!status.ok()) return fail(status);
    status = alloc(&b1_dev, sizeof(float) * kH, "musaMalloc(b1)");
    if (!status.ok()) return fail(status);
    status = alloc(&w2_dev, sizeof(float) * kH * kH, "musaMalloc(w2)");
    if (!status.ok()) return fail(status);
    status = alloc(&b2_dev, sizeof(float) * kH, "musaMalloc(b2)");
    if (!status.ok()) return fail(status);
    status = alloc(&wf_dev, sizeof(float) * kH, "musaMalloc(wf)");
    if (!status.ok()) return fail(status);
    status = alloc(&bf_dev, sizeof(float), "musaMalloc(bf)");
    if (!status.ok()) return fail(status);
    status = alloc(&idt1_dev, sizeof(float) * kH, "musaMalloc(idt1)");
    if (!status.ok()) return fail(status);
    status = alloc(&idt2_dev, sizeof(float) * kH, "musaMalloc(idt2)");
    if (!status.ok()) return fail(status);
    status = alloc(&h0g_dev, sizeof(float) * kActive, "musaMalloc(h0g)");
    if (!status.ok()) return fail(status);
    status = alloc(&h0_dev, sizeof(float) * kActive, "musaMalloc(h0)");
    if (!status.ok()) return fail(status);
    status = alloc(&h1g_dev, sizeof(float) * kActive, "musaMalloc(h1g)");
    if (!status.ok()) return fail(status);
    status = alloc(&h1_dev, sizeof(float) * kActive, "musaMalloc(h1)");
    if (!status.ok()) return fail(status);
    status = alloc(&h2g_dev, sizeof(float) * kActive, "musaMalloc(h2g)");
    if (!status.ok()) return fail(status);
    status = alloc(&h2_dev, sizeof(float) * kActive, "musaMalloc(h2)");
    if (!status.ok()) return fail(status);
    status = alloc(&add2_dev, sizeof(float) * kActive, "musaMalloc(add2)");
    if (!status.ok()) return fail(status);
    status = alloc(&add3_dev, sizeof(float) * kActive, "musaMalloc(add3)");
    if (!status.ok()) return fail(status);
    status = alloc(&fg_dev, sizeof(float) * kM, "musaMalloc(fg)");
    if (!status.ok()) return fail(status);
    status = alloc(&final_dev, sizeof(float) * kM, "musaMalloc(final)");
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
    }

    auto copy_input = [&](float* dst, const Tensor& src, const char* name) {
      return CheckMusa(
          musaMemcpyAsync(dst, src.flat<float>().data(), src.TotalBytes(),
                          musaMemcpyDeviceToDevice, stream),
          name);
    };
    status = copy_input(x_dev, x, "copy(x)");
    if (!status.ok()) return fail(status);
    status = copy_input(w0_dev, w0, "copy(w0)");
    if (!status.ok()) return fail(status);
    status = copy_input(b0_dev, b0, "copy(b0)");
    if (!status.ok()) return fail(status);
    status = copy_input(w1_dev, w1, "copy(w1)");
    if (!status.ok()) return fail(status);
    status = copy_input(b1_dev, b1, "copy(b1)");
    if (!status.ok()) return fail(status);
    status = copy_input(w2_dev, w2, "copy(w2)");
    if (!status.ok()) return fail(status);
    status = copy_input(b2_dev, b2, "copy(b2)");
    if (!status.ok()) return fail(status);
    status = copy_input(wf_dev, wf, "copy(wf)");
    if (!status.ok()) return fail(status);
    status = copy_input(bf_dev, bf, "copy(bf)");
    if (!status.ok()) return fail(status);
    status = copy_input(idt1_dev, idt1, "copy(idt1)");
    if (!status.ok()) return fail(status);
    status = copy_input(idt2_dev, idt2, "copy(idt2)");
    if (!status.ok()) return fail(status);
    if (cache_miss) {
      status = CheckMusa(musaStreamSynchronize(stream), "sync(inputs)");
      if (!status.ok()) return fail(status);
    }

    auto run_sequence = [&]() -> Status {
      Status s = CheckMublas(
          mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, kH, kM, kK, &alpha,
                      w0_dev, kH, x_dev, kK, &beta, h0g_dev, kH),
          "mublasSgemm(layer0)");
      if (!s.ok()) return s;
      LaunchMusaManualGraphBiasTanhFloat(h0g_dev, b0_dev, h0g_dev, h0_dev,
                                         kActive, kH, stream);
      s = CheckMusa(musaGetLastError(), "launch(bias_tanh0)");
      if (!s.ok()) return s;

      s = CheckMublas(
          mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, kH, kM, kH, &alpha,
                      w1_dev, kH, h0_dev, kH, &beta, h1g_dev, kH),
          "mublasSgemm(layer1)");
      if (!s.ok()) return s;
      LaunchMusaManualGraphBiasTanhFloat(h1g_dev, b1_dev, h1g_dev, h1_dev,
                                         kActive, kH, stream);
      s = CheckMusa(musaGetLastError(), "launch(bias_tanh1)");
      if (!s.ok()) return s;
      LaunchMusaManualGraphResidualFloat(h0_dev, h1_dev, idt1_dev, add2_dev,
                                         kActive, kH, stream);
      s = CheckMusa(musaGetLastError(), "launch(residual1)");
      if (!s.ok()) return s;

      s = CheckMublas(
          mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, kH, kM, kH, &alpha,
                      w2_dev, kH, add2_dev, kH, &beta, h2g_dev, kH),
          "mublasSgemm(layer2)");
      if (!s.ok()) return s;
      LaunchMusaManualGraphBiasTanhFloat(h2g_dev, b2_dev, h2g_dev, h2_dev,
                                         kActive, kH, stream);
      s = CheckMusa(musaGetLastError(), "launch(bias_tanh2)");
      if (!s.ok()) return s;
      LaunchMusaManualGraphResidualFloat(add2_dev, h2_dev, idt2_dev, add3_dev,
                                         kActive, kH, stream);
      s = CheckMusa(musaGetLastError(), "launch(residual2)");
      if (!s.ok()) return s;

      s = CheckMublas(
          mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, 1, kM, kH, &alpha,
                      wf_dev, 1, add3_dev, kH, &beta, fg_dev, 1),
          "mublasSgemm(final)");
      if (!s.ok()) return s;
      LaunchMusaManualGraphBiasFloat(fg_dev, bf_dev, final_dev, kM, 1, stream);
      s = CheckMusa(musaGetLastError(), "launch(bias_final)");
      if (!s.ok()) return s;
      LaunchMusaGraphCounterTick(counter, stream);
      return CheckMusa(musaGetLastError(), "launch(tick)");
    };

    if (cache_miss) {
    status = run_sequence();
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream), "sync(warmup)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaMemset(counter, 0, sizeof(*counter)),
                       "musaMemset(counter_after_warmup)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(musaStreamSynchronize(stream), "sync(counter_reset)");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaStreamBeginCapture(stream, musaStreamCaptureModeRelaxed),
        "musaStreamBeginCapture(full_fitting)");
    if (!status.ok()) return fail(status);
    status = run_sequence();
    if (!status.ok()) {
      musaGraph_t aborted = nullptr;
      musaStreamEndCapture(stream, &aborted);
      if (aborted != nullptr) musaGraphDestroy(aborted);
      return fail(status);
    }
    status = CheckMusa(musaStreamEndCapture(stream, &graph),
                       "musaStreamEndCapture(full_fitting)");
    if (!status.ok()) return fail(status);

    auto poison = [&](float* buffer, int elements, const char* name) {
      LaunchMusaManualGraphPoisonFloat(buffer, elements, kPoisonValue, stream);
      return CheckMusa(musaGetLastError(), name);
    };
    Status poison_status = poison(h0g_dev, kActive, "poison(h0g)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(h0_dev, kActive, "poison(h0)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(h1g_dev, kActive, "poison(h1g)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(h1_dev, kActive, "poison(h1)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(h2g_dev, kActive, "poison(h2g)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(h2_dev, kActive, "poison(h2)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(add2_dev, kActive, "poison(add2)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(add3_dev, kActive, "poison(add3)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(fg_dev, kM, "poison(fg)");
    if (!poison_status.ok()) return fail(poison_status);
    poison_status = poison(final_dev, kM, "poison(final)");
    if (!poison_status.ok()) return fail(poison_status);
    status = CheckMusa(musaStreamSynchronize(stream), "sync(poison)");
    if (!status.ok()) return fail(status);

    std::vector<float> poisoned_output(kM);
    status = CheckMusa(musaMemcpy(poisoned_output.data(), final_dev,
                                  sizeof(float) * kM,
                                  musaMemcpyDeviceToHost),
                       "musaMemcpy(poison_check)");
    if (!status.ok()) return fail(status);
    for (float value : poisoned_output) {
      if (value != kPoisonValue) {
        return fail(errors::Internal("full fitting poison check failed"));
      }
    }

    size_t node_count = 0;
    status = CheckMusa(musaGraphGetNodes(graph, nullptr, &node_count),
                       "musaGraphGetNodes(full_fitting)");
    if (!status.ok()) return fail(status);
    std::vector<musaGraphNode_t> nodes(node_count);
    size_t fetched = node_count;
    status = CheckMusa(
        musaGraphGetNodes(graph, nodes.data(), &fetched),
        "musaGraphGetNodes(full_fitting,data)");
    if (!status.ok()) return fail(status);
    size_t kernel_count = 0;
    size_t mublas_kernel_count = 0;
    size_t matrix_trans_kernel_count = 0;
    size_t descriptor_copy_kernel_count = 0;
    size_t tme_sgemm_kernel_count = 0;
    size_t small_gemm_kernel_count = 0;
    size_t unknown_kernel_count = 0;
    size_t bias_tanh_kernel_count = 0;
    size_t bias_kernel_count = 0;
    size_t residual_kernel_count = 0;
    size_t tick_kernel_count = 0;
    void* bias_tanh_kernel = MusaManualGraphBiasTanhFloatKernelAddress();
    void* bias_kernel = MusaManualGraphBiasFloatKernelAddress();
    void* residual_kernel = MusaManualGraphResidualFloatKernelAddress();
    void* tick_kernel = MusaGraphCounterTickKernelAddress();
    if (bias_tanh_kernel == nullptr || bias_kernel == nullptr ||
        residual_kernel == nullptr || tick_kernel == nullptr) {
      return fail(errors::Internal("full fitting kernel address is null"));
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
      musaGraphNodeType type;
      status = CheckMusa(musaGraphNodeGetType(nodes[i], &type),
                         "musaGraphNodeGetType(full_fitting)");
      if (!status.ok()) return fail(status);
      if (type != musaGraphNodeTypeKernel) continue;
      ++kernel_count;
      musaKernelNodeParams params{};
      status = CheckMusa(
          musaGraphKernelNodeGetParams(nodes[i], &params),
          "musaGraphKernelNodeGetParams(full_fitting)");
      if (!status.ok()) return fail(status);
      const char* name = "<unavailable>";
      if (musaFuncGetName(&name, params.func) != musaSuccess) {
        name = "<unavailable>";
      }
      if (params.func == bias_tanh_kernel) {
        ++bias_tanh_kernel_count;
      } else if (params.func == bias_kernel) {
        ++bias_kernel_count;
      } else if (params.func == residual_kernel) {
        ++residual_kernel_count;
      } else if (params.func == tick_kernel) {
        ++tick_kernel_count;
      } else {
        ++mublas_kernel_count;
        if (std::strstr(name, "matrixTrans_Kernel") != nullptr) {
          ++matrix_trans_kernel_count;
        } else if (std::strstr(name, "copy_tensor_descriptors") != nullptr) {
          ++descriptor_copy_kernel_count;
        } else if (std::strstr(name, "sgemm_nt_256_128x256_FullBC_ph1_tme") !=
                   nullptr) {
          ++tme_sgemm_kernel_count;
        } else if (std::strstr(name, "gemm_small_32x32_kernel") != nullptr) {
          ++small_gemm_kernel_count;
        } else {
          ++unknown_kernel_count;
        }
      }
      std::fprintf(
          stderr,
          "FULL_FITTING_KERNEL index=%zu func=%p name=%s "
          "grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u\n",
          i, params.func, name, params.gridDim.x, params.gridDim.y,
          params.gridDim.z, params.blockDim.x, params.blockDim.y,
          params.blockDim.z, params.sharedMemBytes);
    }
    if (mublas_kernel_count != 6 || bias_tanh_kernel_count != 3 ||
        bias_kernel_count != 1 || residual_kernel_count != 2 ||
        tick_kernel_count != 1 || matrix_trans_kernel_count != 1 ||
        descriptor_copy_kernel_count != 1 || tme_sgemm_kernel_count != 1 ||
        small_gemm_kernel_count != 3 || unknown_kernel_count != 0 ||
        kernel_count != 13 || node_count != 13 || fetched != node_count) {
      return fail(errors::Internal(
          "full fitting kernel composition mismatch: mublas=",
          mublas_kernel_count, " bias_tanh=", bias_tanh_kernel_count,
          " bias=", bias_kernel_count, " residual=", residual_kernel_count,
          " tick=", tick_kernel_count, " matrix_trans=",
          matrix_trans_kernel_count, " descriptor_copy=",
          descriptor_copy_kernel_count, " tme_sgemm=",
          tme_sgemm_kernel_count, " small_gemm=", small_gemm_kernel_count,
          " unknown=", unknown_kernel_count, " kernels=", kernel_count,
          " nodes=", node_count, " fetched=", fetched));
    }

    status = CheckMusa(musaGraphInstantiate(&exec, graph, 0),
                       "musaGraphInstantiate(full_fitting)");
    if (!status.ok()) return fail(status);
    cache->node_count = node_count;
    cache->kernel_count = kernel_count;
    cache->mublas_kernel_count = mublas_kernel_count;
    cache->bias_tanh_kernel_count = bias_tanh_kernel_count;
    cache->bias_kernel_count = bias_kernel_count;
    cache->residual_kernel_count = residual_kernel_count;
    cache_ = std::move(pending_cache);
    }

    status = CheckMusa(musaGraphLaunch(exec, stream), "musaGraphLaunch");
    if (!status.ok()) return fail(status);
    status = CheckMusa(
        musaMemcpyAsync(output->flat<float>().data(), final_dev,
                        sizeof(float) * kM, musaMemcpyDeviceToDevice, stream),
        "copy(final)");
    if (!status.ok()) return fail(status);
    ++cache->launch_count;
    std::fprintf(stderr,
                 "FULL_FITTING_CACHE %s=1 launch=%d nodes=%zu kernels=%zu "
                 "mublas=%zu bias_tanh=%zu bias=%zu residual=%zu "
                 "poisoned=%d shape=%dx1\n",
                 cache_miss ? "miss" : "hit", cache->launch_count,
                 cache->node_count, cache->kernel_count,
                 cache->mublas_kernel_count, cache->bias_tanh_kernel_count,
                 cache->bias_kernel_count, cache->residual_kernel_count,
                 cache_miss ? 1 : 0, kM);
  }

 private:
  std::mutex mu_;
  std::unique_ptr<FullFittingCache> cache_;
};

}  // namespace

REGISTER_OP("MusaManualGraphFullFitting")
    .Input("x: float")
    .Input("w0: float")
    .Input("b0: float")
    .Input("w1: float")
    .Input("b1: float")
    .Input("w2: float")
    .Input("b2: float")
    .Input("wf: float")
    .Input("bf: float")
    .Input("idt1: float")
    .Input("idt2: float")
    .Output("output: float")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({kM, 1}));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(
    Name("MusaManualGraphFullFitting").Device("MUSA"),
    MusaManualGraphFullFittingOp);

}  // namespace musa
}  // namespace tensorflow
