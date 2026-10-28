#include <musa_runtime.h>
#include <mublas.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kM = 512;
constexpr int kK = 1600;
constexpr int kH = 128;
constexpr int kNoiseElements = 1 << 20;
constexpr int kThreads = 256;

const char* DeviceError(musaError_t status) {
  return musaGetErrorString(status);
}

#define MUSA_REQUIRE(call)                                                   \
  do {                                                                       \
    const musaError_t status__ = (call);                                     \
    if (status__ != musaSuccess) {                                           \
      std::fprintf(stderr, "%s failed: %s\n", #call, DeviceError(status__)); \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

#define MUBLAS_REQUIRE(call)                                                 \
  do {                                                                       \
    const mublasStatus_t status__ = (call);                                  \
    if (status__ != MUBLAS_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "%s failed: mublas status=%d\n", #call,            \
                   static_cast<int>(status__));                               \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

__global__ void BiasTanhKernel(const float* input, const float* bias,
                               float* output, int rows, int cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (index < count) {
    const int col = index / rows;
    output[index] = tanhf(input[index] + bias[col]);
  }
}

__global__ void BiasKernel(const float* input, const float* bias, float* output,
                           int rows, int cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (index < count) {
    const int col = index / rows;
    output[index] = input[index] + bias[col];
  }
}

__global__ void ResidualKernel(const float* direct, const float* gate,
                               const float* idt, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = direct[index] + gate[index] * idt[index];
  }
}

__global__ void BackwardKernel(const float* grad, const float* activation,
                               const float* idt, float* output,
                               float* direct_grad, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float one_minus = 1.0f - activation[index] * activation[index];
    const float gated = grad[index] * idt[index] * one_minus;
    direct_grad[index] += gated;
    output[index] = gated;
  }
}

__global__ void AddVectorKernel(float* matrix, const float* vector, int rows,
                                int cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (index < count) {
    matrix[index] += vector[index / rows];
  }
}

__global__ void AddMatrixKernel(float* dst, const float* src, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    dst[index] += src[index];
  }
}

__global__ void TransposeKernel(const float* input, float* output, int rows,
                                int cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (index < count) {
    const int row = index % rows;
    const int col = index / rows;
    output[col * rows + row] = input[index];
  }
}

__global__ void TickKernel(unsigned int* counter) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(counter, 1u);
  }
}

__global__ void NoiseKernel(float* data, int count, int iterations) {
  const int stride = blockDim.x * gridDim.x;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    float value = data[index];
    for (int i = 0; i < iterations; ++i) {
      value = fmaf(value, 1.0000001f, 0.0000001f);
    }
    data[index] = value;
  }
}

void CheckLaunch(const char* name) {
  const musaError_t status = musaGetLastError();
  if (status != musaSuccess) {
    std::fprintf(stderr, "%s launch failed: %s\n", name, DeviceError(status));
    std::exit(2);
  }
}

template <typename Kernel, typename... Args>
void Launch1D(Kernel kernel, int count, const char* name, musaStream_t stream,
              Args... args) {
  const int blocks = (count + kThreads - 1) / kThreads;
  kernel<<<blocks, kThreads, 0, stream>>>(args...);
  CheckLaunch(name);
}

struct Buffers {
  float* a = nullptr;
  float* grad_a = nullptr;
  float* w0 = nullptr;
  float* w0t = nullptr;
  float* w1 = nullptr;
  float* w1t = nullptr;
  float* w2 = nullptr;
  float* w2t = nullptr;
  float* wf = nullptr;
  float* wft = nullptr;
  float* h0g = nullptr;
  float* h0 = nullptr;
  float* h1g = nullptr;
  float* h1 = nullptr;
  float* a2 = nullptr;
  float* h2g = nullptr;
  float* h2 = nullptr;
  float* a3 = nullptr;
  float* energy = nullptr;
  float* energy_grad = nullptr;
  float* g3 = nullptr;
  float* g2 = nullptr;
  float* g1 = nullptr;
  float* g0 = nullptr;
  float* wg0 = nullptr;
  float* wg1 = nullptr;
  float* wg2 = nullptr;
  float* b0 = nullptr;
  float* b1 = nullptr;
  float* b2 = nullptr;
  float* bf = nullptr;
  float* idt1 = nullptr;
  float* idt2 = nullptr;
  unsigned int* tick = nullptr;
  float* noise = nullptr;
};

void Allocate(float** pointer, size_t elements) {
  MUSA_REQUIRE(musaMalloc(reinterpret_cast<void**>(pointer),
                          elements * sizeof(float)));
  MUSA_REQUIRE(musaMemset(*pointer, 0, elements * sizeof(float)));
}

void AllocateBuffers(Buffers* b) {
  Allocate(&b->a, static_cast<size_t>(kM) * kK);
  Allocate(&b->grad_a, static_cast<size_t>(kM) * kK);
  Allocate(&b->w0, static_cast<size_t>(kK) * kH);
  Allocate(&b->w0t, static_cast<size_t>(kH) * kK);
  Allocate(&b->w1, static_cast<size_t>(kH) * kH);
  Allocate(&b->w1t, static_cast<size_t>(kH) * kH);
  Allocate(&b->w2, static_cast<size_t>(kH) * kH);
  Allocate(&b->w2t, static_cast<size_t>(kH) * kH);
  Allocate(&b->wf, kH);
  Allocate(&b->wft, kH);
  Allocate(&b->h0g, static_cast<size_t>(kM) * kH);
  Allocate(&b->h0, static_cast<size_t>(kM) * kH);
  Allocate(&b->h1g, static_cast<size_t>(kM) * kH);
  Allocate(&b->h1, static_cast<size_t>(kM) * kH);
  Allocate(&b->a2, static_cast<size_t>(kM) * kH);
  Allocate(&b->h2g, static_cast<size_t>(kM) * kH);
  Allocate(&b->h2, static_cast<size_t>(kM) * kH);
  Allocate(&b->a3, static_cast<size_t>(kM) * kH);
  Allocate(&b->energy, kM);
  Allocate(&b->energy_grad, kM);
  Allocate(&b->g3, static_cast<size_t>(kM) * kH);
  Allocate(&b->g2, static_cast<size_t>(kM) * kH);
  Allocate(&b->g1, static_cast<size_t>(kM) * kH);
  Allocate(&b->g0, static_cast<size_t>(kM) * kH);
  Allocate(&b->wg0, static_cast<size_t>(kH) * kK);
  Allocate(&b->wg1, static_cast<size_t>(kH) * kH);
  Allocate(&b->wg2, static_cast<size_t>(kH) * kH);
  Allocate(&b->b0, kH);
  Allocate(&b->b1, kH);
  Allocate(&b->b2, kH);
  Allocate(&b->bf, kM);
  Allocate(&b->idt1, static_cast<size_t>(kM) * kH);
  Allocate(&b->idt2, static_cast<size_t>(kM) * kH);
  Allocate(&b->noise, kNoiseElements);
  MUSA_REQUIRE(musaMalloc(reinterpret_cast<void**>(&b->tick), sizeof(unsigned)));
  MUSA_REQUIRE(musaMemset(b->tick, 0, sizeof(unsigned)));
}

void FreeBuffers(Buffers* b) {
  musaFree(b->a);
  musaFree(b->grad_a);
  musaFree(b->w0);
  musaFree(b->w0t);
  musaFree(b->w1);
  musaFree(b->w1t);
  musaFree(b->w2);
  musaFree(b->w2t);
  musaFree(b->wf);
  musaFree(b->wft);
  musaFree(b->h0g);
  musaFree(b->h0);
  musaFree(b->h1g);
  musaFree(b->h1);
  musaFree(b->a2);
  musaFree(b->h2g);
  musaFree(b->h2);
  musaFree(b->a3);
  musaFree(b->energy);
  musaFree(b->energy_grad);
  musaFree(b->g3);
  musaFree(b->g2);
  musaFree(b->g1);
  musaFree(b->g0);
  musaFree(b->wg0);
  musaFree(b->wg1);
  musaFree(b->wg2);
  musaFree(b->b0);
  musaFree(b->b1);
  musaFree(b->b2);
  musaFree(b->bf);
  musaFree(b->idt1);
  musaFree(b->idt2);
  musaFree(b->tick);
  musaFree(b->noise);
}

void Gemm(mublasHandle_t handle, int m, int n, int k, const float* a,
          int lda, const float* b, int ldb, float* c, int ldc) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  MUBLAS_REQUIRE(mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, m, n, k, &alpha,
                            a, lda, b, ldb, &beta, c, ldc));
}

void RunSequence(mublasHandle_t handle, const Buffers& b, musaStream_t stream) {
  const int hidden_count = kM * kH;
  const int matrix_count = kM * kK;

  Gemm(handle, kH, kM, kK, b.w0t, kH, b.a, kK, b.h0g, kH);
  Launch1D(BiasTanhKernel, hidden_count, "bias_tanh_h0", stream, b.h0g, b.b0,
           b.h0, kH, kM);
  Gemm(handle, kH, kM, kH, b.w1t, kH, b.h0, kH, b.h1g, kH);
  Launch1D(BiasTanhKernel, hidden_count, "bias_tanh_h1", stream, b.h1g, b.b1,
           b.h1, kH, kM);
  Launch1D(ResidualKernel, hidden_count, "residual_a2", stream, b.h0, b.h1,
           b.idt1, b.a2, hidden_count);
  Gemm(handle, kH, kM, kH, b.w2t, kH, b.a2, kH, b.h2g, kH);
  Launch1D(BiasTanhKernel, hidden_count, "bias_tanh_h2", stream, b.h2g, b.b2,
           b.h2, kH, kM);
  Launch1D(ResidualKernel, hidden_count, "residual_a3", stream, b.a2, b.h2,
           b.idt2, b.a3, hidden_count);
  Gemm(handle, 1, kM, kH, b.wft, 1, b.a3, kH, b.energy, 1);
  Launch1D(BiasKernel, kM, "bias_energy", stream, b.energy, b.bf, b.energy,
           1, kM);
  Launch1D(TransposeKernel, static_cast<int>(static_cast<size_t>(kK) * kH),
           "transpose_w0", stream, b.w0, b.w0t, kK, kH);
  Launch1D(TransposeKernel, kH * kH, "transpose_w1", stream, b.w1, b.w1t, kH,
           kH);
  Launch1D(TransposeKernel, kH * kH, "transpose_w2", stream, b.w2, b.w2t, kH,
           kH);

  Gemm(handle, kH, kM, 1, b.wf, kH, b.energy_grad, 1, b.g3, kH);
  Launch1D(BackwardKernel, hidden_count, "backward_h2", stream, b.g3, b.h2,
           b.idt2, b.g2, b.g2, hidden_count);
  Gemm(handle, kH, kM, kH, b.w2, kH, b.g3, kH, b.g2, kH);
  Launch1D(BackwardKernel, hidden_count, "backward_h1", stream, b.g2, b.h1,
           b.idt1, b.g1, b.g1, hidden_count);
  Gemm(handle, kH, kM, kH, b.w1, kH, b.g2, kH, b.g1, kH);
  Launch1D(BackwardKernel, hidden_count, "backward_h0", stream, b.g1, b.h0,
           b.idt1, b.g0, b.g0, hidden_count);
  Gemm(handle, kK, kM, kH, b.w0, kK, b.g1, kH, b.grad_a, kK);
  Launch1D(AddVectorKernel, matrix_count, "add_vector_grad_a", stream, b.grad_a,
           b.energy_grad, kK, kM);

  Gemm(handle, kH, kK, kM, b.g0, kH, b.a, kM, b.wg0, kH);
  Gemm(handle, kH, kH, kM, b.g1, kH, b.h0, kM, b.wg1, kH);
  Gemm(handle, kH, kH, kM, b.g2, kH, b.h1, kM, b.wg2, kH);
  Launch1D(AddMatrixKernel, hidden_count, "add_matrix_g2", stream, b.g2, b.g3,
           hidden_count);
  Launch1D(TickKernel, 1, "tick", stream, b.tick);
}

struct Sample {
  double device_us = 0.0;
  double launch_us = 0.0;
};

double Percentile(std::vector<double> values, double percentile) {
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::min<double>(values.size() - 1,
                       std::ceil(percentile * values.size()) - 1.0));
  return values[index];
}

int ParseInt(int argc, char** argv, const char* name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  }
  return fallback;
}

std::string ParseString(int argc, char** argv, const char* name,
                        const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  }
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = ParseString(argc, argv, "--mode", "graph");
  const int noise_blocks = ParseInt(argc, argv, "--noise-blocks", 0);
  const int noise_iterations = ParseInt(argc, argv, "--noise-iterations", 50000);
  const int samples = ParseInt(argc, argv, "--samples", 30);
  const int warmup = ParseInt(argc, argv, "--warmup", 10);
  if (mode != "graph" && mode != "direct") {
    std::fprintf(stderr, "mode must be graph or direct\n");
    return 2;
  }

  MUSA_REQUIRE(musaSetDevice(0));
  Buffers buffers;
  AllocateBuffers(&buffers);

  musaStream_t victim_stream = nullptr;
  musaStream_t noise_stream = nullptr;
  MUSA_REQUIRE(musaStreamCreate(&victim_stream));
  MUSA_REQUIRE(musaStreamCreate(&noise_stream));

  mublasHandle_t handle = nullptr;
  MUBLAS_REQUIRE(mublasCreate(&handle));
  MUBLAS_REQUIRE(mublasSetStream(handle, victim_stream));

  musaGraph_t graph = nullptr;
  musaGraphExec_t graph_exec = nullptr;
  MUSA_REQUIRE(
      musaStreamBeginCapture(victim_stream, musaStreamCaptureModeRelaxed));
  RunSequence(handle, buffers, victim_stream);
  MUSA_REQUIRE(musaStreamEndCapture(victim_stream, &graph));
  MUSA_REQUIRE(musaGraphInstantiate(&graph_exec, graph, 0));

  size_t node_count = 0;
  size_t kernel_count = 0;
  MUSA_REQUIRE(musaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<musaGraphNode_t> nodes(node_count);
  MUSA_REQUIRE(musaGraphGetNodes(graph, nodes.data(), &node_count));
  for (musaGraphNode_t node : nodes) {
    musaGraphNodeType type;
    MUSA_REQUIRE(musaGraphNodeGetType(node, &type));
    if (type == musaGraphNodeTypeKernel) ++kernel_count;
  }
  std::fprintf(stderr,
               "GRAPH_CONTENTION_REPRO mode=%s noise_blocks=%d nodes=%zu "
               "kernels=%zu\n",
               mode.c_str(), noise_blocks, node_count, kernel_count);

  musaEvent_t start_event = nullptr;
  musaEvent_t end_event = nullptr;
  MUSA_REQUIRE(musaEventCreate(&start_event));
  MUSA_REQUIRE(musaEventCreate(&end_event));

  auto launch_once = [&]() {
    if (mode == "graph") {
      MUSA_REQUIRE(musaGraphLaunch(graph_exec, victim_stream));
    } else {
      RunSequence(handle, buffers, victim_stream);
    }
  };

  for (int i = 0; i < warmup; ++i) {
    launch_once();
    MUSA_REQUIRE(musaStreamSynchronize(victim_stream));
  }

  std::atomic<bool> stop_noise{false};
  std::atomic<bool> noise_ready{false};
  std::thread noise_thread;
  if (noise_blocks > 0) {
    noise_thread = std::thread([&]() {
      NoiseKernel<<<noise_blocks, kThreads, 0, noise_stream>>>(
          buffers.noise, kNoiseElements, noise_iterations);
      CheckLaunch("noise");
      noise_ready.store(true, std::memory_order_release);
      MUSA_REQUIRE(musaStreamSynchronize(noise_stream));
      while (!stop_noise.load(std::memory_order_acquire)) {
        NoiseKernel<<<noise_blocks, kThreads, 0, noise_stream>>>(
            buffers.noise, kNoiseElements, noise_iterations);
        CheckLaunch("noise");
        MUSA_REQUIRE(musaStreamSynchronize(noise_stream));
      }
    });
    while (!noise_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  std::vector<Sample> results;
  results.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    MUSA_REQUIRE(musaEventRecord(start_event, victim_stream));
    const auto host_begin = std::chrono::steady_clock::now();
    launch_once();
    const auto host_end = std::chrono::steady_clock::now();
    MUSA_REQUIRE(musaEventRecord(end_event, victim_stream));
    MUSA_REQUIRE(musaEventSynchronize(end_event));
    float device_ms = 0.0f;
    MUSA_REQUIRE(
        musaEventElapsedTime(&device_ms, start_event, end_event));
    Sample sample;
    sample.device_us = static_cast<double>(device_ms) * 1000.0;
    sample.launch_us =
        std::chrono::duration<double, std::micro>(host_end - host_begin).count();
    results.push_back(sample);
    std::printf(
        "SAMPLE mode=%s noise_blocks=%d index=%d device_us=%.3f "
        "launch_us=%.3f\n",
        mode.c_str(), noise_blocks, i, sample.device_us, sample.launch_us);
  }

  if (noise_thread.joinable()) {
    stop_noise.store(true, std::memory_order_release);
    noise_thread.join();
  }

  std::vector<double> device_values;
  std::vector<double> launch_values;
  device_values.reserve(results.size());
  launch_values.reserve(results.size());
  for (const Sample& sample : results) {
    device_values.push_back(sample.device_us);
    launch_values.push_back(sample.launch_us);
  }
  const double device_median = Percentile(device_values, 0.50);
  const double device_p95 = Percentile(device_values, 0.95);
  const double device_p99 = Percentile(device_values, 0.99);
  const double device_max =
      *std::max_element(device_values.begin(), device_values.end());
  const int device_spikes = static_cast<int>(
      std::count_if(device_values.begin(), device_values.end(),
                    [](double value) { return value > 2000.0; }));
  const double launch_median = Percentile(launch_values, 0.50);
  std::printf(
      "SUMMARY mode=%s noise_blocks=%d samples=%d nodes=%zu kernels=%zu "
      "device_median_us=%.3f device_p95_us=%.3f device_p99_us=%.3f "
      "device_max_us=%.3f device_spikes_over_2ms=%d launch_median_us=%.3f\n",
      mode.c_str(), noise_blocks, samples, node_count, kernel_count,
      device_median, device_p95, device_p99, device_max, device_spikes,
      launch_median);

  MUSA_REQUIRE(musaEventDestroy(start_event));
  MUSA_REQUIRE(musaEventDestroy(end_event));
  MUSA_REQUIRE(musaGraphExecDestroy(graph_exec));
  MUSA_REQUIRE(musaGraphDestroy(graph));
  MUBLAS_REQUIRE(mublasDestroy(handle));
  MUSA_REQUIRE(musaStreamDestroy(victim_stream));
  MUSA_REQUIRE(musaStreamDestroy(noise_stream));
  FreeBuffers(&buffers);
  return 0;
}
