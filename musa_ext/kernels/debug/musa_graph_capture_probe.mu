#include <musa_runtime.h>

extern "C" __global__ void MusaGraphCaptureProbeAccumulate(const float* x,
                                                           float* y, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += x[idx] + 1.0f;
  }
}

extern "C" void LaunchMusaGraphCaptureProbe(const float* x, float* y, int n,
                                            musaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  MusaGraphCaptureProbeAccumulate<<<blocks, kThreads, 0, stream>>>(x, y, n);
}

extern "C" __global__ void MusaGraphCounterTickKernel(
    unsigned int* counter) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(counter, 1u);
  }
}

extern "C" void LaunchMusaGraphCounterTick(unsigned int* counter,
                                           musaStream_t stream) {
  MusaGraphCounterTickKernel<<<1, 1, 0, stream>>>(counter);
}

extern "C" void* MusaGraphCounterTickKernelAddress() {
  return reinterpret_cast<void*>(MusaGraphCounterTickKernel);
}

extern "C" __global__ void MusaManualGraphReshapeKernel(
    const double* input, double* output) {
  const int idx = threadIdx.x;
  if (idx < 16) {
    output[idx] = input[idx];
  }
}

extern "C" __global__ void MusaManualGraphTanhKernel(
    const double* input, double* output) {
  const int idx = threadIdx.x;
  if (idx < 16) {
    output[idx] = tanh(input[idx]);
  }
}

extern "C" __global__ void MusaManualGraphSumKernel(
    const double* input, double* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    double sum = 0.0;
    for (int idx = 0; idx < 16; ++idx) {
      sum += input[idx];
    }
    output[0] = sum;
  }
}

extern "C" __global__ void MusaManualGraphBiasTanhKernel(
    const double* input, const double* bias, double* output,
    int elements, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = tanh(input[idx] + bias[idx % width]);
  }
}

extern "C" __global__ void MusaManualGraphBiasTanhFloatKernel(
    const float* input, const float* bias, float* pre, float* output,
    int elements, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements) {
    const float value = input[idx] + bias[idx % width];
    pre[idx] = value;
    output[idx] = tanhf(value);
  }
}

extern "C" void* MusaManualGraphReshapeKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphReshapeKernel);
}

extern "C" void* MusaManualGraphTanhKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphTanhKernel);
}

extern "C" void* MusaManualGraphSumKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphSumKernel);
}

extern "C" void* MusaManualGraphBiasTanhKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphBiasTanhKernel);
}

extern "C" void* MusaManualGraphBiasTanhFloatKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphBiasTanhFloatKernel);
}
