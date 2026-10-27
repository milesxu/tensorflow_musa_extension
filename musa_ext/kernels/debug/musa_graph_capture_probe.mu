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

extern "C" __global__ void MusaManualGraphBiasFloatKernel(
    const float* input, const float* bias, float* output,
    int elements, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = input[idx] + bias[idx % width];
  }
}

extern "C" __global__ void MusaManualGraphResidualFloatKernel(
    const float* previous, const float* current, const float* idt,
    float* output, int elements, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = previous[idx] + current[idx] * idt[idx % width];
  }
}

extern "C" __global__ void MusaManualGraphPoisonFloatKernel(
    float* data, int elements, float value) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements;
       idx += blockDim.x * gridDim.x) {
    data[idx] = value;
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

extern "C" void* MusaManualGraphBiasFloatKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphBiasFloatKernel);
}

extern "C" void* MusaManualGraphResidualFloatKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphResidualFloatKernel);
}

extern "C" void* MusaManualGraphPoisonFloatKernelAddress() {
  return reinterpret_cast<void*>(MusaManualGraphPoisonFloatKernel);
}

extern "C" void LaunchMusaManualGraphBiasTanhFloat(
    const float* input, const float* bias, float* pre, float* output,
    int elements, int width, musaStream_t stream) {
  MusaManualGraphBiasTanhFloatKernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      input, bias, pre, output, elements, width);
}

extern "C" void LaunchMusaManualGraphBiasFloat(
    const float* input, const float* bias, float* output,
    int elements, int width, musaStream_t stream) {
  MusaManualGraphBiasFloatKernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      input, bias, output, elements, width);
}

extern "C" void LaunchMusaManualGraphResidualFloat(
    const float* previous, const float* current, const float* idt,
    float* output, int elements, int width, musaStream_t stream) {
  MusaManualGraphResidualFloatKernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      previous, current, idt, output, elements, width);
}

extern "C" void LaunchMusaManualGraphPoisonFloat(
    float* data, int elements, float value, musaStream_t stream) {
  MusaManualGraphPoisonFloatKernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      data, elements, value);
}
