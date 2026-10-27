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
