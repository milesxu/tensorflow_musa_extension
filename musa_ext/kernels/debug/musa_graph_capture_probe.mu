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
