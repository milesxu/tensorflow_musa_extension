#include <musa_runtime.h>

extern "C" __global__ void MusaGraphCaptureProbeAddOne(const float* x,
                                                       float* y, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1.0f;
  }
}

extern "C" __global__ void MusaGraphCaptureProbeMulTwo(float* y, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] *= 2.0f;
  }
}

extern "C" void LaunchMusaGraphCaptureProbe(const float* x, float* y, int n,
                                            musaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  MusaGraphCaptureProbeAddOne<<<blocks, kThreads, 0, stream>>>(x, y, n);
  MusaGraphCaptureProbeMulTwo<<<blocks, kThreads, 0, stream>>>(y, n);
}
