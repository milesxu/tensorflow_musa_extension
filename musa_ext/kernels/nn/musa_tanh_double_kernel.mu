#include <math.h>
#include <stdint.h>

#include <cstdlib>

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

__global__ void TanhDoubleStdKernel(const double* x, double* y, int64_t n) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    y[idx] = tanh(x[idx]);
  }
}

__global__ void TanhDoubleExpm1Kernel(const double* x, double* y, int64_t n) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const double value = x[idx];
    const double magnitude = fabs(value);
    if (magnitude >= 20.0) {
      y[idx] = copysign(1.0, value);
    } else {
      const double e = expm1(2.0 * magnitude);
      y[idx] = copysign(e / (e + 2.0), value);
    }
  }
}

__global__ void TanhGradDoubleKernel(const double* y, const double* dy,
                                     double* dz, int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dz[idx] = dy[idx] * (1.0 - y[idx] * y[idx]);
  }
}

}  // namespace musa
}  // namespace tensorflow

extern "C" void LaunchMusaTanhDouble(const double* x, double* y, int64_t n,
                                     musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  const int64_t needed_blocks = (n + kThreads - 1) / kThreads;
  const int blocks = needed_blocks > 65535 ? 65535
                                           : static_cast<int>(needed_blocks);
  static const bool disable_fast = []() {
    const char* value = std::getenv("MUSA_DISABLE_FAST_TANH_DOUBLE");
    return value != nullptr && std::atoi(value) != 0;
  }();
  if (disable_fast) {
    tensorflow::musa::TanhDoubleStdKernel<<<blocks, kThreads, 0, stream>>>(
        x, y, n);
  } else {
    tensorflow::musa::TanhDoubleExpm1Kernel<<<blocks, kThreads, 0, stream>>>(
        x, y, n);
  }
}

extern "C" void LaunchMusaTanhGradDouble(const double* y, const double* dy,
                                         double* dz, int64_t n,
                                         musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  tensorflow::musa::TanhGradDoubleKernel<<<blocks, kThreads, 0, stream>>>(
      y, dy, dz, n);
}
