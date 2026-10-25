#include <stdint.h>

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

__global__ void BiasAddDoubleKernel(double* output, const double* bias,
                                    int64_t rows, int64_t cols) {
  for (int64_t row =
           static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       row < rows; row += static_cast<int64_t>(gridDim.y) * blockDim.y) {
    for (int64_t col =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         col < cols; col += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      output[row * cols + col] += bias[col];
    }
  }
}

}  // namespace musa
}  // namespace tensorflow

extern "C" musaError_t LaunchMusaBiasAddDouble(double* output,
                                               const double* bias,
                                               int64_t rows, int64_t cols,
                                               musaStream_t stream) {
  if (output == nullptr || bias == nullptr || rows <= 0 || cols <= 0) {
    return musaSuccess;
  }
  constexpr int kThreadsX = 32;
  constexpr int kThreadsY = 8;
  const dim3 block(kThreadsX, kThreadsY);
  const unsigned int grid_x =
      static_cast<unsigned int>((cols + kThreadsX - 1) / kThreadsX);
  const unsigned int grid_y =
      static_cast<unsigned int>((rows + kThreadsY - 1) / kThreadsY);
  const dim3 grid(grid_x > 65535U ? 65535U : grid_x,
                  grid_y > 65535U ? 65535U : grid_y);
  tensorflow::musa::BiasAddDoubleKernel<<<grid, block, 0, stream>>>(
      output, bias, rows, cols);
  return musaGetLastError();
}
