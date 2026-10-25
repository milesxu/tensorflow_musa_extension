#include <stdint.h>

#include <musa_runtime.h>

#include "musa_strided_slice_double.h"

namespace {

__global__ void StridedSliceDoubleKernel(
    const double* input, double* output,
    StridedSliceDoubleParams params) {
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < params.out_elems; idx += stride) {
    int64_t rem = idx;
    int64_t coords[8] = {0};
    for (int32_t d = params.proc_rank - 1; d >= 0; --d) {
      const int64_t dim = params.proc_dims[d];
      coords[d] = dim == 0 ? 0 : rem % dim;
      rem = dim == 0 ? 0 : rem / dim;
    }

    int32_t input_dim = 0;
    int64_t input_idx = 0;
    for (int32_t d = 0; d < params.proc_rank; ++d) {
      if (params.new_axis_mask & (1u << d)) continue;
      if (input_dim >= params.input_rank) break;
      const int64_t coord = params.starts[input_dim] +
                            coords[d] * params.strides[input_dim];
      input_idx += coord * params.input_strides[input_dim];
      ++input_dim;
    }
    output[idx] = input[input_idx];
  }
}

}  // namespace

extern "C" void LaunchStridedSliceDouble(
    const double* input, double* output,
    StridedSliceDoubleParams params, musaStream_t stream) {
  if (params.out_elems <= 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = (params.out_elems + kThreads - 1) / kThreads;
  StridedSliceDoubleKernel<<<(blocks > 65535 ? 65535 : blocks), kThreads, 0,
                             stream>>>(input, output, params);
}
