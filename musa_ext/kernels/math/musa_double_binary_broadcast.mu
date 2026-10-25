#include <stdint.h>

#include <musa_runtime.h>

#include "musa_double_binary_broadcast.h"

namespace {

__device__ inline int64_t InputIndex(
    int64_t out_idx, const int64_t* out_dims, int32_t out_rank,
    const int64_t* in_dims, const int64_t* in_strides, int32_t in_rank) {
  int64_t in_idx = 0;
  int64_t rem = out_idx;
  for (int32_t d = out_rank - 1; d >= 0; --d) {
    const int64_t out_dim = out_dims[d];
    const int64_t coord = out_dim == 0 ? 0 : rem % out_dim;
    rem = out_dim == 0 ? 0 : rem / out_dim;
    const int32_t in_d = d - (out_rank - in_rank);
    if (in_d >= 0) {
      const int64_t in_dim = in_dims[in_d];
      in_idx += (in_dim == 1 ? 0 : coord) * in_strides[in_d];
    }
  }
  return in_idx;
}

__global__ void MulDoubleBroadcastKernel(
    const double* a, const double* b, double* out,
    DoubleBinaryBroadcastParams params) {
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < params.out_elems; idx += stride) {
    const int64_t a_idx = InputIndex(idx, params.out_dims, params.out_rank,
                                     params.a_dims, params.a_strides,
                                     params.a_rank);
    const int64_t b_idx = InputIndex(idx, params.out_dims, params.out_rank,
                                     params.b_dims, params.b_strides,
                                     params.b_rank);
    out[idx] = a[a_idx] * b[b_idx];
  }
}

__global__ void RealDivDoubleBroadcastKernel(
    const double* a, const double* b, double* out,
    DoubleBinaryBroadcastParams params) {
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < params.out_elems; idx += stride) {
    const int64_t a_idx = InputIndex(idx, params.out_dims, params.out_rank,
                                     params.a_dims, params.a_strides,
                                     params.a_rank);
    const int64_t b_idx = InputIndex(idx, params.out_dims, params.out_rank,
                                     params.b_dims, params.b_strides,
                                     params.b_rank);
    out[idx] = a[a_idx] / b[b_idx];
  }
}

int BlocksFor(int64_t n) {
  constexpr int64_t kThreads = 256;
  const int64_t blocks = (n + kThreads - 1) / kThreads;
  return static_cast<int>(blocks > 65535 ? 65535 : blocks);
}

}  // namespace

extern "C" void LaunchMulDoubleBroadcast(
    const double* a, const double* b, double* out,
    DoubleBinaryBroadcastParams params, musaStream_t stream) {
  if (params.out_elems <= 0) return;
  MulDoubleBroadcastKernel<<<BlocksFor(params.out_elems), 256, 0, stream>>>(
      a, b, out, params);
}

extern "C" void LaunchRealDivDoubleBroadcast(
    const double* a, const double* b, double* out,
    DoubleBinaryBroadcastParams params, musaStream_t stream) {
  if (params.out_elems <= 0) return;
  RealDivDoubleBroadcastKernel<<<BlocksFor(params.out_elems), 256, 0, stream>>>(
      a, b, out, params);
}
