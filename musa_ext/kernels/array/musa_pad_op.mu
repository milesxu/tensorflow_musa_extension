// MUSA Pad Custom Kernel
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>

#include "musa_pad_params.h"

#ifndef MAX_DIM
#define MAX_DIM 4  // NHWC 4d support only
#endif

template <typename T>
__global__ void nd_pad_kernel(const T *__restrict__ in, T *__restrict__ out,
                              MusaPadParams params, const T pad_val) {
  const int64_t stride = blockDim.x * gridDim.x;
  for (int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
       global_idx < params.total_out_elements; global_idx += stride) {
    int64_t in_stride[MAX_DIM] = {1, 1, 1, 1};
    int64_t out_stride[MAX_DIM] = {1, 1, 1, 1};
    for (int d = params.dims - 2; d >= 0; d--) {
      in_stride[d] = in_stride[d + 1] * params.in_dims[d + 1];
      out_stride[d] = out_stride[d + 1] * params.out_dims[d + 1];
    }

    int64_t remain = global_idx;
    int64_t in_idx = 0;
    bool is_inside = true;

    for (int d = 0; d < params.dims; ++d) {
      int64_t out_coord = remain / out_stride[d];
      remain = remain % out_stride[d];

      int64_t in_coord = out_coord - params.pad_before[d];
      if (in_coord < 0 || in_coord >= params.in_dims[d]) {
        is_inside = false;
        break;
      }

      in_idx += in_coord * in_stride[d];
    }

    if (is_inside) {
      out[global_idx] = in[in_idx];
    } else {
      out[global_idx] = pad_val;
    }
  }
}

#define INSTANTIATE_ND_PAD_KERNEL(T) \
  template __global__ void nd_pad_kernel<T>( \
      const T *__restrict__ in, T *__restrict__ out, MusaPadParams params, \
      const T pad_val);

INSTANTIATE_ND_PAD_KERNEL(float)
INSTANTIATE_ND_PAD_KERNEL(double)
INSTANTIATE_ND_PAD_KERNEL(int32_t)
INSTANTIATE_ND_PAD_KERNEL(int64_t)
INSTANTIATE_ND_PAD_KERNEL(uint8_t)
INSTANTIATE_ND_PAD_KERNEL(uint16_t)

#undef INSTANTIATE_ND_PAD_KERNEL

extern "C" {
#define DEFINE_ND_PAD_LAUNCHER(T, name_suffix) \
  void nd_pad_kernel_launcher_##name_suffix(const T *input_data, T *output_data, \
                                            MusaPadParams params, const T pad_value, \
                                            const musaStream_t stream) { \
    const int block_size = 256; \
    const int64_t blocks = \
        (params.total_out_elements + block_size - 1) / block_size; \
    const int grid_size = \
        blocks > 65535 ? 65535 : static_cast<int>(blocks); \
    nd_pad_kernel<T><<<grid_size, block_size, 0, stream>>>( \
        input_data, output_data, params, pad_value); \
  }

DEFINE_ND_PAD_LAUNCHER(float, float)
DEFINE_ND_PAD_LAUNCHER(double, double)
DEFINE_ND_PAD_LAUNCHER(int32_t, int32)
DEFINE_ND_PAD_LAUNCHER(int64_t, int64)
DEFINE_ND_PAD_LAUNCHER(uint8_t, uint8)

// uint16_t launcher used by both half and bfloat16 (same bit-width, zero = 0x0000)
DEFINE_ND_PAD_LAUNCHER(uint16_t, uint16)

#undef DEFINE_ND_PAD_LAUNCHER
} // extern "C"
