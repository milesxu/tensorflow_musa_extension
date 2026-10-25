#pragma once

#include <stdint.h>

#include <musa_runtime.h>

struct StridedSliceDoubleParams {
  int64_t out_elems;
  int32_t proc_rank;
  int32_t input_rank;
  int64_t proc_dims[8];
  int64_t input_strides[8];
  int64_t starts[8];
  int64_t strides[8];
  uint32_t new_axis_mask;
};

extern "C" void LaunchStridedSliceDouble(
    const double* input, double* output,
    StridedSliceDoubleParams params, musaStream_t stream);
