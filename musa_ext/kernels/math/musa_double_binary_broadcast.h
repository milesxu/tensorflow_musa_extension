#pragma once

#include <stdint.h>

#include <musa_runtime.h>

struct DoubleBinaryBroadcastParams {
  int64_t out_elems;
  int32_t out_rank;
  int32_t a_rank;
  int32_t b_rank;
  int64_t out_dims[8];
  int64_t a_dims[8];
  int64_t b_dims[8];
  int64_t a_strides[8];
  int64_t b_strides[8];
};

extern "C" void LaunchMulDoubleBroadcast(
    const double* a, const double* b, double* out,
    DoubleBinaryBroadcastParams params, musaStream_t stream);

extern "C" void LaunchRealDivDoubleBroadcast(
    const double* a, const double* b, double* out,
    DoubleBinaryBroadcastParams params, musaStream_t stream);
