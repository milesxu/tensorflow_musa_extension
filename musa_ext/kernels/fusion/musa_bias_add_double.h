#pragma once

#include <stdint.h>

#include <musa_runtime.h>

extern "C" musaError_t LaunchMusaBiasAddDouble(double* output,
                                               const double* bias,
                                               int64_t rows, int64_t cols,
                                               musaStream_t stream);
