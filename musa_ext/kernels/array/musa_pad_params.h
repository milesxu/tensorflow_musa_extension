#ifndef TENSORFLOW_MUSA_MU1_KERNELS_ARRAY_MUSA_PAD_PARAMS_H_
#define TENSORFLOW_MUSA_MU1_KERNELS_ARRAY_MUSA_PAD_PARAMS_H_

#include <cstdint>

// Small, by-value kernel parameter block. Keeping the shape and padding
// metadata in the kernel argument buffer avoids four tiny H2D copies and a
// stream synchronization for every Pad op.
struct MusaPadParams {
  int32_t dims;
  int64_t total_out_elements;
  int64_t in_dims[4];
  int64_t out_dims[4];
  int64_t pad_before[4];
  int64_t pad_after[4];
};

#endif  // TENSORFLOW_MUSA_MU1_KERNELS_ARRAY_MUSA_PAD_PARAMS_H_
