#pragma once

#include "musa_double_binary_broadcast.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace musa {

Status FillDoubleBinaryBroadcastParams(
    const Tensor& a, const Tensor& b, const Tensor& out,
    DoubleBinaryBroadcastParams* params);

}  // namespace musa
}  // namespace tensorflow
