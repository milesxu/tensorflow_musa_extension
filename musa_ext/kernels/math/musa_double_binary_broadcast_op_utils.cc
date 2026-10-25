#include "musa_double_binary_broadcast_op_utils.h"

#include <cstring>

namespace tensorflow {
namespace musa {

Status FillDoubleBinaryBroadcastParams(
    const Tensor& a, const Tensor& b, const Tensor& out,
    DoubleBinaryBroadcastParams* params) {
  const int out_rank = out.dims();
  const int a_rank = a.dims();
  const int b_rank = b.dims();
  if (out_rank > 8 || a_rank > 8 || b_rank > 8) {
    return errors::InvalidArgument(
        "MUSA double binary broadcast supports rank <= 8");
  }

  std::memset(params, 0, sizeof(*params));
  params->out_elems = out.NumElements();
  params->out_rank = out_rank;
  params->a_rank = a_rank;
  params->b_rank = b_rank;
  for (int d = 0; d < out_rank; ++d) {
    params->out_dims[d] = out.dim_size(d);
  }
  for (int d = 0; d < a_rank; ++d) {
    params->a_dims[d] = a.dim_size(d);
  }
  for (int d = 0; d < b_rank; ++d) {
    params->b_dims[d] = b.dim_size(d);
  }
  int64_t stride = 1;
  for (int d = a_rank - 1; d >= 0; --d) {
    params->a_strides[d] = stride;
    stride *= a.dim_size(d);
  }
  stride = 1;
  for (int d = b_rank - 1; d >= 0; --d) {
    params->b_strides[d] = stride;
    stride *= b.dim_size(d);
  }
  return OkStatus();
}

}  // namespace musa
}  // namespace tensorflow
