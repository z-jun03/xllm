#pragma once
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "aclnn_apply_top_k_top_p.h"
#include "util/tensor_helper.h"
#include "utils/tensor_checks.h"
#include "utils/tensor_utils.h"

namespace xllm_ops {
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);
}  // namespace xllm_ops