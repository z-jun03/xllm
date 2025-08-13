#pragma once
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "aclnn_replace_token.h"
#include "util/tensor_helper.h"
#include "utils/tensor_checks.h"
#include "utils/tensor_utils.h"
namespace xllm_ops {
void replace_token(torch::Tensor& forked, torch::Tensor& lastStepOutPut);
}  // namespace xllm_ops
