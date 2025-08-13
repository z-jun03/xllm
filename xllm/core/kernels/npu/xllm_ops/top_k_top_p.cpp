#include "top_k_top_p.h"

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "aclnn_apply_top_k_top_p.h"

#ifdef TORCH_HIGHER_THAN_PTA6
// #include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <nlohmann/json.hpp>

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)

namespace xllm_ops {
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP) {
  xllm_ops_utils::check_tensor(logits, "logits", "top_k_top_p");
  xllm_ops_utils::check_tensor(topK, "topK", "top_k_top_p");
  xllm_ops_utils::check_tensor(topP, "topP", "top_k_top_p");
  aclTensor* logits_ids = nullptr;
  aclTensor* topK_ids = nullptr;
  aclTensor* topP_ids = nullptr;
  int32_t device_id = logits.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  xllm_ops_utils::create_acltensor(&logits_ids, logits);
  xllm_ops_utils::create_acltensor(&topK_ids, topK);
  xllm_ops_utils::create_acltensor(&topP_ids, topP);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(aclnnApplyTopKTopPGetWorkspaceSize(logits_ids,
                                                       topP_ids,
                                                       topK_ids,
                                                       logits_ids,
                                                       &workspace_size,
                                                       &executor),
                    "top_k_top_p: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "top_k_top_p: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(
      aclnnApplyTopKTopP(workspace_addr, workspace_size, executor, stream),
      "top_k_top_p: failed to apply top k top p");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "top_k_top_p: failed to synchronize stream");
  aclDestroyTensor(logits_ids);
  aclDestroyTensor(topK_ids);
  aclDestroyTensor(topP_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "top_k_top_p: failed to free workspace");
  }
}
}  // namespace xllm_ops