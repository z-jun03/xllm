#include "replace_token.h"

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "aclnn_replace_token.h"
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

void replace_token(torch::Tensor& dst, torch::Tensor& src) {
  xllm_ops_utils::check_tensor(dst, "dst", "replace_token");
  xllm_ops_utils::check_tensor(src, "src", "replace_token");
  aclTensor* dst_ids = nullptr;
  aclTensor* src_ids = nullptr;
  int32_t device_id = dst.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  xllm_ops_utils::create_acltensor(&dst_ids, dst);
  xllm_ops_utils::create_acltensor(&src_ids, src);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor;
  CHECK_ACL_SUCCESS(aclnnReplaceTokenGetWorkspaceSize(
                        dst_ids, src_ids, dst_ids, &workspace_size, &executor),
                    "replace_token: failed to get workspace size");
  void* workspace_addr = nullptr;
  CHECK_ACL_SUCCESS(
      aclnnReplaceToken(workspace_addr, workspace_size, executor, stream),
      "replace_token: failed to replace token");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "replace_token: failed to synchronize stream");
  aclDestroyTensor(dst_ids);
  aclDestroyTensor(src_ids);
}
}  // namespace xllm_ops