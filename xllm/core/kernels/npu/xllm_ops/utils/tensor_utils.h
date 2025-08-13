#pragma once
#include <torch/torch.h>

#include "acl/acl.h"

namespace xllm_ops_utils {

struct type_info {
  static aclDataType get_acl_type(const torch::ScalarType& dtype) {
    switch (dtype) {
      case torch::kInt64:
        return ACL_INT64;
      case torch::kInt32:
        return ACL_INT32;
      case torch::kFloat32:
        return ACL_FLOAT;
      case torch::kInt16:
        return ACL_INT16;
      case torch::kFloat16:
        return ACL_FLOAT16;
      case torch::kBFloat16:
        return ACL_BF16;
      case torch::kInt8:
        return ACL_INT8;
      default:
        return ACL_INT32;
    }
  }
};

inline void create_acltensor(aclTensor** tensor,
                             const torch::Tensor& tensor_data) {
  aclDataType acl_tensor_type =
      xllm_ops_utils::type_info::get_acl_type(tensor_data.scalar_type());
  void* deviceData = const_cast<void*>(tensor_data.storage().data());
  c10::SmallVector<int64_t, 8> storageDims;
  storageDims.push_back(tensor_data.storage().nbytes() /
                        tensor_data.itemsize());
  *tensor = aclCreateTensor(tensor_data.sizes().data(),
                            tensor_data.sizes().size(),
                            acl_tensor_type,
                            tensor_data.strides().data(),
                            tensor_data.storage_offset(),
                            aclFormat::ACL_FORMAT_ND,
                            storageDims.data(),
                            storageDims.size(),
                            deviceData);
  if (*tensor == nullptr) {
    LOG(ERROR) << "create_acltensor: failed to create acltensor";
    throw std::runtime_error("create_acltensor: failed to create acltensor");
  }
}
}  // namespace xllm_ops_utils