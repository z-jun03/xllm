/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "hccl_kv_cache_transfer.h"

#include <glog/logging.h>

#include "util/net.h"

namespace xllm {

const std::unordered_map<torch::ScalarType, HcclDataType> kScalarTypeToDtype = {
    {torch::kByte, HCCL_DATA_TYPE_UINT8},
    {torch::kChar, HCCL_DATA_TYPE_INT8},
    {torch::kShort, HCCL_DATA_TYPE_INT16},
    {torch::kInt, HCCL_DATA_TYPE_INT32},
    {torch::kLong, HCCL_DATA_TYPE_INT64},
    {torch::kBFloat16, HCCL_DATA_TYPE_BFP16},
    {torch::kHalf, HCCL_DATA_TYPE_FP16},
    {torch::kFloat, HCCL_DATA_TYPE_FP32},
    {torch::kDouble, HCCL_DATA_TYPE_FP64},
};

HcclKVCacheTransfer::HcclKVCacheTransfer(const int32_t device_id,
                                         const int32_t listen_port)
    : device_id_(device_id), KVCacheTransfer() {
  std::string instance_ip = net::get_local_ip_addr();
  addr_ = instance_ip + ":" + std::to_string(listen_port);
  hccl_transfer_ = std::make_unique<HcclTransfer>(addr_, device_id);
  CHECK(hccl_transfer_->initialize(listen_port))
      << "Initialize HcclTransfer failed.";
}

void HcclKVCacheTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  const auto& it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupport data type : " << dtype;
  auto hccl_dtype = it->second;
  std::vector<void*> k_cache_addrs;
  std::vector<void*> v_cache_addrs;
  num_layers_ = kv_caches.size();
  k_cache_addrs.reserve(num_layers_);
  v_cache_addrs.reserve(num_layers_);
  for (int32_t i = 0; i < num_layers_; ++i) {
    k_cache_addrs.emplace_back(kv_caches[i].get_k_cache().data_ptr());
    v_cache_addrs.emplace_back(kv_caches[i].get_v_cache().data_ptr());
  }
  k_cache_id_ = hccl_transfer_->register_memory(
      k_cache_addrs, kv_cache_shape[0], hccl_dtype);
  v_cache_id_ = hccl_transfer_->register_memory(
      v_cache_addrs, kv_cache_shape[1], hccl_dtype);
}

void HcclKVCacheTransfer::get_cache_info(uint64_t& cluster_id,
                                         std::string& addr,
                                         int64_t& key_cache_id,
                                         int64_t& value_cache_id) {
  addr = addr_;
  key_cache_id = k_cache_id_;
  value_cache_id = v_cache_id_;
}

bool HcclKVCacheTransfer::link_cluster(const uint64_t cluster_id,
                                       const std::string& remote_addr,
                                       const std::string& device_ip,
                                       const uint16_t port) {
  if (linked_addrs_.find(remote_addr) != linked_addrs_.end()) {
    // The addr is connected.
    return true;
  }

  auto ret = hccl_transfer_->create_comm_domain(remote_addr);
  if (!ret) {
    LOG(ERROR) << "Create comm domain failed.";
    return false;
  }
  linked_addrs_.insert(remote_addr);

  return true;
}

bool HcclKVCacheTransfer::unlink_cluster(const uint64_t& cluster_id,
                                         const std::string& remote_addr,
                                         const std::string& device_ip,
                                         const uint16_t port,
                                         bool force_flag) {
  linked_addrs_.erase(remote_addr);
  return true;
}

bool HcclKVCacheTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  std::vector<int64_t> layer_ids;
  auto k_ret = hccl_transfer_->pull_memory_blocks(
      src_addr, src_k_cache_id, src_blocks, k_cache_id_, dst_blocks, layer_ids);
  auto v_ret = hccl_transfer_->pull_memory_blocks(
      src_addr, src_v_cache_id, src_blocks, v_cache_id_, dst_blocks, layer_ids);
  if (!k_ret || !v_ret) {
    LOG(ERROR) << "Pull kv cache blocks failed, k_ret = " << k_ret
               << ", v_ret = " << v_ret;
    return false;
  }
  return true;
}

bool HcclKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);
    // Push the KV Cache computed at this layer for all requests to the
    // designated worker.
    for (const auto& pair : merged_kv_infos) {
      std::vector<int64_t> layer_ids = {layer_index};
      const KVCacheInfo& kv_info = pair.second;
      auto k_ret = hccl_transfer_->push_memory_blocks(kv_info.dst_addr,
                                                      k_cache_id_,
                                                      kv_info.src_blocks,
                                                      kv_info.dst_k_cache_id,
                                                      kv_info.dst_blocks,
                                                      layer_ids);
      auto v_ret = hccl_transfer_->push_memory_blocks(kv_info.dst_addr,
                                                      v_cache_id_,
                                                      kv_info.src_blocks,
                                                      kv_info.dst_v_cache_id,
                                                      kv_info.dst_blocks,
                                                      layer_ids);
      if (!k_ret || !v_ret) {
        LOG(ERROR) << "Push kv blocks failed, layer = " << layer_index
                   << ", k_ret = " << k_ret << ", v_ret = " << v_ret;
        return false;
      }
    }
  }
  return true;
}

}  // namespace xllm
