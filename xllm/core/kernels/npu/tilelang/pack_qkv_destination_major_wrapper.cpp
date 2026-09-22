/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_PACK_QKV_DESTINATION_MAJOR_REGISTRY_INC
#error "XLLM_TL_PACK_QKV_DESTINATION_MAJOR_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_PACK_QKV_DESTINATION_MAJOR_REGISTRY_INC

PackQkvDestinationMajorSpecialization build_runtime_specialization(
    const torch::Tensor& query) {
  return make_pack_qkv_destination_major_specialization(
      PackQkvDestinationMajorDType{to_tilelang_dtype(query.scalar_type())});
}

bool have_matching_input_contract(const torch::Tensor& query,
                                  const torch::Tensor& key,
                                  const torch::Tensor& value) {
  return query.defined() && key.defined() && value.defined() &&
         query.dim() == 4 && query.sizes() == key.sizes() &&
         query.sizes() == value.sizes() && query.device() == key.device() &&
         query.device() == value.device() &&
         query.scalar_type() == key.scalar_type() &&
         query.scalar_type() == value.scalar_type() && query.is_contiguous() &&
         key.is_contiguous() && value.is_contiguous();
}

}  // namespace

bool can_pack_qkv_destination_major(const torch::Tensor& query,
                                    const torch::Tensor& key,
                                    const torch::Tensor& value,
                                    const torch::Tensor& packed_output,
                                    int64_t world_size) {
  if (!have_matching_input_contract(query, key, value) ||
      !packed_output.defined() ||
      query.device().type() != c10::DeviceType::PrivateUse1 ||
      query.scalar_type() != torch::kBFloat16 ||
      packed_output.device() != query.device() ||
      packed_output.scalar_type() != query.scalar_type() ||
      !packed_output.is_contiguous() || world_size <= 0 || query.size(0) <= 0 ||
      query.size(1) <= 0 || query.size(2) <= 0 || query.size(3) <= 0 ||
      query.size(2) % world_size != 0) {
    return false;
  }

  const int64_t local_head_num = query.size(2) / world_size;
  if (world_size != 2 || query.size(2) != 32 || local_head_num != 16 ||
      query.size(3) != 128) {
    return false;
  }
  const std::vector<int64_t> expected_output_shape = {world_size,
                                                      query.size(1),
                                                      query.size(0),
                                                      local_head_num,
                                                      3 * query.size(3)};
  if (packed_output.sizes().vec() != expected_output_shape) {
    return false;
  }

  return query.size(0) <= std::numeric_limits<int32_t>::max() &&
         query.size(1) <= std::numeric_limits<int32_t>::max() &&
         query.size(2) <= std::numeric_limits<int32_t>::max() &&
         local_head_num <= std::numeric_limits<int32_t>::max() &&
         query.size(3) <= std::numeric_limits<int32_t>::max() &&
         find_pack_qkv_destination_major_kernel_entry(
             build_runtime_specialization(query)) != nullptr;
}

void pack_qkv_destination_major(const torch::Tensor& query,
                                const torch::Tensor& key,
                                const torch::Tensor& value,
                                torch::Tensor& packed_output,
                                int64_t world_size) {
  CHECK(can_pack_qkv_destination_major(
      query, key, value, packed_output, world_size))
      << "TileLang pack_qkv_destination_major: unsupported tensor contract";

  const PackQkvDestinationMajorSpecialization specialization =
      build_runtime_specialization(query);
  const auto* entry =
      find_pack_qkv_destination_major_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang pack_qkv_destination_major: no compiled variant. "
      << "Available variants: "
      << available_pack_qkv_destination_major_variant_keys();

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(query.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(const_cast<void*>(query.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(key.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(value.data_ptr())),
            reinterpret_cast<uint8_t*>(packed_output.data_ptr()),
            static_cast<int32_t>(query.size(0)),
            static_cast<int32_t>(query.size(1)),
            static_cast<int32_t>(query.size(2)),
            static_cast<int32_t>(query.size(2) / world_size),
            static_cast<int32_t>(query.size(3)),
            packed_output.numel(),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
