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

#include <algorithm>
#include <cstdint>
#include <limits>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_ONLINE_SOFTMAX_STATE_UPDATE_REGISTRY_INC
#error "XLLM_TL_ONLINE_SOFTMAX_STATE_UPDATE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_ONLINE_SOFTMAX_STATE_UPDATE_REGISTRY_INC

constexpr int64_t kHeadDim = 128;

OnlineSoftmaxStateUpdateSpecialization build_runtime_specialization(
    const torch::Tensor& tile_output) {
  return make_online_softmax_state_update_specialization(
      OnlineSoftmaxStateUpdateHeadDim{
          static_cast<int32_t>(tile_output.size(/*dim=*/3))},
      OnlineSoftmaxStateUpdateDType{
          to_tilelang_dtype(tile_output.scalar_type())});
}

bool fits_int32(int64_t value) {
  return value >= 0 &&
         value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

bool has_matching_state_contract(const torch::Tensor& state_max,
                                 const torch::Tensor& state_normalizer,
                                 const torch::Tensor& state_weighted_value,
                                 const torch::Tensor& tile_output,
                                 const torch::Tensor& tile_max_stats,
                                 const torch::Tensor& tile_normalizer_stats) {
  if (!state_max.defined() || !state_normalizer.defined() ||
      !state_weighted_value.defined() || !tile_output.defined() ||
      !tile_max_stats.defined() || !tile_normalizer_stats.defined()) {
    return false;
  }
  if (state_max.device().type() != c10::DeviceType::PrivateUse1 ||
      state_max.device() != state_normalizer.device() ||
      state_max.device() != state_weighted_value.device() ||
      state_max.device() != tile_output.device() ||
      state_max.device() != tile_max_stats.device() ||
      state_max.device() != tile_normalizer_stats.device()) {
    return false;
  }
  if (state_max.scalar_type() != torch::kFloat32 ||
      state_normalizer.scalar_type() != torch::kFloat32 ||
      state_weighted_value.scalar_type() != torch::kFloat32 ||
      tile_output.scalar_type() != torch::kBFloat16 ||
      tile_max_stats.scalar_type() != torch::kFloat32 ||
      tile_normalizer_stats.scalar_type() != torch::kFloat32) {
    return false;
  }
  if (state_max.dim() != 3 || state_normalizer.dim() != 3 ||
      state_weighted_value.dim() != 4 || tile_output.dim() != 4 ||
      tile_max_stats.dim() != 4 || tile_normalizer_stats.dim() != 4 ||
      !state_max.is_contiguous() || !state_normalizer.is_contiguous() ||
      !state_weighted_value.is_contiguous() || !tile_output.is_contiguous() ||
      !tile_max_stats.is_contiguous() ||
      !tile_normalizer_stats.is_contiguous()) {
    return false;
  }
  if (state_max.sizes() != state_normalizer.sizes() ||
      state_weighted_value.size(/*dim=*/0) != state_max.size(/*dim=*/0) ||
      state_weighted_value.size(/*dim=*/1) != state_max.size(/*dim=*/1) ||
      state_weighted_value.size(/*dim=*/2) != state_max.size(/*dim=*/2) ||
      state_weighted_value.size(/*dim=*/3) != kHeadDim ||
      tile_output.size(/*dim=*/0) != state_max.size(/*dim=*/0) ||
      tile_output.size(/*dim=*/1) != state_max.size(/*dim=*/2) ||
      tile_output.size(/*dim=*/2) != state_max.size(/*dim=*/1) ||
      tile_output.size(/*dim=*/3) != kHeadDim ||
      tile_max_stats.size(/*dim=*/0) != state_max.size(/*dim=*/0) ||
      tile_max_stats.size(/*dim=*/1) != state_max.size(/*dim=*/1) ||
      tile_max_stats.size(/*dim=*/2) != state_max.size(/*dim=*/2) ||
      tile_normalizer_stats.sizes() != tile_max_stats.sizes() ||
      tile_max_stats.size(/*dim=*/3) <= 0) {
    return false;
  }
  return fits_int32(state_max.size(/*dim=*/0)) &&
         fits_int32(state_max.size(/*dim=*/1)) &&
         fits_int32(state_max.size(/*dim=*/2)) &&
         fits_int32(tile_max_stats.size(/*dim=*/3));
}

}  // namespace

bool can_update_online_softmax_state(
    const torch::Tensor& state_max,
    const torch::Tensor& state_normalizer,
    const torch::Tensor& state_weighted_value,
    const torch::Tensor& tile_output,
    const torch::Tensor& tile_max_stats,
    const torch::Tensor& tile_normalizer_stats) {
  return has_matching_state_contract(state_max,
                                     state_normalizer,
                                     state_weighted_value,
                                     tile_output,
                                     tile_max_stats,
                                     tile_normalizer_stats) &&
         find_online_softmax_state_update_kernel_entry(
             build_runtime_specialization(tile_output)) != nullptr;
}

void update_online_softmax_state(torch::Tensor& state_max,
                                 torch::Tensor& state_normalizer,
                                 torch::Tensor& state_weighted_value,
                                 const torch::Tensor& tile_output,
                                 const torch::Tensor& tile_max_stats,
                                 const torch::Tensor& tile_normalizer_stats) {
  CHECK(can_update_online_softmax_state(state_max,
                                        state_normalizer,
                                        state_weighted_value,
                                        tile_output,
                                        tile_max_stats,
                                        tile_normalizer_stats))
      << "TileLang online-softmax state update: unsupported tensor contract";

  const OnlineSoftmaxStateUpdateSpecialization specialization =
      build_runtime_specialization(tile_output);
  const auto* entry =
      find_online_softmax_state_update_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang online-softmax state update: no compiled variant. "
      << "Available variants: "
      << available_online_softmax_state_update_variant_keys();

  const int64_t buffer_capacity = std::max({state_weighted_value.numel(),
                                            tile_output.numel(),
                                            tile_max_stats.numel(),
                                            tile_normalizer_stats.numel()});
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(state_max.device().index()).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(state_max.data_ptr()),
      reinterpret_cast<uint8_t*>(state_normalizer.data_ptr()),
      reinterpret_cast<uint8_t*>(state_weighted_value.data_ptr()),
      reinterpret_cast<uint8_t*>(const_cast<void*>(tile_max_stats.data_ptr())),
      reinterpret_cast<uint8_t*>(
          const_cast<void*>(tile_normalizer_stats.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(tile_output.data_ptr())),
      static_cast<int32_t>(state_max.size(/*dim=*/0)),
      static_cast<int32_t>(state_max.size(/*dim=*/2)),
      static_cast<int32_t>(state_max.size(/*dim=*/1)),
      static_cast<int32_t>(tile_max_stats.size(/*dim=*/3)),
      buffer_capacity,
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
