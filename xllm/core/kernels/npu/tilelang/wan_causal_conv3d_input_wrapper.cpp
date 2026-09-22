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
#include <utility>
#include <vector>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_WAN_CAUSAL_CONV3D_INPUT_REGISTRY_INC
#error "XLLM_TL_WAN_CAUSAL_CONV3D_INPUT_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kTemporalPadding = 2;
constexpr int64_t kNextCacheTemporal = 2;
constexpr int64_t kMinimumPlaneElements = 16;

#include XLLM_TL_WAN_CAUSAL_CONV3D_INPUT_REGISTRY_INC

WanCausalConv3dInputSpecialization build_runtime_specialization(
    const torch::Tensor& hidden_states,
    const torch::Tensor& feature_cache) {
  return make_wan_causal_conv3d_input_specialization(
      WanCausalConv3dInputDType{to_tilelang_dtype(hidden_states.scalar_type())},
      WanCausalConv3dInputHiddenTemporal{
          static_cast<int32_t>(hidden_states.size(2))},
      WanCausalConv3dInputCacheTemporal{
          static_cast<int32_t>(feature_cache.size(2))});
}

bool have_matching_input_contract(const torch::Tensor& hidden_states,
                                  const torch::Tensor& feature_cache) {
  return hidden_states.defined() && feature_cache.defined() &&
         hidden_states.dim() == 5 && feature_cache.dim() == 5 &&
         hidden_states.device() == feature_cache.device() &&
         hidden_states.scalar_type() == feature_cache.scalar_type() &&
         hidden_states.size(0) == feature_cache.size(0) &&
         hidden_states.size(1) == feature_cache.size(1) &&
         hidden_states.size(3) == feature_cache.size(3) &&
         hidden_states.size(4) == feature_cache.size(4) &&
         hidden_states.is_contiguous() && feature_cache.is_contiguous();
}

}  // namespace

bool can_wan_causal_conv3d_input(const torch::Tensor& hidden_states,
                                 const torch::Tensor& feature_cache) {
  if (!have_matching_input_contract(hidden_states, feature_cache) ||
      hidden_states.device().type() != c10::DeviceType::PrivateUse1 ||
      hidden_states.scalar_type() != torch::kBFloat16 ||
      hidden_states.size(0) <= 0 || hidden_states.size(1) <= 0 ||
      hidden_states.size(2) != 4 || hidden_states.size(3) <= 0 ||
      hidden_states.size(4) <= 0 || feature_cache.size(2) <= 0 ||
      feature_cache.size(2) > kTemporalPadding) {
    return false;
  }

  const int64_t plane_elements = hidden_states.size(3) * hidden_states.size(4);
  if (plane_elements < kMinimumPlaneElements ||
      plane_elements > std::numeric_limits<int32_t>::max() ||
      hidden_states.size(0) > std::numeric_limits<int32_t>::max() ||
      hidden_states.size(1) > std::numeric_limits<int32_t>::max()) {
    return false;
  }

  return find_wan_causal_conv3d_input_kernel_entry(build_runtime_specialization(
             hidden_states, feature_cache)) != nullptr;
}

std::pair<torch::Tensor, torch::Tensor> wan_causal_conv3d_input(
    const torch::Tensor& hidden_states,
    const torch::Tensor& feature_cache) {
  CHECK(can_wan_causal_conv3d_input(hidden_states, feature_cache))
      << "TileLang wan_causal_conv3d_input: unsupported tensor contract; "
      << "hidden_states=" << hidden_states.sizes()
      << ", feature_cache=" << feature_cache.sizes();

  const int64_t output_temporal = kTemporalPadding + hidden_states.size(2);
  torch::Tensor conv_input = torch::empty({hidden_states.size(0),
                                           hidden_states.size(1),
                                           output_temporal,
                                           hidden_states.size(3),
                                           hidden_states.size(4)},
                                          hidden_states.options());
  torch::Tensor next_cache = torch::empty({hidden_states.size(0),
                                           hidden_states.size(1),
                                           kNextCacheTemporal,
                                           hidden_states.size(3),
                                           hidden_states.size(4)},
                                          hidden_states.options());

  const WanCausalConv3dInputSpecialization specialization =
      build_runtime_specialization(hidden_states, feature_cache);
  const auto* entry = find_wan_causal_conv3d_input_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang wan_causal_conv3d_input: no compiled variant. "
      << "Available variants: "
      << available_wan_causal_conv3d_input_variant_keys();

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(hidden_states.device().index()).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(hidden_states.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(feature_cache.data_ptr())),
      reinterpret_cast<uint8_t*>(conv_input.data_ptr()),
      reinterpret_cast<uint8_t*>(next_cache.data_ptr()),
      static_cast<int32_t>(hidden_states.size(0)),
      static_cast<int32_t>(hidden_states.size(1)),
      static_cast<int32_t>(hidden_states.size(3) * hidden_states.size(4)),
      hidden_states.numel(),
      feature_cache.numel(),
      conv_input.numel(),
      next_cache.numel(),
      stream);
  return {conv_input, next_cache};
}

}  // namespace xllm::kernel::npu::tilelang
