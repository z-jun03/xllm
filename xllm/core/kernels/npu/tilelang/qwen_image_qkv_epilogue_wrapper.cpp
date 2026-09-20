/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_QWEN_IMAGE_QKV_EPILOGUE_REGISTRY_INC
#error "XLLM_TL_QWEN_IMAGE_QKV_EPILOGUE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kNumHeads = 24;
constexpr int64_t kRopeHeadsPerGroup = 12;
constexpr int64_t kMaxFusedHeadRows = 512 * 1024;

struct RotaryConstants {
  torch::Tensor rotate_gather_offsets;
};

std::unordered_map<int32_t, RotaryConstants> g_rotary_constants_cache;
std::mutex g_rotary_constants_mutex;

RotaryConstants get_or_create_rotary_constants(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_rotary_constants_mutex);
  auto it = g_rotary_constants_cache.find(device_index);
  if (it != g_rotary_constants_cache.end()) {
    return it->second;
  }

  const torch::TensorOptions int_options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  torch::Tensor rotate_gather_offsets =
      torch::empty({kRopeHeadsPerGroup * kHeadDim}, int_options);
  int32_t* rotate_gather_offsets_ptr =
      rotate_gather_offsets.data_ptr<int32_t>();
  for (int64_t index = 0; index < kRopeHeadsPerGroup * kHeadDim; ++index) {
    const int64_t head_offset = (index / kHeadDim) * kHeadDim;
    const int64_t head_index = index % kHeadDim;
    const int64_t pair_start = (head_index / 2) * 2;
    rotate_gather_offsets_ptr[index] = static_cast<int32_t>(
        4 * (head_offset + pair_start + 1 - head_index % 2));
  }

  RotaryConstants constants;
  constants.rotate_gather_offsets =
      rotate_gather_offsets.to(device).view(torch::kUInt32).contiguous();
  g_rotary_constants_cache.emplace(device_index, constants);
  return constants;
}

#include XLLM_TL_QWEN_IMAGE_QKV_EPILOGUE_REGISTRY_INC

QwenImageQkvEpilogueSpecialization build_runtime_specialization(
    const torch::Tensor& query) {
  return make_qwen_image_qkv_epilogue_specialization(
      QwenImageQkvEpilogueHeadDim{static_cast<int32_t>(query.size(3))},
      QwenImageQkvEpilogueNumHeads{static_cast<int32_t>(query.size(2))},
      QwenImageQkvEpilogueDType{to_tilelang_dtype(query.scalar_type())});
}

bool have_matching_qkv_contract(const torch::Tensor& query,
                                const torch::Tensor& key,
                                const torch::Tensor& value) {
  return query.defined() && key.defined() && value.defined() &&
         query.dim() == 4 && query.sizes() == key.sizes() &&
         query.sizes() == value.sizes() && query.device() == key.device() &&
         query.device() == value.device() &&
         query.scalar_type() == key.scalar_type() &&
         query.scalar_type() == value.scalar_type() && query.stride(3) == 1 &&
         key.stride(3) == 1 && value.stride(3) == 1 &&
         query.stride(2) == query.size(3) && key.stride(2) == key.size(3) &&
         value.stride(2) == value.size(3) && query.stride(0) == key.stride(0) &&
         query.stride(0) == value.stride(0) &&
         query.stride(1) == key.stride(1) && query.stride(1) == value.stride(1);
}

bool have_matching_weight_contract(const torch::Tensor& weight,
                                   const torch::Tensor& query) {
  return weight.defined() && weight.device() == query.device() &&
         weight.scalar_type() == torch::kFloat32 && weight.dim() == 1 &&
         weight.size(0) == kHeadDim && weight.is_contiguous();
}

bool have_matching_rotary_contract(const torch::Tensor& rotary,
                                   const torch::Tensor& query,
                                   int64_t joint_tokens) {
  return rotary.defined() && rotary.device() == query.device() &&
         rotary.scalar_type() == torch::kFloat32 && rotary.dim() == 4 &&
         rotary.size(0) == 1 && rotary.size(1) >= joint_tokens &&
         rotary.size(2) == 1 && rotary.size(3) == kHeadDim &&
         rotary.is_contiguous();
}

bool have_matching_freqs_contract(const torch::Tensor& freqs,
                                  const torch::Tensor& query) {
  return freqs.defined() && freqs.device() == query.device() &&
         freqs.scalar_type() == torch::kComplexFloat && freqs.dim() == 2 &&
         freqs.size(0) >= query.size(1) && freqs.size(1) == kHeadDim / 2 &&
         freqs.is_contiguous();
}

bool fits_int32(int64_t value) {
  return value >= 0 &&
         value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

}  // namespace

bool can_qwen_image_qkv_epilogue(const torch::Tensor& img_q,
                                 const torch::Tensor& img_k,
                                 const torch::Tensor& img_v,
                                 const torch::Tensor& txt_q,
                                 const torch::Tensor& txt_k,
                                 const torch::Tensor& txt_v,
                                 const torch::Tensor& img_q_weight,
                                 const torch::Tensor& img_k_weight,
                                 const torch::Tensor& txt_q_weight,
                                 const torch::Tensor& txt_k_weight,
                                 const torch::Tensor& rotary_cos,
                                 const torch::Tensor& rotary_sin) {
  if (!have_matching_qkv_contract(img_q, img_k, img_v) ||
      !have_matching_qkv_contract(txt_q, txt_k, txt_v) ||
      img_q.device().type() != c10::DeviceType::PrivateUse1 ||
      img_q.scalar_type() != torch::kBFloat16 ||
      txt_q.device() != img_q.device() ||
      txt_q.scalar_type() != img_q.scalar_type() ||
      img_q.size(0) != txt_q.size(0) || img_q.size(2) != txt_q.size(2) ||
      img_q.size(2) != kNumHeads || img_q.size(3) != kHeadDim ||
      txt_q.size(3) != kHeadDim || img_q.size(0) <= 0 || img_q.size(1) <= 0 ||
      img_q.size(2) <= 0 || txt_q.size(1) <= 0) {
    return false;
  }

  const int64_t joint_tokens = img_q.size(1) + txt_q.size(1);
  if (joint_tokens > kMaxFusedHeadRows / img_q.size(0) / img_q.size(2)) {
    return false;
  }

  const std::array<torch::Tensor, 4> weights = {
      img_q_weight, img_k_weight, txt_q_weight, txt_k_weight};
  for (const torch::Tensor& weight : weights) {
    if (!have_matching_weight_contract(weight, img_q)) {
      return false;
    }
  }
  const bool have_preexpanded_rotary =
      have_matching_rotary_contract(rotary_cos, img_q, joint_tokens) &&
      have_matching_rotary_contract(rotary_sin, img_q, joint_tokens);
  const bool have_legacy_freqs =
      have_matching_freqs_contract(rotary_cos, img_q) &&
      have_matching_freqs_contract(rotary_sin, txt_q);
  if (!have_preexpanded_rotary && !have_legacy_freqs) {
    return false;
  }

  const std::array<int64_t, 8> runtime_values = {img_q.size(0),
                                                 img_q.size(1),
                                                 txt_q.size(1),
                                                 img_q.size(2),
                                                 img_q.stride(0),
                                                 img_q.stride(1),
                                                 txt_q.stride(0),
                                                 txt_q.stride(1)};
  for (int64_t value : runtime_values) {
    if (!fits_int32(value)) {
      return false;
    }
  }

  return find_qwen_image_qkv_epilogue_kernel_entry(
             build_runtime_specialization(img_q)) != nullptr;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qwen_image_qkv_epilogue(
    const torch::Tensor& img_q,
    const torch::Tensor& img_k,
    const torch::Tensor& img_v,
    const torch::Tensor& txt_q,
    const torch::Tensor& txt_k,
    const torch::Tensor& txt_v,
    const torch::Tensor& img_q_weight,
    const torch::Tensor& img_k_weight,
    const torch::Tensor& txt_q_weight,
    const torch::Tensor& txt_k_weight,
    const torch::Tensor& rotary_cos,
    const torch::Tensor& rotary_sin,
    double img_eps,
    double txt_eps) {
  CHECK(can_qwen_image_qkv_epilogue(img_q,
                                    img_k,
                                    img_v,
                                    txt_q,
                                    txt_k,
                                    txt_v,
                                    img_q_weight,
                                    img_k_weight,
                                    txt_q_weight,
                                    txt_k_weight,
                                    rotary_cos,
                                    rotary_sin))
      << "TileLang Qwen-Image QKV epilogue: unsupported tensor contract";

  torch::Tensor normalized_img_q = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(img_q, img_q_weight, img_eps));
  torch::Tensor normalized_img_k = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(img_k, img_k_weight, img_eps));
  torch::Tensor normalized_txt_q = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(txt_q, txt_q_weight, txt_eps));
  torch::Tensor normalized_txt_k = std::get<0>(
      at_npu::native::custom_ops::npu_rms_norm(txt_k, txt_k_weight, txt_eps));

  torch::Tensor kernel_rotary_cos = rotary_cos;
  torch::Tensor kernel_rotary_sin = rotary_sin;
  if (rotary_cos.scalar_type() == torch::kComplexFloat) {
    torch::Tensor joint_freqs = torch::cat({rotary_sin, rotary_cos}, 0);
    const int64_t sequence_length = joint_freqs.size(0);
    kernel_rotary_cos = torch::real(joint_freqs)
                            .unsqueeze(0)
                            .unsqueeze(2)
                            .unsqueeze(-1)
                            .expand({-1, -1, -1, -1, 2})
                            .reshape({1, sequence_length, 1, kHeadDim});
    torch::Tensor sin = torch::imag(joint_freqs);
    kernel_rotary_sin = torch::stack({-sin, sin}, -1)
                            .reshape({1, sequence_length, 1, kHeadDim});
  }

  const int64_t joint_tokens = txt_q.size(1) + img_q.size(1);
  const std::vector<int64_t> output_shape = {
      img_q.size(0), joint_tokens, img_q.size(2), kHeadDim};
  torch::Tensor joint_q = torch::empty(output_shape, img_q.options());
  torch::Tensor joint_k = torch::empty_like(joint_q);
  torch::Tensor joint_v = torch::empty_like(joint_q);
  const RotaryConstants rotary_constants =
      get_or_create_rotary_constants(img_q.device());

  const QwenImageQkvEpilogueSpecialization specialization =
      build_runtime_specialization(img_q);
  const auto* entry = find_qwen_image_qkv_epilogue_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang Qwen-Image QKV epilogue: no compiled variant. "
      << "Available variants: "
      << available_qwen_image_qkv_epilogue_variant_keys();

  const auto required_input_capacity = [](const torch::Tensor& tensor) {
    return (tensor.size(0) - 1) * tensor.stride(0) +
           (tensor.size(1) - 1) * tensor.stride(1) +
           tensor.size(2) * tensor.size(3);
  };
  const int64_t buffer_capacity =
      std::max({joint_q.numel(),
                required_input_capacity(normalized_img_q),
                required_input_capacity(img_v),
                required_input_capacity(normalized_txt_q),
                required_input_capacity(txt_v),
                kernel_rotary_cos.numel(),
                kernel_rotary_sin.numel(),
                kHeadDim});
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(img_q.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(
                const_cast<void*>(normalized_img_q.data_ptr())),
            reinterpret_cast<uint8_t*>(
                const_cast<void*>(normalized_img_k.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(img_v.data_ptr())),
            reinterpret_cast<uint8_t*>(
                const_cast<void*>(normalized_txt_q.data_ptr())),
            reinterpret_cast<uint8_t*>(
                const_cast<void*>(normalized_txt_k.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(txt_v.data_ptr())),
            reinterpret_cast<uint8_t*>(
                const_cast<void*>(kernel_rotary_cos.data_ptr())),
            reinterpret_cast<uint8_t*>(
                const_cast<void*>(kernel_rotary_sin.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(
                rotary_constants.rotate_gather_offsets.data_ptr())),
            reinterpret_cast<uint8_t*>(joint_q.data_ptr()),
            reinterpret_cast<uint8_t*>(joint_k.data_ptr()),
            reinterpret_cast<uint8_t*>(joint_v.data_ptr()),
            static_cast<int32_t>(img_q.size(0)),
            static_cast<int32_t>(img_q.size(1)),
            static_cast<int32_t>(txt_q.size(1)),
            static_cast<int32_t>(img_q.size(2)),
            static_cast<int32_t>(normalized_img_q.stride(0)),
            static_cast<int32_t>(normalized_img_q.stride(1)),
            static_cast<int32_t>(img_v.stride(0)),
            static_cast<int32_t>(img_v.stride(1)),
            static_cast<int32_t>(normalized_txt_q.stride(0)),
            static_cast<int32_t>(normalized_txt_q.stride(1)),
            static_cast<int32_t>(txt_v.stride(0)),
            static_cast<int32_t>(txt_v.stride(1)),
            buffer_capacity,
            stream);

  return std::make_tuple(joint_q, joint_k, joint_v);
}

}  // namespace xllm::kernel::npu::tilelang
