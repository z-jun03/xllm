/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class DiTConfig final {
 public:
  DiTConfig() = default;
  ~DiTConfig() = default;

  static DiTConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "DiT MODEL OPTIONS",
        {"max_requests_per_batch",
         "dit_cache_policy",
         "dit_cache_warmup_steps",
         "dit_cache_n_derivatives",
         "dit_cache_skip_interval_steps",
         "dit_cache_residual_diff_threshold",
         "dit_cache_start_steps",
         "dit_cache_end_steps",
         "dit_cache_start_blocks",
         "dit_cache_end_blocks",
         "dit_qwen_image_cfg_modulation_cache",
         "dit_sp_communication_overlap",
         "dit_sp_packed_qkv_all_to_all",
         "dit_sp_packed_qkv_triton_pack",
         "dit_sp_packed_qkv_input_overlap",
         "dit_sp_packed_qkv_comm_stream_overlap",
         "dit_sp_fused_qkv_projection",
         "dit_sp_fused_qkv_postprocess",
         "dit_sp_ring_kv_attention",
         "dit_sp_ring_kv_packed_transfer",
         "dit_sp_ring_kv_comm_stream_overlap",
         "dit_sp_ring_kv_native_attention_update",
         "dit_sp_ring_kv_tilelang_online_update",
         "dit_sp_ring_kv_sequence_chunks",
         "dit_sp_packed_qkv_attention_overlap",
         "dit_sp_packed_qkv_attention_overlap_tiles",
         "dit_sp_profile",
         "dit_debug_print",
         "dit_laser_attention_enabled",
         "dit_generation_image_area_max",
         "dit_vae_image_size",
         "dit_enable_vae_tiling",
         "dit_sparse_attention_enabled",
         "dit_sparse_attention_sparsity",
         "dit_sparse_attention_pool_size",
         "dit_sparse_attention_sparse_start_step",
         "dit_sparse_attention_version",
         "dit_sparse_attention_mask_refresh_steps",
         "max_sequence_length"}};
    return kOptionCategory;
  }

  PROPERTY(int32_t, max_requests_per_batch) = 1;

  PROPERTY(std::string, dit_cache_policy) = "TaylorSeer";

  PROPERTY(int64_t, dit_cache_warmup_steps) = 0;

  PROPERTY(int64_t, dit_cache_n_derivatives) = 3;

  PROPERTY(int64_t, dit_cache_skip_interval_steps) = 3;

  PROPERTY(double, dit_cache_residual_diff_threshold) = 0.09;

  PROPERTY(int64_t, dit_cache_start_steps) = 5;

  PROPERTY(int64_t, dit_cache_end_steps) = 5;

  PROPERTY(int64_t, dit_cache_start_blocks) = 5;

  PROPERTY(int64_t, dit_cache_end_blocks) = 5;

  PROPERTY(bool, dit_qwen_image_cfg_modulation_cache) = false;

  PROPERTY(bool, dit_sp_communication_overlap) = true;

  PROPERTY(bool, dit_sp_packed_qkv_all_to_all) = false;

  PROPERTY(bool, dit_sp_packed_qkv_triton_pack) = false;

  PROPERTY(bool, dit_sp_packed_qkv_input_overlap) = false;

  PROPERTY(bool, dit_sp_packed_qkv_comm_stream_overlap) = false;

  PROPERTY(bool, dit_sp_fused_qkv_projection) = false;

  PROPERTY(bool, dit_sp_fused_qkv_postprocess) = false;

  PROPERTY(bool, dit_sp_ring_kv_attention) = false;

  PROPERTY(bool, dit_sp_ring_kv_packed_transfer) = false;

  PROPERTY(bool, dit_sp_ring_kv_comm_stream_overlap) = false;

  PROPERTY(bool, dit_sp_ring_kv_native_attention_update) = false;

  PROPERTY(bool, dit_sp_ring_kv_tilelang_online_update) = false;

  PROPERTY(int32_t, dit_sp_ring_kv_sequence_chunks) = 1;

  PROPERTY(bool, dit_sp_packed_qkv_attention_overlap) = false;

  PROPERTY(int32_t, dit_sp_packed_qkv_attention_overlap_tiles) = 2;

  PROPERTY(bool, dit_sp_profile) = false;

  PROPERTY(bool, dit_debug_print) = false;

  PROPERTY(bool, dit_laser_attention_enabled) = false;

  PROPERTY(int64_t, dit_generation_image_area_max) = 0;

  PROPERTY(int64_t, dit_vae_image_size) = 1048576;

  PROPERTY(bool, dit_enable_vae_tiling) = false;

  PROPERTY(bool, dit_sparse_attention_enabled) = false;

  PROPERTY(double, dit_sparse_attention_sparsity) = 0.5;

  PROPERTY(int64_t, dit_sparse_attention_pool_size) = 128;

  PROPERTY(int64_t, dit_sparse_attention_sparse_start_step) = 0;

  PROPERTY(std::string, dit_sparse_attention_version) = "rain_fusion";

  PROPERTY(int64_t, dit_sparse_attention_mask_refresh_steps) = 1;

  PROPERTY(int32_t, max_sequence_length) = 0;
};

}  // namespace xllm
