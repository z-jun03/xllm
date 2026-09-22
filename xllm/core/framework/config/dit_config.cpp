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

#include "core/framework/config/dit_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_int32(max_requests_per_batch, 1, "Max number of request per batch.");

DEFINE_string(dit_cache_policy,
              "TaylorSeer",
              "The policy of dit cache(e.g. None, FBCache, TaylorSeer, "
              "FBCacheTaylorSeer, ResidualCache).");

DEFINE_int64(dit_cache_warmup_steps, 0, "The number of warmup steps.");

DEFINE_int64(dit_cache_n_derivatives,
             3,
             "The number of derivatives to use in TaylorSeer.");

DEFINE_int64(dit_cache_skip_interval_steps,
             3,
             "The interval steps to skip for derivative calculation.");

DEFINE_double(dit_cache_residual_diff_threshold,
              0.09,
              "The residual difference threshold for cache reuse.");

DEFINE_int64(dit_cache_start_steps,
             5,
             "The number of steps to skip at the start");

DEFINE_int64(dit_cache_end_steps, 5, "The number of steps to skip at the end.");

DEFINE_int64(dit_cache_start_blocks,
             5,
             "The number of blocks to skip at the start.");

DEFINE_int64(dit_cache_end_blocks,
             5,
             "The number of blocks to skip at the end.");

DEFINE_bool(dit_qwen_image_cfg_modulation_cache,
            false,
            "Reuse Qwen-Image modulation projections across serial true-CFG "
            "branches.");

DEFINE_bool(dit_sp_communication_overlap,
            true,
            "Communication & Computation overlap for sequence parallel");

DEFINE_bool(dit_sp_packed_qkv_all_to_all,
            false,
            "Pack sequence-parallel Q/K/V all-to-all into one collective.");

DEFINE_bool(dit_sp_packed_qkv_triton_pack,
            false,
            "Use the experimental Triton Q/K/V pack kernel for packed "
            "sequence-parallel all-to-all.");

DEFINE_bool(dit_sp_packed_qkv_input_overlap,
            false,
            "Overlap image Q/K normalization with text packed Q/K/V "
            "sequence-parallel all-to-all.");

DEFINE_bool(dit_sp_packed_qkv_comm_stream_overlap,
            false,
            "Launch packed Q/K/V sequence-parallel all-to-all on a dedicated "
            "NPU stream with event-based compute dependencies.");

DEFINE_bool(dit_sp_fused_qkv_projection,
            false,
            "Fuse JoyImageEdit Q/K/V projections before packed "
            "sequence-parallel all-to-all.");

DEFINE_bool(dit_sp_fused_qkv_postprocess,
            false,
            "Fuse JoyImageEdit packed Q/K/V unpack, image Q/K RMSNorm, and "
            "RoPE after sequence-parallel all-to-all.");

DEFINE_bool(dit_sp_ring_kv_attention,
            false,
            "Use the experimental JoyImageEdit Ring-KV sequence-parallel "
            "attention backend.");

DEFINE_bool(dit_sp_ring_kv_packed_transfer,
            false,
            "Pack Ring-KV K/V into one P2P transfer. This is experimental "
            "and disabled by default.");

DEFINE_bool(dit_sp_ring_kv_comm_stream_overlap,
            false,
            "Launch the next Ring-KV P2P transfer on a dedicated NPU stream "
            "while the current KV tile attention runs on the compute stream.");

DEFINE_bool(
    dit_sp_ring_kv_native_attention_update,
    false,
    "Use CANN npu_attention_update to merge Ring-KV attention tiles. "
    "This experimental path can differ by BF16 rounding and is disabled "
    "by default.");

DEFINE_bool(
    dit_sp_ring_kv_tilelang_online_update,
    false,
    "Use the experimental TileLang FP32 online-softmax state update for "
    "Ring-KV attention tiles. This only supports BF16 head_dim=128.");

DEFINE_int32(dit_sp_ring_kv_sequence_chunks,
             1,
             "Split each SP=2 Ring-KV shard into sequence chunks and pipeline "
             "the next P2P transfer with remote-chunk attention. This "
             "experimental path can optionally pack each K/V chunk.");

DEFINE_bool(dit_sp_packed_qkv_attention_overlap,
            false,
            "Overlap packed Q/K/V sequence-parallel all-to-all with "
            "JoyImageEdit attention over destination-head tiles.");

DEFINE_int32(dit_sp_packed_qkv_attention_overlap_tiles,
             2,
             "Number of destination-head tiles for packed Q/K/V attention "
             "overlap.");

DEFINE_bool(dit_sp_profile,
            false,
            "Log sequence-parallel attention stage timings for diagnostics.");

DEFINE_bool(dit_debug_print,
            false,
            "whether print the debug info for dit models");

DEFINE_bool(dit_laser_attention_enabled,
            false,
            "whether to use the laser attention kernel (MindIE-SD, tuned for "
            "Wan2.2) in place of npu_fusion_attention for DiT attention.");

DEFINE_int64(dit_generation_image_area_max,
             0,
             "Maximum allowed image area (width * height) for image generation "
             "requests. If set to 0, there is no limit.");
// --- dit vae tiling ---

DEFINE_bool(
    dit_enable_vae_tiling,
    false,
    "whether enable vae tiling, currently only support qwen-image-edit-plus");

DEFINE_int64(
    dit_vae_image_size,
    1048576,
    "Qwen Image Edit Plus VAE image size used to calculate dimensions.");

DEFINE_bool(dit_sparse_attention_enabled,
            false,
            "Enable block-wise sparse attention / RainFusion for WAN.");

DEFINE_double(dit_sparse_attention_sparsity,
              0.5,
              "Sparse attention sparsity ratio in [0.0, 1.0). 0.0 = dense "
              "attention, 0.5 = drop 50 percent blocks.");

DEFINE_int64(
    dit_sparse_attention_pool_size,
    128,
    "Sparse attention pooling window size for block-wise mask generation.");

DEFINE_int64(dit_sparse_attention_sparse_start_step,
             0,
             "Sparse attention step index to start sparse attention. "
             "Steps before this use dense attention.");

DEFINE_string(
    dit_sparse_attention_version,
    "rain_fusion",
    "Sparse attention version: 'rain_fusion' (frame-pairing + "
    "aclnnRainFusionAttention) or 'sparse_attention' (block-decompose + "
    "aclnnBlockSparseAttention).");

DEFINE_int64(dit_sparse_attention_mask_refresh_steps,
             1,
             "Sparse attention: recompute block sparse mask every N diffusion "
             "steps. 1 = every step (default), higher = reuse mask longer.");

DEFINE_int32(
    max_sequence_length,
    0,
    "Max sequence length for Flux2 text encoder tokenizer. 0 means disabled.");

namespace xllm {

void DiTConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_requests_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_policy);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_warmup_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_n_derivatives);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_skip_interval_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_residual_diff_threshold);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_start_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_end_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_start_blocks);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_end_blocks);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_qwen_image_cfg_modulation_cache);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_communication_overlap);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_packed_qkv_all_to_all);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_packed_qkv_triton_pack);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_packed_qkv_input_overlap);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_packed_qkv_comm_stream_overlap);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_fused_qkv_projection);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_fused_qkv_postprocess);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_ring_kv_attention);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_ring_kv_packed_transfer);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_ring_kv_comm_stream_overlap);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_ring_kv_native_attention_update);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_ring_kv_tilelang_online_update);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_ring_kv_sequence_chunks);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_packed_qkv_attention_overlap);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_packed_qkv_attention_overlap_tiles);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_profile);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_debug_print);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_laser_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_generation_image_area_max);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_vae_image_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_enable_vae_tiling);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_sparsity);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_pool_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_sparse_start_step);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_version);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_mask_refresh_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_sequence_length);
}

void DiTConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_requests_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_policy);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_warmup_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_n_derivatives);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_skip_interval_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_residual_diff_threshold);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_start_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_end_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_start_blocks);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_end_blocks);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_qwen_image_cfg_modulation_cache);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_communication_overlap);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_packed_qkv_all_to_all);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_packed_qkv_triton_pack);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_packed_qkv_input_overlap);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_packed_qkv_comm_stream_overlap);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_fused_qkv_projection);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_fused_qkv_postprocess);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_ring_kv_attention);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_ring_kv_packed_transfer);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_ring_kv_comm_stream_overlap);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_ring_kv_native_attention_update);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_ring_kv_tilelang_online_update);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_ring_kv_sequence_chunks);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_packed_qkv_attention_overlap);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_packed_qkv_attention_overlap_tiles);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_profile);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_debug_print);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_laser_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_generation_image_area_max);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_vae_image_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_enable_vae_tiling);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_sparsity);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_pool_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_sparse_start_step);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_version);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_mask_refresh_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_sequence_length);
}

void DiTConfig::append_config_json(nlohmann::ordered_json& config_json) const {
  const DiTConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_requests_per_batch);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_policy);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_warmup_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_n_derivatives);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_skip_interval_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_residual_diff_threshold);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_start_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_end_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_start_blocks);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_end_blocks);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_qwen_image_cfg_modulation_cache);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_communication_overlap);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_packed_qkv_all_to_all);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_packed_qkv_triton_pack);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_packed_qkv_input_overlap);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_packed_qkv_comm_stream_overlap);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_fused_qkv_projection);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_fused_qkv_postprocess);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_ring_kv_attention);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_ring_kv_packed_transfer);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_ring_kv_comm_stream_overlap);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_packed_qkv_attention_overlap);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_packed_qkv_attention_overlap_tiles);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_profile);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_debug_print);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_laser_attention_enabled);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_generation_image_area_max);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_vae_image_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_enable_vae_tiling);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_enabled);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_sparsity);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_pool_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_sparse_start_step);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_version);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_mask_refresh_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_sequence_length);
}

DiTConfig& DiTConfig::get_instance() {
  static DiTConfig config;
  return config;
}

void DiTConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
