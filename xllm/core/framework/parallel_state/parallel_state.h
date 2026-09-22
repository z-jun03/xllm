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

#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "parallel_args.h"
#include "process_group.h"

namespace xllm {

// Forward declaration
namespace runtime {
struct Options;
}

namespace parallel_state {

struct GatherAsyncCtx {
  torch::Tensor input;
  torch::Tensor stacked;
  c10::intrusive_ptr<c10d::Work> work;
  std::vector<int32_t> token_num_list;
};

struct ReduceAsyncCtx {
  torch::Tensor tensor;
  c10::intrusive_ptr<c10d::Work> work;
};

// FP32 running state for attention over a sequence of KV tiles. The state is
// laid out as [B, H, S_q] for scores/normalizers and [B, H, S_q, D] for the
// unnormalized weighted values.
struct OnlineSoftmaxAttentionState {
  torch::Tensor max_scores;
  torch::Tensor normalizers;
  torch::Tensor weighted_values;
};

// FP32 running state for NPU fusion-attention tiles. The state is laid out as
// [B, H, S_q] for scores/normalizers and [B, H, S_q, D] for the unnormalized
// weighted values.
struct NpuFusionAttentionOnlineState {
  torch::Tensor max_scores;
  torch::Tensor normalizers;
  torch::Tensor weighted_values;
};

using PackedQkvPostprocessor =
    std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
        const torch::Tensor& packed_output)>;

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args);

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     int32_t dim = -1);

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     const std::vector<int32_t>& token_num_list);

GatherAsyncCtx launch_gather(const torch::Tensor& input,
                             ProcessGroup* process_group,
                             const std::vector<int32_t>& token_num_list);

torch::Tensor finish_gather(GatherAsyncCtx ctx);

ReduceAsyncCtx launch_reduce(torch::Tensor input, ProcessGroup* process_group);

torch::Tensor finish_reduce(ReduceAsyncCtx ctx);

torch::Tensor all_gather_interleaved(const torch::Tensor& input,
                                     ProcessGroup* process_group);

torch::Tensor reduce(torch::Tensor& input, ProcessGroup* process_group);

torch::Tensor reduce_scatter(const torch::Tensor& input,
                             ProcessGroup* process_group);

// Global ranks in this rank's CP group, ordered by CP rank.
std::vector<int32_t> compute_cp_group_ranks(int32_t global_rank,
                                            int32_t world_size,
                                            int32_t dp_size,
                                            int32_t cp_size);

OnlineSoftmaxAttentionState initialize_online_softmax_attention(
    const torch::Tensor& query);

OnlineSoftmaxAttentionState update_online_softmax_attention(
    OnlineSoftmaxAttentionState state,
    const torch::Tensor& query,
    const torch::Tensor& key_tile,
    const torch::Tensor& value_tile,
    double scale);

torch::Tensor finalize_online_softmax_attention(
    const OnlineSoftmaxAttentionState& state,
    torch::ScalarType output_dtype);

NpuFusionAttentionOnlineState initialize_npu_fusion_attention_online(
    const torch::Tensor& query);

NpuFusionAttentionOnlineState update_npu_fusion_attention_online(
    NpuFusionAttentionOnlineState state,
    const torch::Tensor& query,
    const torch::Tensor& key_tile,
    const torch::Tensor& value_tile,
    int64_t num_heads,
    double scale,
    const torch::Tensor& attn_mask = torch::Tensor());

torch::Tensor finalize_npu_fusion_attention_online(
    const NpuFusionAttentionOnlineState& state,
    torch::ScalarType output_dtype);

// Ring-KV attention using NPU fusion-attention tiles and FP32 online-softmax
// state. Every rank keeps its local sequence queries and circulates K/V tiles.
torch::Tensor ring_kv_npu_fusion_attention(
    const torch::Tensor& query,
    const torch::Tensor& local_key,
    const torch::Tensor& local_value,
    int64_t num_heads,
    double scale,
    ProcessGroup* process_group,
    const std::vector<torch::Tensor>& attention_masks = {});

// Synchronous correctness reference for Ring-KV attention. Every rank keeps
// its query shard and circulates KV shards through the process group.
torch::Tensor ring_kv_online_softmax_attention_reference(
    const torch::Tensor& query,
    const torch::Tensor& local_key,
    const torch::Tensor& local_value,
    double scale,
    ProcessGroup* process_group);

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim = -1);

std::function<torch::Tensor()> all_to_all_4D(const torch::Tensor& input,
                                             int32_t scatter_idx,
                                             int32_t gather_idx,
                                             bool async_ops,
                                             ProcessGroup* process_group);

std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>()>
all_to_all_4D_packed_qkv(const torch::Tensor& query,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         bool async_ops,
                         ProcessGroup* process_group,
                         const std::string& buffer_role);

std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>()>
all_to_all_4D_packed_fused_qkv(const torch::Tensor& qkv,
                               bool async_ops,
                               ProcessGroup* process_group,
                               const std::string& buffer_role,
                               PackedQkvPostprocessor postprocessor = {});

std::vector<
    std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>()>>
all_to_all_4D_packed_fused_qkv_chunked(const torch::Tensor& qkv,
                                       int32_t tile_count,
                                       ProcessGroup* process_group,
                                       const std::string& buffer_role);

// Create a process group where each process has a single device
// devices: list of devices to create process groups on.
std::vector<std::unique_ptr<ProcessGroup>> create_npu_process_groups(
    const std::vector<torch::Device>& devices);

// Create process groups for local (single-node) scenarios
// Supports GPU (CUDA/MLU) and NPU, including single-device case
// Parse port from options.master_node_addr() to support multiple instances
std::vector<std::unique_ptr<ProcessGroup>> create_local_process_groups(
    const std::vector<torch::Device>& devices,
    const runtime::Options& options);

}  // namespace parallel_state
}  // namespace xllm
