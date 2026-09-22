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

#include "parallel_state.h"

#include <limits>
#include <mutex>
#include <unordered_map>

#include "core/framework/config/dit_config.h"
#include "core/util/utils.h"
#include "runtime/options.h"
#include "util/net.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "hccl/hccl.h"
#include "npu_process_group.h"
#include "third_party/torch_npu_ops/triton_npu/torch_api/triton_ops_api.h"
#endif

namespace xllm {
namespace parallel_state {

namespace {

torch::Tensor assemble_gathered(const torch::Tensor& stacked_tensor,
                                const std::vector<int32_t>& token_num_list) {
  CHECK(stacked_tensor.defined()) << "stacked_tensor must be defined.";
  CHECK_GT(stacked_tensor.dim(), 1)
      << "stacked_tensor must be stacked by rank.";
  CHECK_EQ(stacked_tensor.size(0), static_cast<int64_t>(token_num_list.size()))
      << "stacked_tensor size " << stacked_tensor.size(0)
      << " does not match token_num_list size " << token_num_list.size();

  int64_t total_tokens = xllm::util::sum(token_num_list);
  auto out_shape = stacked_tensor[0].sizes().vec();
  out_shape[0] = total_tokens;
  torch::Tensor output = torch::empty(out_shape, stacked_tensor.options());

  int64_t offset = 0;
  for (size_t i = 0; i < token_num_list.size(); ++i) {
    const int32_t valid_tokens = token_num_list[i];
    if (valid_tokens <= 0) {
      continue;
    }
    CHECK_GE(stacked_tensor[static_cast<int64_t>(i)].size(0), valid_tokens)
        << "sequence-parallel gather received fewer rows than expected: "
        << "src_rank=" << i
        << ", gathered_rows=" << stacked_tensor[static_cast<int64_t>(i)].size(0)
        << ", expected_rows=" << valid_tokens;
    output.slice(0, offset, offset + valid_tokens)
        .copy_(
            stacked_tensor[static_cast<int64_t>(i)].slice(0, 0, valid_tokens));
    offset += valid_tokens;
  }
  return output;
}

void check_online_softmax_attention_tensor(const torch::Tensor& tensor,
                                           const char* tensor_name) {
  CHECK(tensor.defined()) << tensor_name << " must be defined.";
  CHECK_EQ(tensor.dim(), 4) << tensor_name << " must have shape [B, S, H, D].";
}

torch::Tensor get_a2a_staging_buffer(const torch::Tensor& input,
                                     const std::vector<int64_t>& shape,
                                     const std::string& buffer_role) {
  if (input.requires_grad()) {
    return torch::empty(shape, input.options());
  }

  static std::mutex staging_buffers_mutex;
  static std::unordered_map<std::string, torch::Tensor> staging_buffers;
  std::string buffer_key =
      buffer_role + ":" + input.device().str() + ":" +
      std::to_string(static_cast<int32_t>(input.scalar_type()));
  int64_t required_numel = 1;
  for (int64_t dim : shape) {
    CHECK_GE(dim, 0) << "staging buffer dimensions must be non-negative.";
    required_numel *= dim;
  }

  std::lock_guard<std::mutex> lock(staging_buffers_mutex);
  auto iterator = staging_buffers.find(buffer_key);
  if (iterator == staging_buffers.end() ||
      iterator->second.numel() < required_numel) {
    iterator =
        staging_buffers
            .insert_or_assign(buffer_key,
                              torch::empty({required_numel}, input.options()))
            .first;
  }
  return iterator->second.narrow(/*dim=*/0, /*start=*/0, required_numel)
      .view(shape);
}

#if defined(USE_NPU)
c10_npu::NPUStream& get_dit_comm_stream(int32_t device_index) {
  static std::mutex comm_streams_mutex;
  static std::unordered_map<int32_t, std::unique_ptr<c10_npu::NPUStream>>
      comm_streams;
  std::lock_guard<std::mutex> lock(comm_streams_mutex);
  auto iterator = comm_streams.find(device_index);
  if (iterator == comm_streams.end()) {
    iterator = comm_streams
                   .emplace(device_index,
                            std::make_unique<c10_npu::NPUStream>(
                                c10_npu::getNPUStreamFromPool(device_index)))
                   .first;
  }
  return *iterator->second;
}
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_packed_qkv(
    const torch::Tensor& packed_output,
    int32_t group_size,
    int64_t batch_size,
    int64_t shard_sequence_length,
    int64_t local_head_num,
    int64_t head_size) {
  torch::Tensor packed;
  if (batch_size == 1) {
    packed = packed_output.reshape(
        {1,
         static_cast<int64_t>(group_size) * shard_sequence_length,
         local_head_num,
         3 * head_size});
  } else {
    packed =
        packed_output
            .reshape({static_cast<int64_t>(group_size) * shard_sequence_length,
                      batch_size,
                      local_head_num,
                      3 * head_size})
            .transpose(0, 1)
            .contiguous()
            .reshape({batch_size,
                      static_cast<int64_t>(group_size) * shard_sequence_length,
                      local_head_num,
                      3 * head_size});
  }
  std::vector<torch::Tensor> qkv = packed.split(head_size, /*dim=*/3);
  return std::make_tuple(
      qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous());
}

}  // namespace

OnlineSoftmaxAttentionState initialize_online_softmax_attention(
    const torch::Tensor& query) {
  check_online_softmax_attention_tensor(query, "query");
  const torch::Tensor query_by_head = query.transpose(/*dim0=*/1, /*dim1=*/2);
  const torch::TensorOptions state_options =
      query.options().dtype(torch::kFloat32);
  const std::vector<int64_t> score_shape = {
      query_by_head.size(0), query_by_head.size(1), query_by_head.size(2)};
  OnlineSoftmaxAttentionState state;
  state.max_scores = torch::full(
      score_shape, -std::numeric_limits<float>::infinity(), state_options);
  state.normalizers = torch::zeros(score_shape, state_options);
  state.weighted_values = torch::zeros(query_by_head.sizes(), state_options);
  return state;
}

OnlineSoftmaxAttentionState update_online_softmax_attention(
    OnlineSoftmaxAttentionState state,
    const torch::Tensor& query,
    const torch::Tensor& key_tile,
    const torch::Tensor& value_tile,
    double scale) {
  check_online_softmax_attention_tensor(query, "query");
  check_online_softmax_attention_tensor(key_tile, "key_tile");
  check_online_softmax_attention_tensor(value_tile, "value_tile");
  CHECK_GT(scale, 0.0) << "attention scale must be positive.";
  CHECK_EQ(query.size(0), key_tile.size(0));
  CHECK_EQ(query.size(0), value_tile.size(0));
  CHECK_EQ(query.size(2), key_tile.size(2));
  CHECK_EQ(query.size(2), value_tile.size(2));
  CHECK_EQ(query.size(3), key_tile.size(3));
  CHECK_EQ(query.size(3), value_tile.size(3));

  const torch::Tensor query_by_head =
      query.transpose(/*dim0=*/1, /*dim1=*/2).to(torch::kFloat32);
  const torch::Tensor key_by_head =
      key_tile.transpose(/*dim0=*/1, /*dim1=*/2).to(torch::kFloat32);
  const torch::Tensor value_by_head =
      value_tile.transpose(/*dim0=*/1, /*dim1=*/2).to(torch::kFloat32);
  CHECK_EQ(state.max_scores.sizes(),
           torch::IntArrayRef({query_by_head.size(0),
                               query_by_head.size(1),
                               query_by_head.size(2)}));
  CHECK_EQ(state.normalizers.sizes(), state.max_scores.sizes());
  CHECK_EQ(state.weighted_values.sizes(), query_by_head.sizes());

  const torch::Tensor scores =
      torch::matmul(query_by_head,
                    key_by_head.transpose(/*dim0=*/-1,
                                          /*dim1=*/-2)) *
      scale;
  const torch::Tensor tile_max_scores =
      std::get<0>(scores.max(/*dim=*/-1, /*keepdim=*/false));
  const torch::Tensor next_max_scores =
      torch::maximum(state.max_scores, tile_max_scores);
  const torch::Tensor previous_scale =
      torch::exp(state.max_scores - next_max_scores);
  const torch::Tensor tile_probabilities =
      torch::exp(scores - next_max_scores.unsqueeze(/*dim=*/-1));

  state.normalizers =
      previous_scale * state.normalizers + tile_probabilities.sum(/*dim=*/-1);
  state.weighted_values =
      previous_scale.unsqueeze(/*dim=*/-1) * state.weighted_values +
      torch::matmul(tile_probabilities, value_by_head);
  state.max_scores = next_max_scores;
  return state;
}

torch::Tensor finalize_online_softmax_attention(
    const OnlineSoftmaxAttentionState& state,
    torch::ScalarType output_dtype) {
  CHECK(state.max_scores.defined());
  CHECK(state.normalizers.defined());
  CHECK(state.weighted_values.defined());
  CHECK(torch::all(state.normalizers > 0).item<bool>())
      << "online softmax attention requires at least one KV tile.";
  return (state.weighted_values / state.normalizers.unsqueeze(/*dim=*/-1))
      .transpose(/*dim0=*/1, /*dim1=*/2)
      .to(output_dtype);
}

NpuFusionAttentionOnlineState initialize_npu_fusion_attention_online(
    const torch::Tensor& query) {
  check_online_softmax_attention_tensor(query, "query");
  const std::vector<int64_t> score_shape = {
      query.size(0), query.size(2), query.size(1)};
  const std::vector<int64_t> weighted_value_shape = {
      query.size(0), query.size(2), query.size(1), query.size(3)};
  const torch::TensorOptions fp32_options =
      query.options().dtype(torch::kFloat32);
  NpuFusionAttentionOnlineState state;
  state.max_scores = torch::full(
      score_shape, -std::numeric_limits<float>::infinity(), fp32_options);
  state.normalizers = torch::zeros(score_shape, fp32_options);
  state.weighted_values = torch::zeros(weighted_value_shape, fp32_options);
  return state;
}

NpuFusionAttentionOnlineState update_npu_fusion_attention_online(
    NpuFusionAttentionOnlineState state,
    const torch::Tensor& query,
    const torch::Tensor& key_tile,
    const torch::Tensor& value_tile,
    int64_t num_heads,
    double scale,
    const torch::Tensor& attn_mask) {
  check_online_softmax_attention_tensor(query, "query");
  check_online_softmax_attention_tensor(key_tile, "key_tile");
  check_online_softmax_attention_tensor(value_tile, "value_tile");
  CHECK_EQ(query.size(0), key_tile.size(0));
  CHECK_EQ(query.size(0), value_tile.size(0));
  CHECK_EQ(query.size(2), key_tile.size(2));
  CHECK_EQ(query.size(2), value_tile.size(2));
  CHECK_EQ(query.size(3), key_tile.size(3));
  CHECK_EQ(query.size(3), value_tile.size(3));
  CHECK_EQ(query.size(2), num_heads);
  CHECK(state.max_scores.defined());
  CHECK(state.normalizers.defined());
  CHECK(state.weighted_values.defined());

#if defined(USE_NPU)
  const auto results = at_npu::native::custom_ops::npu_fusion_attention(
      query,
      key_tile,
      value_tile,
      num_heads,
      /*input_layout=*/"BSND",
      /*pse=*/torch::nullopt,
      /*padding_mask=*/torch::nullopt,
      /*atten_mask=*/attn_mask.defined()
          ? c10::optional<torch::Tensor>(attn_mask)
          : torch::nullopt,
      scale,
      /*keep_prob=*/1.0,
      /*pre_tockens=*/65535,
      /*next_tockens=*/65535);
  const torch::Tensor tile_output = std::get<0>(results);
  const torch::Tensor tile_max_scores =
      std::get<1>(results).select(/*dim=*/-1, /*index=*/0);
  const torch::Tensor tile_normalizers =
      std::get<2>(results).select(/*dim=*/-1, /*index=*/0);
  CHECK_EQ(tile_max_scores.sizes(), state.max_scores.sizes());
  CHECK_EQ(tile_normalizers.sizes(), state.normalizers.sizes());

  const torch::Tensor next_max_scores =
      torch::maximum(state.max_scores, tile_max_scores);
  const torch::Tensor previous_scale =
      torch::exp(state.max_scores - next_max_scores);
  const torch::Tensor tile_scale =
      torch::exp(tile_max_scores - next_max_scores);
  state.normalizers =
      previous_scale * state.normalizers + tile_scale * tile_normalizers;
  state.weighted_values =
      previous_scale.unsqueeze(/*dim=*/-1) * state.weighted_values +
      tile_scale.unsqueeze(/*dim=*/-1) *
          tile_normalizers.unsqueeze(/*dim=*/-1) *
          tile_output.transpose(/*dim0=*/1, /*dim1=*/2).to(torch::kFloat32);
  state.max_scores = next_max_scores;
  return state;
#else
  LOG(FATAL) << "NPU fusion-attention online update requires USE_NPU.";
#endif
}

torch::Tensor finalize_npu_fusion_attention_online(
    const NpuFusionAttentionOnlineState& state,
    torch::ScalarType output_dtype) {
  CHECK(state.max_scores.defined());
  CHECK(state.normalizers.defined());
  CHECK(state.weighted_values.defined());
  CHECK(torch::all(state.normalizers > 0).item<bool>())
      << "NPU fusion-attention online update requires at least one KV tile.";
  return (state.weighted_values / state.normalizers.unsqueeze(/*dim=*/-1))
      .transpose(/*dim0=*/1, /*dim1=*/2)
      .to(output_dtype);
}

torch::Tensor ring_kv_npu_fusion_attention(
    const torch::Tensor& query,
    const torch::Tensor& local_key,
    const torch::Tensor& local_value,
    int64_t num_heads,
    double scale,
    ProcessGroup* process_group,
    const std::vector<torch::Tensor>& attention_masks) {
  check_online_softmax_attention_tensor(query, "query");
  check_online_softmax_attention_tensor(local_key, "local_key");
  check_online_softmax_attention_tensor(local_value, "local_value");
  CHECK_EQ(query.size(0), local_key.size(0));
  CHECK_EQ(query.size(0), local_value.size(0));
  CHECK_EQ(query.size(2), local_key.size(2));
  CHECK_EQ(query.size(2), local_value.size(2));
  CHECK_EQ(query.size(3), local_key.size(3));
  CHECK_EQ(query.size(3), local_value.size(3));
  CHECK_EQ(query.size(2), num_heads);
  CHECK_EQ(local_key.sizes(), local_value.sizes());

  NpuFusionAttentionOnlineState state;
  torch::Tensor key_tile = local_key.contiguous();
  torch::Tensor value_tile = local_value.contiguous();
  const int32_t world_size =
      process_group == nullptr ? 1 : process_group->world_size();
  const bool packed_kv_transfer =
      world_size > 1 &&
      DiTConfig::get_instance().dit_sp_ring_kv_packed_transfer();
  const int32_t sequence_chunk_count =
      DiTConfig::get_instance().dit_sp_ring_kv_sequence_chunks();
  CHECK_GT(sequence_chunk_count, 0)
      << "Ring-KV sequence chunk count must be positive.";
  const bool use_sequence_chunk_pipeline =
      world_size > 1 && sequence_chunk_count > 1;
  if (use_sequence_chunk_pipeline) {
    CHECK_EQ(world_size, 2)
        << "Ring-KV sequence chunk pipeline currently supports SP=2 only.";
    CHECK_EQ(query.size(/*dim=*/0), 1)
        << "Ring-KV sequence chunk pipeline currently supports batch size 1 "
           "only.";
    CHECK_LE(sequence_chunk_count, local_key.size(/*dim=*/1))
        << "Ring-KV sequence chunk count cannot exceed local KV length.";
  }
#if defined(USE_NPU)
  const bool use_comm_stream =
      world_size > 1 &&
      DiTConfig::get_instance().dit_sp_ring_kv_comm_stream_overlap();
  const bool use_native_attention_update =
      world_size > 1 &&
      DiTConfig::get_instance().dit_sp_ring_kv_native_attention_update();
  const bool use_tilelang_online_update =
      world_size > 1 && !use_native_attention_update &&
      DiTConfig::get_instance().dit_sp_ring_kv_tilelang_online_update();
  const int32_t device_index = query.device().index();
  const c10_npu::NPUStream compute_stream =
      c10_npu::getCurrentNPUStream(device_index);
  c10_npu::NPUStream* comm_stream =
      use_comm_stream ? &get_dit_comm_stream(device_index) : nullptr;
  std::vector<torch::Tensor> attention_tile_lses;
  std::vector<torch::Tensor> attention_tile_outputs;
  if (use_native_attention_update) {
    const int32_t attention_tile_count =
        use_sequence_chunk_pipeline ? sequence_chunk_count + 1 : world_size;
    attention_tile_lses.reserve(attention_tile_count);
    attention_tile_outputs.reserve(attention_tile_count);
  } else {
    state = initialize_npu_fusion_attention_online(query);
  }
#else
  state = initialize_npu_fusion_attention_online(query);
#endif

  auto update_attention_tile = [&](const torch::Tensor& key,
                                   const torch::Tensor& value,
                                   const torch::Tensor& attention_mask) {
#if defined(USE_NPU)
    const auto results = at_npu::native::custom_ops::npu_fusion_attention(
        query,
        key,
        value,
        num_heads,
        /*input_layout=*/"BSND",
        /*pse=*/torch::nullopt,
        /*padding_mask=*/torch::nullopt,
        /*atten_mask=*/attention_mask.defined()
            ? c10::optional<torch::Tensor>(attention_mask)
            : torch::nullopt,
        scale,
        /*keep_prob=*/1.0,
        /*pre_tockens=*/65535,
        /*next_tockens=*/65535);
    const torch::Tensor tile_output = std::get<0>(results);
    const torch::Tensor tile_max_scores =
        std::get<1>(results).select(/*dim=*/-1, /*index=*/0);
    const torch::Tensor tile_normalizers =
        std::get<2>(results).select(/*dim=*/-1, /*index=*/0);
    if (use_native_attention_update) {
      attention_tile_lses.emplace_back(
          (tile_max_scores + torch::log(tile_normalizers))
              .transpose(/*dim0=*/1, /*dim1=*/2)
              .contiguous()
              .reshape({-1}));
      attention_tile_outputs.emplace_back(
          tile_output.reshape({-1, query.size(/*dim=*/3)}));
      return;
    }
    if (use_tilelang_online_update &&
        kernel::npu::tilelang::can_update_online_softmax_state(
            state.max_scores,
            state.normalizers,
            state.weighted_values,
            tile_output,
            std::get<1>(results),
            std::get<2>(results))) {
      kernel::npu::tilelang::update_online_softmax_state(state.max_scores,
                                                         state.normalizers,
                                                         state.weighted_values,
                                                         tile_output,
                                                         std::get<1>(results),
                                                         std::get<2>(results));
      return;
    }
    const torch::Tensor next_max_scores =
        torch::maximum(state.max_scores, tile_max_scores);
    const torch::Tensor previous_scale =
        torch::exp(state.max_scores - next_max_scores);
    const torch::Tensor tile_scale =
        torch::exp(tile_max_scores - next_max_scores);
    state.normalizers =
        previous_scale * state.normalizers + tile_scale * tile_normalizers;
    state.weighted_values =
        previous_scale.unsqueeze(/*dim=*/-1) * state.weighted_values +
        tile_scale.unsqueeze(/*dim=*/-1) *
            tile_normalizers.unsqueeze(/*dim=*/-1) *
            tile_output.transpose(/*dim0=*/1, /*dim1=*/2).to(torch::kFloat32);
    state.max_scores = next_max_scores;
    return;
#endif
    state = update_npu_fusion_attention_online(
        std::move(state), query, key, value, num_heads, scale, attention_mask);
  };

  torch::Tensor packed_kv_tile;
  if (world_size > 1) {
    process_group->warmup_p2p();
    if (packed_kv_transfer && !use_sequence_chunk_pipeline) {
      packed_kv_tile = torch::stack({key_tile, value_tile}, /*dim=*/0);
    }
  }
  CHECK(attention_masks.empty() ||
        attention_masks.size() == static_cast<size_t>(world_size))
      << "Ring-KV attention masks must be empty or contain one mask per "
         "ring step.";

  if (use_sequence_chunk_pipeline) {
    const int64_t local_kv_sequence_length = local_key.size(/*dim=*/1);
    std::vector<torch::Tensor> local_key_chunks;
    std::vector<torch::Tensor> local_value_chunks;
    std::vector<torch::Tensor> packed_local_kv_chunks;
    std::vector<torch::Tensor> remote_attention_mask_chunks;
    local_key_chunks.reserve(sequence_chunk_count);
    local_value_chunks.reserve(sequence_chunk_count);
    packed_local_kv_chunks.reserve(sequence_chunk_count);
    remote_attention_mask_chunks.reserve(sequence_chunk_count);
    for (int32_t chunk_index = 0; chunk_index < sequence_chunk_count;
         ++chunk_index) {
      const int64_t chunk_start =
          local_kv_sequence_length * chunk_index / sequence_chunk_count;
      const int64_t chunk_end =
          local_kv_sequence_length * (chunk_index + 1) / sequence_chunk_count;
      torch::Tensor local_key_chunk =
          key_tile.slice(/*dim=*/1, chunk_start, chunk_end);
      torch::Tensor local_value_chunk =
          value_tile.slice(/*dim=*/1, chunk_start, chunk_end);
      CHECK(local_key_chunk.is_contiguous())
          << "Ring-KV chunked P2P requires contiguous K chunks.";
      CHECK(local_value_chunk.is_contiguous())
          << "Ring-KV chunked P2P requires contiguous V chunks.";
      local_key_chunks.emplace_back(std::move(local_key_chunk));
      local_value_chunks.emplace_back(std::move(local_value_chunk));
      if (packed_kv_transfer) {
        packed_local_kv_chunks.emplace_back(torch::stack(
            {local_key_chunks.back(), local_value_chunks.back()}, /*dim=*/0));
      }
      if (!attention_masks.empty()) {
        remote_attention_mask_chunks.emplace_back(
            attention_masks[1].slice(/*dim=*/3, chunk_start, chunk_end));
      }
    }

    const int32_t local_rank = process_group->rank();
    const int32_t send_rank = (local_rank + 1) % world_size;
    const int32_t recv_rank = (local_rank - 1 + world_size) % world_size;
#if defined(USE_NPU)
    std::shared_ptr<c10_npu::NPUEvent> kv_ready_event;
    if (comm_stream != nullptr) {
      kv_ready_event = std::make_shared<c10_npu::NPUEvent>();
      kv_ready_event->record(compute_stream);
    }
#endif
    auto launch_chunk_exchange = [&](int32_t chunk_index,
                                     torch::Tensor* recv_key_chunk,
                                     torch::Tensor* recv_value_chunk,
                                     torch::Tensor* recv_packed_kv_chunk) {
      std::vector<std::string> op_types;
      std::vector<torch::Tensor> tensors;
      std::vector<int64_t> remote_ranks;
      if (packed_kv_transfer) {
        *recv_packed_kv_chunk =
            torch::empty_like(packed_local_kv_chunks[chunk_index]);
        op_types = {"send", "recv"};
        tensors = {packed_local_kv_chunks[chunk_index], *recv_packed_kv_chunk};
        remote_ranks = {send_rank, recv_rank};
      } else {
        *recv_key_chunk = torch::empty_like(local_key_chunks[chunk_index]);
        *recv_value_chunk = torch::empty_like(local_value_chunks[chunk_index]);
        op_types = {"send", "send", "recv", "recv"};
        tensors = {local_key_chunks[chunk_index],
                   local_value_chunks[chunk_index],
                   *recv_key_chunk,
                   *recv_value_chunk};
        remote_ranks = {send_rank, send_rank, recv_rank, recv_rank};
      }
#if defined(USE_NPU)
      if (comm_stream != nullptr) {
        c10::StreamGuard stream_guard(comm_stream->unwrap());
        if (chunk_index == 0) {
          kv_ready_event->block(*comm_stream);
        }
        return process_group->batch_isend_irecv(
            op_types, tensors, remote_ranks);
      }
#endif
      return process_group->batch_isend_irecv(op_types, tensors, remote_ranks);
    };

    torch::Tensor recv_key_chunk;
    torch::Tensor recv_value_chunk;
    torch::Tensor recv_packed_kv_chunk;
    c10::intrusive_ptr<c10d::Work> p2p_work = launch_chunk_exchange(
        /*chunk_index=*/0,
        &recv_key_chunk,
        &recv_value_chunk,
        &recv_packed_kv_chunk);
    CHECK(p2p_work != nullptr) << "Ring-KV chunk P2P must return work.";
    const torch::Tensor local_attention_mask =
        attention_masks.empty() ? torch::Tensor() : attention_masks[0];
    update_attention_tile(key_tile, value_tile, local_attention_mask);
    for (int32_t chunk_index = 0; chunk_index < sequence_chunk_count;
         ++chunk_index) {
      p2p_work->wait();

      torch::Tensor next_recv_key_chunk;
      torch::Tensor next_recv_value_chunk;
      torch::Tensor next_recv_packed_kv_chunk;
      c10::intrusive_ptr<c10d::Work> next_p2p_work;
      if (chunk_index + 1 < sequence_chunk_count) {
        next_p2p_work = launch_chunk_exchange(chunk_index + 1,
                                              &next_recv_key_chunk,
                                              &next_recv_value_chunk,
                                              &next_recv_packed_kv_chunk);
        CHECK(next_p2p_work != nullptr)
            << "Ring-KV chunk P2P must return work.";
      }
      const torch::Tensor remote_attention_mask =
          attention_masks.empty() ? torch::Tensor()
                                  : remote_attention_mask_chunks[chunk_index];
      const torch::Tensor remote_key_chunk =
          packed_kv_transfer
              ? recv_packed_kv_chunk.select(/*dim=*/0, /*index=*/0)
              : recv_key_chunk;
      const torch::Tensor remote_value_chunk =
          packed_kv_transfer
              ? recv_packed_kv_chunk.select(/*dim=*/0, /*index=*/1)
              : recv_value_chunk;
      update_attention_tile(
          remote_key_chunk, remote_value_chunk, remote_attention_mask);
      p2p_work = std::move(next_p2p_work);
      recv_key_chunk = std::move(next_recv_key_chunk);
      recv_value_chunk = std::move(next_recv_value_chunk);
      recv_packed_kv_chunk = std::move(next_recv_packed_kv_chunk);
    }
#if defined(USE_NPU)
    if (use_native_attention_update) {
      const auto merged = at_npu::native::custom_ops::npu_attention_update(
          attention_tile_lses,
          attention_tile_outputs,
          /*update_type=*/0);
      return std::get<0>(merged).reshape(query.sizes());
    }
#endif
    return finalize_npu_fusion_attention_online(state, query.scalar_type());
  }

  for (int32_t ring_step = 0; ring_step < world_size; ++ring_step) {
    torch::Tensor recv_key_tile;
    torch::Tensor recv_value_tile;
    torch::Tensor recv_packed_kv_tile;
    c10::intrusive_ptr<c10d::Work> p2p_work;
#if defined(USE_NPU)
    std::shared_ptr<c10_npu::NPUEvent> local_kv_ready_event;
#endif
    if (ring_step + 1 < world_size) {
      const int32_t local_rank = process_group->rank();
      const int32_t send_rank = (local_rank + 1) % world_size;
      const int32_t recv_rank = (local_rank - 1 + world_size) % world_size;
      std::vector<std::string> op_types;
      std::vector<torch::Tensor> tensors;
      std::vector<int64_t> remote_ranks;
      if (packed_kv_transfer) {
        recv_packed_kv_tile = torch::empty_like(packed_kv_tile);
        op_types = {"send", "recv"};
        tensors = {packed_kv_tile, recv_packed_kv_tile};
        remote_ranks = {send_rank, recv_rank};
      } else {
        recv_key_tile = torch::empty_like(key_tile);
        recv_value_tile = torch::empty_like(value_tile);
        op_types = {"send", "send", "recv", "recv"};
        tensors = {key_tile, value_tile, recv_key_tile, recv_value_tile};
        remote_ranks = {send_rank, send_rank, recv_rank, recv_rank};
      }
#if defined(USE_NPU)
      if (comm_stream != nullptr) {
        local_kv_ready_event = std::make_shared<c10_npu::NPUEvent>();
        local_kv_ready_event->record(compute_stream);
        {
          c10::StreamGuard stream_guard(comm_stream->unwrap());
          local_kv_ready_event->block(*comm_stream);
          p2p_work =
              process_group->batch_isend_irecv(op_types, tensors, remote_ranks);
        }
      } else {
        p2p_work =
            process_group->batch_isend_irecv(op_types, tensors, remote_ranks);
      }
#else
      p2p_work =
          process_group->batch_isend_irecv(op_types, tensors, remote_ranks);
#endif
      CHECK(p2p_work != nullptr) << "Ring-KV P2P exchange must return work.";
    }

    const torch::Tensor attention_mask =
        attention_masks.empty() ? torch::Tensor() : attention_masks[ring_step];
    update_attention_tile(key_tile, value_tile, attention_mask);
    if (ring_step + 1 == world_size) {
      break;
    }
    p2p_work->wait();
    if (packed_kv_transfer) {
      packed_kv_tile = std::move(recv_packed_kv_tile);
      key_tile = packed_kv_tile.select(/*dim=*/0, /*index=*/0);
      value_tile = packed_kv_tile.select(/*dim=*/0, /*index=*/1);
    } else {
      key_tile = std::move(recv_key_tile);
      value_tile = std::move(recv_value_tile);
    }
  }
#if defined(USE_NPU)
  if (use_native_attention_update) {
    const auto merged =
        at_npu::native::custom_ops::npu_attention_update(attention_tile_lses,
                                                         attention_tile_outputs,
                                                         /*update_type=*/0);
    return std::get<0>(merged).reshape(query.sizes());
  }
#endif
  return finalize_npu_fusion_attention_online(state, query.scalar_type());
}

torch::Tensor ring_kv_online_softmax_attention_reference(
    const torch::Tensor& query,
    const torch::Tensor& local_key,
    const torch::Tensor& local_value,
    double scale,
    ProcessGroup* process_group) {
  check_online_softmax_attention_tensor(query, "query");
  check_online_softmax_attention_tensor(local_key, "local_key");
  check_online_softmax_attention_tensor(local_value, "local_value");
  CHECK_EQ(query.size(0), local_key.size(0));
  CHECK_EQ(query.size(0), local_value.size(0));
  CHECK_EQ(query.size(2), local_key.size(2));
  CHECK_EQ(query.size(2), local_value.size(2));
  CHECK_EQ(query.size(3), local_key.size(3));
  CHECK_EQ(query.size(3), local_value.size(3));

  OnlineSoftmaxAttentionState state =
      initialize_online_softmax_attention(query);
  torch::Tensor key_tile = local_key.contiguous();
  torch::Tensor value_tile = local_value.contiguous();
  const int32_t world_size =
      process_group == nullptr ? 1 : process_group->world_size();
  if (world_size > 1) {
    CHECK_EQ(local_key.sizes(), local_value.sizes());
    process_group->warmup_p2p();
  }

  for (int32_t ring_step = 0; ring_step < world_size; ++ring_step) {
    torch::Tensor recv_key_tile;
    torch::Tensor recv_value_tile;
    c10::intrusive_ptr<c10d::Work> p2p_work;
    if (ring_step + 1 < world_size) {
      const int32_t local_rank = process_group->rank();
      const int32_t send_rank = (local_rank + 1) % world_size;
      const int32_t recv_rank = (local_rank - 1 + world_size) % world_size;
      recv_key_tile = torch::empty_like(key_tile);
      recv_value_tile = torch::empty_like(value_tile);
      std::vector<std::string> op_types = {"send", "send", "recv", "recv"};
      std::vector<torch::Tensor> tensors = {
          key_tile, value_tile, recv_key_tile, recv_value_tile};
      std::vector<int64_t> remote_ranks = {
          send_rank, send_rank, recv_rank, recv_rank};
      p2p_work =
          process_group->batch_isend_irecv(op_types, tensors, remote_ranks);
      CHECK(p2p_work != nullptr) << "Ring-KV P2P exchange must return work.";
    }

    state = update_online_softmax_attention(
        std::move(state), query, key_tile, value_tile, scale);
    if (ring_step + 1 == world_size) {
      break;
    }
    p2p_work->wait();
    key_tile = std::move(recv_key_tile);
    value_tile = std::move(recv_value_tile);
  }
  return finalize_online_softmax_attention(state, query.scalar_type());
}

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args) {
  if (parallel_args.dp_size() <= 1) {
    return std::nullopt;
  }

  // tp=1 in each dp group
  if (parallel_args.dp_size() == parallel_args.world_size()) {
    return ParallelArgs(0,  // local rank
                        1,  // world_size
                        nullptr,
                        nullptr,
                        parallel_args.dp_size());
  }

  return ParallelArgs(parallel_args.dp_local_process_group_->rank(),
                      parallel_args.dp_local_process_group_->world_size(),
                      parallel_args.dp_local_process_group_,
                      nullptr,
                      parallel_args.dp_size());
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     int32_t dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  torch::Tensor stacked = process_group->allgather_base_sync(input);
  return torch::cat(stacked.unbind(0), /*dim=*/dim).contiguous();
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     const std::vector<int32_t>& token_num_list) {
  if (!process_group) return input;
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) return input;

  CHECK_EQ(token_num_list.size(), world_size)
      << "token_num_list size " << token_num_list.size()
      << " does not match world_size " << world_size;

  const bool num_tokens_equal =
      std::all_of(token_num_list.begin(),
                  token_num_list.end(),
                  [first_token_num = token_num_list[0]](int64_t num) {
                    return num == first_token_num;
                  });
  if (num_tokens_equal) {
    return gather(input, process_group, 0);
  }
  return finish_gather(launch_gather(input, process_group, token_num_list));
}

torch::Tensor finish_gather(GatherAsyncCtx ctx) {
  if (ctx.work.defined()) {
    ctx.work->wait();
  }
  if (ctx.stacked.defined() && ctx.stacked.size(0) == 1 &&
      ctx.token_num_list.size() == 1) {
    if (ctx.stacked[0].size(0) == ctx.token_num_list.front()) {
      return ctx.stacked[0];
    }
  }
  const bool num_tokens_equal =
      !ctx.token_num_list.empty() &&
      std::all_of(ctx.token_num_list.begin(),
                  ctx.token_num_list.end(),
                  [first_token_num = ctx.token_num_list[0]](int64_t num) {
                    return num == first_token_num;
                  });
  if (num_tokens_equal) {
    return ctx.stacked.flatten(0, 1).contiguous();
  }
  return assemble_gathered(ctx.stacked, ctx.token_num_list);
}

torch::Tensor all_gather_interleaved(const torch::Tensor& input,
                                     ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  torch::Tensor gathered_tensors = process_group->allgather_base_sync(input);

  int32_t dim = -1;
  size_t num_chunks = 3;
  std::vector<torch::Tensor> ordered_tensors;
  int64_t shard_size = input.size(dim) / num_chunks;
  for (size_t i = 0; i < num_chunks; ++i) {
    for (size_t j = 0; j < world_size; ++j) {
      auto shard_tensor =
          gathered_tensors[j].slice(dim, shard_size * i, shard_size * (i + 1));
      ordered_tensors.push_back(shard_tensor);
    }
  }
  return torch::cat(ordered_tensors, dim).contiguous();
}

torch::Tensor finish_reduce(ReduceAsyncCtx ctx) {
  if (ctx.work.defined()) {
    ctx.work->wait();
  }
  return ctx.tensor;
}

torch::Tensor reduce(torch::Tensor& input, ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }
  return finish_reduce(launch_reduce(input, process_group));
}

torch::Tensor reduce_scatter(const torch::Tensor& input,
                             ProcessGroup* process_group) {
  // currently only support scatter_dim == 0
  if (!process_group) return input;
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) return input;

  const int32_t rank = process_group->rank();
  const int64_t original_dim_size = input.size(0);

  // check if padding is needed
  // round up to the nearest multiple of world_size: (N + W - 1) / W * W or N +
  // (W - N%W)%W
  int64_t remainder = original_dim_size % world_size;
  int64_t target_size = (remainder == 0)
                            ? original_dim_size
                            : (original_dim_size + world_size - remainder);
  int64_t num_padding = target_size - original_dim_size;
  torch::Tensor padded_input = input;
  if (num_padding > 0) {
    std::vector<int64_t> pad = {0, 0, 0, num_padding};
    // Explicitly calling kConstant and value of 0 ensures consistency across
    // platforms and versions.
    padded_input = torch::nn::functional::pad(
        input, torch::nn::functional::PadFuncOptions(pad));
  }

  // prepare output tensor
  // at this point, padded_input size along dim 0 is divisible by world_size
  const int64_t padded_dim_size = padded_input.size(0);
  const int64_t chunk_size = padded_dim_size / world_size;

  auto output_shape = padded_input.sizes().vec();
  output_shape[0] = chunk_size;
  torch::Tensor output = torch::empty(output_shape, padded_input.options());

  // perform reduce scatter operation
  process_group->reduce_scatter(padded_input, output);

  // remove padding
  if (num_padding > 0) {
    int64_t global_start = rank * chunk_size;
    int64_t global_end = global_start + chunk_size;

    if (global_start >= original_dim_size) {
      return output.slice(0, 0, 0);
    } else if (global_end > original_dim_size) {
      return output.slice(0, 0, original_dim_size - global_start);
    }
  }

  return output;
}

std::vector<int32_t> compute_cp_group_ranks(int32_t global_rank,
                                            int32_t world_size,
                                            int32_t dp_size,
                                            int32_t cp_size) {
  CHECK_GT(cp_size, 1) << "compute_cp_group_ranks requires cp_size > 1.";
  CHECK_GT(dp_size, 0) << "dp_size must be positive.";
  CHECK_GT(world_size, 0) << "world_size must be positive.";
  CHECK_EQ(world_size % (dp_size * cp_size), 0)
      << "world_size (" << world_size
      << ") must be divisible by dp_size * cp_size (" << dp_size * cp_size
      << ") so that attn_tp_size is integral.";
  const int32_t attn_tp_size = world_size / (dp_size * cp_size);
  CHECK_GE(global_rank, 0);
  CHECK_LT(global_rank, world_size);

  // rank layout: dp_rank * (cp_size * attn_tp_size) + cp_rank * attn_tp_size +
  // tp_rank
  const int32_t tp_stride = attn_tp_size;
  const int32_t dp_stride = cp_size * attn_tp_size;
  const int32_t dp_rank = global_rank / dp_stride;
  const int32_t tp_rank = global_rank % attn_tp_size;

  std::vector<int32_t> ranks;
  ranks.reserve(cp_size);
  for (int32_t cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    ranks.push_back(dp_rank * dp_stride + cp_rank * tp_stride + tp_rank);
  }
  return ranks;
}

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  // get the size for last dimension
  const int32_t dim_size = input.size(dim);
  CHECK(dim_size % world_size == 0)
      << "dim_size " << dim_size << " cannot be divided by world_size "
      << world_size;

  const auto tensor_list = input.split(dim_size / world_size, dim);
  const int32_t rank = process_group->rank();
  return tensor_list[rank];
}

std::function<torch::Tensor()> all_to_all_4D(const torch::Tensor& input,
                                             int32_t scatter_idx,
                                             int32_t gather_idx,
                                             bool async_ops,
                                             ProcessGroup* process_group) {
  if (!process_group) {
    return [input]() { return input; };
  }
  const int32_t group_size = process_group->world_size();

  if (group_size == 1) {
    return [input]() { return input; };
  }

  TORCH_CHECK(input.dim() == 4,
              "all_to_all_4D: input must be 4D, got dim=",
              input.dim());
  torch::Tensor send_input = input;

  if (scatter_idx == 2 && gather_idx == 1) {
    // branch A : from "sequence shard" -> "head shard"
    // input: (bs, seqlen / group_size (shard_seqlen), head_num, head_dim)
    //   output (bs, seqlen, head_num / group_size, head_dim)
    auto sizes = send_input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t shard_seqlen = sizes[1];
    const int64_t head_num = sizes[2];
    const int64_t head_size = sizes[3];
    const int64_t seqlen = shard_seqlen * group_size;
    TORCH_CHECK(head_num % group_size == 0,
                "all_to_all_4D(A): head_num must be divisible by group_size");
    const int64_t shard_head_num = head_num / group_size;

    // prepare expected shape for All2All (group_size, shard_seqlen, bs,
    // shard_head_num, head_size)
    auto input_t =
        send_input
            .reshape({bs, shard_seqlen, group_size, shard_head_num, head_size})
            .transpose(
                0,
                2)  // (group_size, shard_seqlen, bs, shard_head_num, head_size)
            .contiguous();
    torch::Tensor output = torch::empty_like(input_t);
    std::vector<int64_t> input_split_size = {};
    std::vector<int64_t> output_split_size = {};

    if (!async_ops) {
      process_group->all_to_all_single(
          output, input_t, output_split_size, input_split_size, async_ops);
      output = output.reshape({seqlen, bs, shard_head_num, head_size})
                   .transpose(0, 1)
                   .contiguous()
                   .reshape({bs, seqlen, shard_head_num, head_size});
      return [output]() { return output; };
    } else {
      c10::intrusive_ptr<c10d::Work> all2all_work;
      process_group->all_to_all_single(output,
                                       input_t,
                                       output_split_size,
                                       input_split_size,
                                       async_ops,
                                       &all2all_work);
      return [output,
              all2all_work,
              bs,
              seqlen,
              shard_head_num,
              head_size]() mutable -> torch::Tensor {
        all2all_work->wait();
        return output.reshape({seqlen, bs, shard_head_num, head_size})
            .transpose(0, 1)
            .contiguous()
            .reshape({bs, seqlen, shard_head_num, head_size});
      };
    }
  } else if (scatter_idx == 1 && gather_idx == 2) {
    // branch B : from "head shard" -> "sequence shard"
    // input: (bs, seqlen, head_num / group_size, head_size)
    // output (bs, seqlen / group_size, head_num, haed_size)
    auto sizes = send_input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t seqlen = sizes[1];
    const int64_t shard_head_num = sizes[2];
    const int64_t head_size = sizes[3];
    TORCH_CHECK(seqlen % group_size == 0,
                "all_to_all_4D(B): seqlen must be divisible by group_size");
    const int64_t shard_seqlen = seqlen / group_size;
    const int64_t head_num = shard_head_num * group_size;

    // prepare expected shape for All2All (group_size, shard_head_num,
    // shard_seqlen, bs, head_size)
    auto input_t =
        send_input
            .reshape({bs, group_size, shard_seqlen, shard_head_num, head_size})
            .transpose(
                0,
                3)  // (shard_head_num, group_size, shard_seqlen, bs, head_size)
            .transpose(
                0,
                1)  // (group_size, shard_head_num, shard_seqlen, bs, head_size)
            .contiguous();
    torch::Tensor output = torch::empty_like(input_t);
    std::vector<int64_t> input_split_size = {};
    std::vector<int64_t> output_split_size = {};

    if (!async_ops) {
      process_group->all_to_all_single(output,
                                       input_t,
                                       output_split_size,
                                       input_split_size,
                                       /*async_op=*/false);
      output = output.reshape({head_num, shard_seqlen, bs, head_size})
                   .transpose(0, 2)
                   .contiguous()
                   .reshape({bs, shard_seqlen, head_num, head_size});
      return [output]() { return output; };
    } else {
      c10::intrusive_ptr<c10d::Work> all2all_work;
      process_group->all_to_all_single(output,
                                       input_t,
                                       output_split_size,
                                       input_split_size,
                                       /*async_op=*/true,
                                       &all2all_work);
      return [output,
              all2all_work,
              head_num,
              shard_seqlen,
              bs,
              head_size]() mutable -> torch::Tensor {
        all2all_work->wait();
        auto comm_output =
            output.reshape({head_num, shard_seqlen, bs, head_size})
                .transpose(0, 2)
                .contiguous()
                .reshape({bs, shard_seqlen, head_num, head_size});
        return comm_output;
      };
    }
  } else {
    TORCH_CHECK(false,
                "all_to_all_4D: only (scatter_idx,gather_idx)=(2,1) or (1,2) "
                "are supported");
  }
}

std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>()>
all_to_all_4D_packed_qkv(const torch::Tensor& query,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         bool async_ops,
                         ProcessGroup* process_group,
                         const std::string& buffer_role) {
  if (process_group == nullptr || process_group->world_size() == 1) {
    return [query, key, value]() { return std::make_tuple(query, key, value); };
  }

  const int32_t group_size = process_group->world_size();
  const bool can_pack =
      query.dim() == 4 && query.sizes() == key.sizes() &&
      query.sizes() == value.sizes() &&
      query.scalar_type() == key.scalar_type() &&
      query.scalar_type() == value.scalar_type() &&
      query.device() == key.device() && query.device() == value.device() &&
      query.is_contiguous() && key.is_contiguous() && value.is_contiguous() &&
      query.size(2) % group_size == 0;
  if (!can_pack) {
    auto query_work = all_to_all_4D(query,
                                    /*scatter_idx=*/2,
                                    /*gather_idx=*/1,
                                    async_ops,
                                    process_group);
    auto key_work = all_to_all_4D(key,
                                  /*scatter_idx=*/2,
                                  /*gather_idx=*/1,
                                  async_ops,
                                  process_group);
    auto value_work = all_to_all_4D(value,
                                    /*scatter_idx=*/2,
                                    /*gather_idx=*/1,
                                    async_ops,
                                    process_group);
    return [query_work = std::move(query_work),
            key_work = std::move(key_work),
            value_work = std::move(value_work)]() mutable {
      return std::make_tuple(query_work(), key_work(), value_work());
    };
  }

  const int64_t batch_size = query.size(0);
  const int64_t shard_sequence_length = query.size(1);
  const int64_t global_head_num = query.size(2);
  const int64_t head_size = query.size(3);
  const int64_t local_head_num = global_head_num / group_size;
  const std::vector<int64_t> packed_shape = {group_size,
                                             shard_sequence_length,
                                             batch_size,
                                             local_head_num,
                                             3 * head_size};
  torch::Tensor packed_input =
      get_a2a_staging_buffer(query, packed_shape, buffer_role + "_input");
  torch::Tensor packed_output =
      get_a2a_staging_buffer(query, packed_shape, buffer_role + "_output");

#if defined(USE_NPU)
  if (DiTConfig::get_instance().dit_sp_packed_qkv_triton_pack() &&
      kernel::npu::can_pack_qkv_destination_major_triton(
          query, key, value, packed_input, group_size)) {
    kernel::npu::pack_qkv_destination_major_triton(
        query, key, value, packed_input, group_size);
  } else {
#endif
    const std::vector<int64_t> qkv_view_shape = {batch_size,
                                                 shard_sequence_length,
                                                 group_size,
                                                 local_head_num,
                                                 head_size};
    packed_input.slice(/*dim=*/4, /*start=*/0, /*end=*/head_size)
        .copy_(query.reshape(qkv_view_shape).permute({2, 1, 0, 3, 4}));
    packed_input.slice(/*dim=*/4, /*start=*/head_size, /*end=*/2 * head_size)
        .copy_(key.reshape(qkv_view_shape).permute({2, 1, 0, 3, 4}));
    packed_input
        .slice(/*dim=*/4, /*start=*/2 * head_size, /*end=*/3 * head_size)
        .copy_(value.reshape(qkv_view_shape).permute({2, 1, 0, 3, 4}));
#if defined(USE_NPU)
  }
#endif

  std::vector<int64_t> input_split_size = {};
  std::vector<int64_t> output_split_size = {};
  if (!async_ops) {
    process_group->all_to_all_single(packed_output,
                                     packed_input,
                                     output_split_size,
                                     input_split_size,
                                     /*async_op=*/false);
    return [packed_output,
            group_size,
            batch_size,
            shard_sequence_length,
            local_head_num,
            head_size]() {
      return unpack_packed_qkv(packed_output,
                               group_size,
                               batch_size,
                               shard_sequence_length,
                               local_head_num,
                               head_size);
    };
  }

  c10::intrusive_ptr<c10d::Work> all2all_work;
  process_group->all_to_all_single(packed_output,
                                   packed_input,
                                   output_split_size,
                                   input_split_size,
                                   /*async_op=*/true,
                                   &all2all_work);
  return [packed_input,
          packed_output,
          all2all_work,
          group_size,
          batch_size,
          shard_sequence_length,
          local_head_num,
          head_size]() mutable {
    all2all_work->wait();
    return unpack_packed_qkv(packed_output,
                             group_size,
                             batch_size,
                             shard_sequence_length,
                             local_head_num,
                             head_size);
  };
}

std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>()>
all_to_all_4D_packed_fused_qkv(const torch::Tensor& qkv,
                               bool async_ops,
                               ProcessGroup* process_group,
                               const std::string& buffer_role,
                               PackedQkvPostprocessor postprocessor) {
  CHECK_EQ(qkv.dim(), 5) << "fused QKV must have shape [B, S, 3, H, D].";
  CHECK_EQ(qkv.size(2), 3) << "fused QKV dimension 2 must have size 3.";
  CHECK(qkv.is_contiguous()) << "fused QKV must be contiguous.";

  if (process_group == nullptr || process_group->world_size() == 1) {
    return [qkv]() {
      return std::make_tuple(qkv.select(/*dim=*/2, /*index=*/0),
                             qkv.select(/*dim=*/2, /*index=*/1),
                             qkv.select(/*dim=*/2, /*index=*/2));
    };
  }

  const int32_t group_size = process_group->world_size();
  CHECK_EQ(qkv.size(3) % group_size, 0)
      << "QKV head count must divide sequence-parallel world size.";

  const int64_t batch_size = qkv.size(0);
  const int64_t shard_sequence_length = qkv.size(1);
  const int64_t global_head_num = qkv.size(3);
  const int64_t head_size = qkv.size(4);
  const int64_t local_head_num = global_head_num / group_size;
  const std::vector<int64_t> packed_shape = {group_size,
                                             shard_sequence_length,
                                             batch_size,
                                             local_head_num,
                                             3 * head_size};
  torch::Tensor packed_input =
      get_a2a_staging_buffer(qkv, packed_shape, buffer_role + "_input");
  torch::Tensor packed_output =
      get_a2a_staging_buffer(qkv, packed_shape, buffer_role + "_output");

#if defined(USE_NPU)
  if (DiTConfig::get_instance().dit_sp_packed_qkv_triton_pack() &&
      kernel::npu::can_pack_fused_qkv_destination_major_triton(
          qkv, packed_input, group_size)) {
    kernel::npu::pack_fused_qkv_destination_major_triton(
        qkv, packed_input, group_size);
  } else {
#endif
    const std::vector<int64_t> qkv_view_shape = {batch_size,
                                                 shard_sequence_length,
                                                 3,
                                                 group_size,
                                                 local_head_num,
                                                 head_size};
    packed_input
        .view({group_size,
               shard_sequence_length,
               batch_size,
               local_head_num,
               3,
               head_size})
        .copy_(qkv.view(qkv_view_shape).permute({3, 1, 0, 4, 2, 5}));
#if defined(USE_NPU)
  }
#endif

  std::vector<int64_t> input_split_size = {};
  std::vector<int64_t> output_split_size = {};
  if (!async_ops) {
    process_group->all_to_all_single(packed_output,
                                     packed_input,
                                     output_split_size,
                                     input_split_size,
                                     /*async_op=*/false);
    return [packed_output,
            group_size,
            batch_size,
            shard_sequence_length,
            local_head_num,
            head_size,
            postprocessor = std::move(postprocessor)]() {
      if (postprocessor) {
        return postprocessor(packed_output);
      }
      return unpack_packed_qkv(packed_output,
                               group_size,
                               batch_size,
                               shard_sequence_length,
                               local_head_num,
                               head_size);
    };
  }

  c10::intrusive_ptr<c10d::Work> all2all_work;
#if defined(USE_NPU)
  if (DiTConfig::get_instance().dit_sp_packed_qkv_comm_stream_overlap()) {
    const int32_t device_index = qkv.device().index();
    const c10_npu::NPUStream compute_stream =
        c10_npu::getCurrentNPUStream(device_index);
    c10_npu::NPUStream& comm_stream = get_dit_comm_stream(device_index);
    auto input_ready_event = std::make_shared<c10_npu::NPUEvent>();
    input_ready_event->record(compute_stream);
    {
      c10::StreamGuard stream_guard(comm_stream.unwrap());
      input_ready_event->block(comm_stream);
      process_group->all_to_all_single(packed_output,
                                       packed_input,
                                       output_split_size,
                                       input_split_size,
                                       /*async_op=*/true,
                                       &all2all_work);
    }
    return [packed_input,
            packed_output,
            all2all_work,
            input_ready_event,
            group_size,
            batch_size,
            shard_sequence_length,
            local_head_num,
            head_size,
            postprocessor = std::move(postprocessor)]() mutable {
      all2all_work->wait();
      if (postprocessor) {
        return postprocessor(packed_output);
      }
      return unpack_packed_qkv(packed_output,
                               group_size,
                               batch_size,
                               shard_sequence_length,
                               local_head_num,
                               head_size);
    };
  }
#endif
  process_group->all_to_all_single(packed_output,
                                   packed_input,
                                   output_split_size,
                                   input_split_size,
                                   /*async_op=*/true,
                                   &all2all_work);
  return [packed_input,
          packed_output,
          all2all_work,
          group_size,
          batch_size,
          shard_sequence_length,
          local_head_num,
          head_size,
          postprocessor = std::move(postprocessor)]() mutable {
    all2all_work->wait();
    if (postprocessor) {
      return postprocessor(packed_output);
    }
    return unpack_packed_qkv(packed_output,
                             group_size,
                             batch_size,
                             shard_sequence_length,
                             local_head_num,
                             head_size);
  };
}

std::vector<
    std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>()>>
all_to_all_4D_packed_fused_qkv_chunked(const torch::Tensor& qkv,
                                       int32_t tile_count,
                                       ProcessGroup* process_group,
                                       const std::string& buffer_role) {
  CHECK_EQ(qkv.dim(), 5) << "fused QKV must have shape [B, S, 3, H, D].";
  CHECK_EQ(qkv.size(2), 3) << "fused QKV dimension 2 must have size 3.";
  CHECK(qkv.is_contiguous()) << "fused QKV must be contiguous.";
  CHECK_GT(tile_count, 0)
      << "QKV attention-overlap tile count must be positive.";

  const int32_t group_size =
      process_group == nullptr ? 1 : process_group->world_size();
  CHECK_EQ(qkv.size(3) % group_size, 0)
      << "QKV head count must divide sequence-parallel world size.";
  const int64_t local_head_num = qkv.size(3) / group_size;
  CHECK_EQ(local_head_num % tile_count, 0)
      << "Local QKV head count must divide attention-overlap tile count.";

  const int64_t batch_size = qkv.size(0);
  const int64_t shard_sequence_length = qkv.size(1);
  const int64_t head_size = qkv.size(4);
  const int64_t tile_head_num = local_head_num / tile_count;
  std::vector<
      std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>()>>
      tile_works;
  tile_works.reserve(tile_count);

  if (group_size == 1) {
    for (int32_t tile_index = 0; tile_index < tile_count; ++tile_index) {
      const int64_t head_start = tile_index * tile_head_num;
      const torch::Tensor qkv_tile =
          qkv.slice(/*dim=*/3, head_start, head_start + tile_head_num);
      tile_works.emplace_back([qkv_tile]() {
        return std::make_tuple(qkv_tile.select(/*dim=*/2, /*index=*/0),
                               qkv_tile.select(/*dim=*/2, /*index=*/1),
                               qkv_tile.select(/*dim=*/2, /*index=*/2));
      });
    }
    return tile_works;
  }

  const std::vector<int64_t> qkv_view_shape = {batch_size,
                                               shard_sequence_length,
                                               3,
                                               group_size,
                                               local_head_num,
                                               head_size};
  const torch::Tensor qkv_view = qkv.view(qkv_view_shape);
  const std::vector<int64_t> packed_shape = {group_size,
                                             shard_sequence_length,
                                             batch_size,
                                             tile_head_num,
                                             3 * head_size};
  const std::vector<int64_t> input_split_size = {};
  const std::vector<int64_t> output_split_size = {};
#if defined(USE_NPU)
  const bool use_comm_stream =
      DiTConfig::get_instance().dit_sp_packed_qkv_comm_stream_overlap();
  const int32_t device_index = qkv.device().index();
  const c10_npu::NPUStream compute_stream =
      c10_npu::getCurrentNPUStream(device_index);
  c10_npu::NPUStream* comm_stream =
      use_comm_stream ? &get_dit_comm_stream(device_index) : nullptr;
#endif
  for (int32_t tile_index = 0; tile_index < tile_count; ++tile_index) {
    const int64_t head_start = tile_index * tile_head_num;
    const std::string tile_buffer_role =
        buffer_role + "_attention_tile_" + std::to_string(tile_index);
    torch::Tensor packed_input =
        get_a2a_staging_buffer(qkv, packed_shape, tile_buffer_role + "_input");
    torch::Tensor packed_output =
        get_a2a_staging_buffer(qkv, packed_shape, tile_buffer_role + "_output");
    packed_input
        .view({group_size,
               shard_sequence_length,
               batch_size,
               tile_head_num,
               3,
               head_size})
        .copy_(qkv_view.slice(/*dim=*/4, head_start, head_start + tile_head_num)
                   .permute({3, 1, 0, 4, 2, 5}));

    c10::intrusive_ptr<c10d::Work> all2all_work;
#if defined(USE_NPU)
    std::shared_ptr<c10_npu::NPUEvent> input_ready_event;
    if (comm_stream != nullptr) {
      input_ready_event = std::make_shared<c10_npu::NPUEvent>();
      input_ready_event->record(compute_stream);
      c10::StreamGuard stream_guard(comm_stream->unwrap());
      input_ready_event->block(*comm_stream);
      process_group->all_to_all_single(packed_output,
                                       packed_input,
                                       output_split_size,
                                       input_split_size,
                                       /*async_op=*/true,
                                       &all2all_work);
    } else {
#endif
      process_group->all_to_all_single(packed_output,
                                       packed_input,
                                       output_split_size,
                                       input_split_size,
                                       /*async_op=*/true,
                                       &all2all_work);
#if defined(USE_NPU)
    }
#endif
    tile_works.emplace_back([packed_input,
                             packed_output,
                             all2all_work,
#if defined(USE_NPU)
                             input_ready_event,
#endif
                             group_size,
                             batch_size,
                             shard_sequence_length,
                             tile_head_num,
                             head_size]() mutable {
      all2all_work->wait();
      return unpack_packed_qkv(packed_output,
                               group_size,
                               batch_size,
                               shard_sequence_length,
                               tile_head_num,
                               head_size);
    });
  }
  return tile_works;
}

std::vector<std::unique_ptr<ProcessGroup>> create_npu_process_groups(
    const std::vector<torch::Device>& devices) {
#if defined(USE_NPU)
  CHECK(!devices.empty()) << "devices should not be empty";

  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }

  std::vector<HcclComm> comms(devices.size());
  const int32_t world_size = static_cast<int32_t>(devices.size());
  HCCLCHECK(HcclCommInitAll(world_size, device_idxs.data(), comms.data()));

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());
  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupImpl>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }

  return process_groups;
#else
  LOG(FATAL) << "non-NPU device is not supported";
#endif
}

std::vector<std::unique_ptr<ProcessGroup>> create_local_process_groups(
    const std::vector<torch::Device>& devices,
    const runtime::Options& options) {
  CHECK(!devices.empty()) << "devices should not be empty";
  const int32_t world_size = static_cast<int32_t>(devices.size());

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());

#if defined(USE_NPU)
  std::vector<HcclComm> comms(devices.size());
  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupImpl>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_ILU) || \
    defined(USE_DCU)
  // For GPU: use create_process_group with localhost
  // Parse port from options.master_node_addr() to support multiple instances
  std::string host;
  int port;

  // Parse port from options.master_node_addr()
  // Note: master_node_addr always has a default value (127.0.0.1:19888)
  net::parse_host_port_from_addr(
      options.master_node_addr().value(), host, port);

  // Override host to localhost for local communication
  host = "127.0.0.1";

  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(create_process_group(
        /*rank=*/i,
        /*world_size=*/world_size,
        /*rank_size=*/world_size,
        /*port=*/port,
        /*trans=*/false,
        host,
        /*group_name=*/"local_tp_group",
        devices[i]));
  }
#else
  LOG(FATAL) << "Unsupported device type for create_local_process_groups";
#endif

  return process_groups;
}

}  // namespace parallel_state
}  // namespace xllm
