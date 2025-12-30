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

#include "npu_process_group.h"

#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/TCPStore.hpp>
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>

namespace {
#if defined(USE_NPU)
#define HCCLCHECK(cmd)                      \
  do {                                      \
    HcclResult r = cmd;                     \
    if (r != HCCL_SUCCESS) {                \
      LOG(FATAL) << "Failed, HCCL error :"; \
    }                                       \
  } while (0)
#endif
inline bool is_npu(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}
inline bool is_npu(const at::TensorOptions& options) {
  return options.device().is_privateuseone();
}
inline bool is_npu(const at::Device& device) {
  return device.is_privateuseone();
}
at::Tensor flatten_for_scatter_gather(std::vector<at::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}
#if defined(USE_NPU)
HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
  switch (type) {
    case at::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case at::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case at::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case at::kLong:
      return HCCL_DATA_TYPE_INT64;
    case at::kInt:
      return HCCL_DATA_TYPE_INT32;
    case at::kChar:
      return HCCL_DATA_TYPE_INT8;
    case at::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      TORCH_CHECK(false, "Unconvertible HCCL type ", type);
  }
}
#endif
void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}
}  // namespace

namespace xllm {

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   bool trans,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(global_rank, world_size, device),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
  hccl_pg_options->group_name = group_name;
#endif
  int32_t rank = global_rank;
  if (world_size != rank_size) {
    auto [local_rank, group_ranks] =
        get_group_rank(world_size, global_rank, rank_size, trans);
    std::vector<uint32_t> uint32_ranks;
    for (auto rank : group_ranks) {
      uint32_ranks.push_back(static_cast<uint32_t>(rank));
    }
    hccl_pg_options->global_ranks_in_group = uint32_ranks;
    rank = local_rank;
  }

  auto store = create_tcp_store(host, port, rank);
  pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(
      store, rank, rank_size, hccl_pg_options);
}

// Destructor.
ProcessGroupImpl::~ProcessGroupImpl() {
  if (pg_) {
    pg_->shutdown();
  } else {
    HCCLCHECK(HcclCommDestroy(comm_));
  }
}

ProcessGroupImpl::ProcessGroupImpl(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(rank, world_size, device),
      comm_(comm),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {}

// void ProcessGroupImpl::allgather(
//     const torch::Tensor& input,
//     std::vector<torch::Tensor>& outputs) {

//   CHECK(outputs.size() == world_size())
//       << "outputs should have the same size as world_size";
//   DCHECK(input.device() == device())
//       << "input should be on the same device as the process group";

//   torch::DeviceGuard guard(device());

//   // 1. flatten 输出
//   torch::Tensor flattened_output =
//       flatten_for_scatter_gather(outputs);

//   const auto count = input.numel();
//   const auto data_type = to_hccl_data_type(input);

//   auto compute_stream = c10_npu::getCurrentNPUStream();

//   // 2. compute -> comm
//   c10_npu::NPUEvent ready;
//   ready.record(compute_stream);
//   ready.block(comm_stream_);

//   // 3. allocator 记录
//   c10_npu::NPUCachingAllocator::recordStream(
//       input.storage().data_ptr(), comm_stream_);
//   c10_npu::NPUCachingAllocator::recordStream(
//       flattened_output.storage().data_ptr(), comm_stream_);

//   // 4. 发起 AllGather（异步）
//   HCCLCHECK(HcclAllGather(
//       input.data_ptr(),
//       flattened_output.data_ptr(),
//       count,
//       data_type,
//       comm_,
//       comm_stream_.stream()));

//   // 5. 只记录完成事件
//   last_comm_event_.record(comm_stream_);
// }

// void ProcessGroupImpl::allgather(const torch::Tensor& input,
//                          std::vector<torch::Tensor>& outputs) {
//   // check_input(input);
//   CHECK(outputs.size() == world_size())
//       << "outputs should have the same size as world_size";
//   DCHECK(input.device() == device())
//       << "input should be on the same device as the process group";
//   torch::DeviceGuard device_guard(device());
//   torch::Tensor flattened_output = flatten_for_scatter_gather(outputs);
//   const auto count = input.numel();
//   const auto data_type = to_hccl_data_type(input);

//   auto stream = c10_npu::getCurrentNPUStream();
//   HCCLCHECK(HcclAllGather(
//       /*sendbuff=*/input.data_ptr(),
//       /*recvbuff=*/flattened_output.data_ptr(),
//       /*sendcount=*/count,
//       /*datatype=*/data_type,
//       /*comm=*/comm_,
//       /*stream=*/stream));
//   // copy the flattened output tensors to the outputs.
//   for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
//     outputs[i].copy_(flattened_output[i], /*non_blocking=*/true);
//   }
// }

void ProcessGroupImpl::allgather(const torch::Tensor& input,
                                 std::vector<torch::Tensor>& outputs) {
  CHECK(outputs.size() == world_size())
      << "outputs should have the same size as world_size";
  DCHECK(input.device() == device())
      << "input should be on the same device as the process group";

  torch::DeviceGuard device_guard(device());

  // 1. 展平输出
  torch::Tensor flattened_output = flatten_for_scatter_gather(outputs);

  const auto count = input.numel();
  const auto data_type = to_hccl_data_type(input);

  // 2. 当前计算 stream
  auto compute_stream = c10_npu::getCurrentNPUStream();

  // 3. 建立 compute -> comm 依赖
  c10_npu::NPUEvent ready;
  ready.record(compute_stream);
  ready.block(comm_stream_);

  // 4. allocator 记录通信 stream
  c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                             comm_stream_);
  c10_npu::NPUCachingAllocator::recordStream(
      flattened_output.storage().data_ptr(), comm_stream_);

  // 5. 发起 AllGather（异步，入 comm stream）
  HCCLCHECK(HcclAllGather(
      /*sendbuff=*/input.data_ptr(),
      /*recvbuff=*/flattened_output.data_ptr(),
      /*sendcount=*/count,
      /*datatype=*/data_type,
      /*comm=*/comm_,
      /*stream=*/comm_stream_.stream()));

  // 6. 通信完成事件
  // c10_npu::NPUEvent done;
  auto done = std::make_shared<c10_npu::NPUEvent>();
  done->record(comm_stream_);
  // if (out_done) {
  //   *out_done = std::move(done);
  // } else {
  done->block(compute_stream);
  comm_stream_.synchronize();
  // }
  // 7. compute 等待通信完成
  // done.block(compute_stream);

  // 8. 拆分回 outputs（此时数据已安全可用）
  for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
    outputs[i].copy_(flattened_output[i], /*non_blocking=*/true);
  }
}

// void ProcessGroupImpl::allreduce(torch::Tensor& input) {
//   DCHECK(input.device() == device())
//       << "input should be on the same device as the process group";

//   torch::DeviceGuard guard(device());

//   const auto count = input.numel();
//   const auto data_type = to_hccl_data_type(input);

//   auto compute_stream = c10_npu::getCurrentNPUStream();

//   // 1. compute -> comm
//   c10_npu::NPUEvent ready;
//   ready.record(compute_stream);
//   ready.block(comm_stream_);

//   // 2. allocator 记录 comm stream
//   c10_npu::NPUCachingAllocator::recordStream(
//       input.storage().data_ptr(), comm_stream_);

//   // 3. 发起 AllReduce（异步）
//   HCCLCHECK(HcclAllReduce(
//       input.data_ptr(),
//       input.data_ptr(),
//       count,
//       data_type,
//       HCCL_REDUCE_SUM,
//       comm_,
//       comm_stream_.stream()));

//   // 4. 只记录完成事件，不 block
//   last_comm_event_.record(comm_stream_);
// }

// void ProcessGroupImpl::allreduce(torch::Tensor& input) {
//   DCHECK(input.device() == device())
//       << "input should be on the same device as the process group";
//   torch::DeviceGuard device_guard(device());
//   const auto count = input.numel();
//   const auto data_type = to_hccl_data_type(input);
//   auto stream = c10_npu::getCurrentNPUStream();
//   HCCLCHECK(HcclAllReduce(
//       /*sendbuff=*/input.data_ptr(),
//       /*recvbuff=*/input.data_ptr(),
//       /*count=*/count,
//       /*datatype=*/data_type,
//       /*op=*/HCCL_REDUCE_SUM,
//       /*comm=*/comm_,
//       /*stream=*/stream));
// }

void ProcessGroupImpl::allreduce(torch::Tensor& input) {
  DCHECK(input.device() == device())
      << "input should be on the same device as the process group";

  torch::DeviceGuard device_guard(device());

  const auto count = input.numel();
  const auto data_type = to_hccl_data_type(input);

  auto compute_stream = c10_npu::getCurrentNPUStream();

  // 1. compute -> comm
  c10_npu::NPUEvent ready;
  ready.record(compute_stream);
  ready.block(comm_stream_);

  // 2. allocator 记录
  c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                             comm_stream_);

  // 3. 发起 AllReduce
  HCCLCHECK(HcclAllReduce(
      /*sendbuff=*/input.data_ptr(),
      /*recvbuff=*/input.data_ptr(),
      /*count=*/count,
      /*datatype=*/data_type,
      /*op=*/HCCL_REDUCE_SUM,
      /*comm=*/comm_,
      /*stream=*/comm_stream_.stream()));

  auto done = std::make_shared<c10_npu::NPUEvent>();
  done->record(comm_stream_);
  // if (out_done) {
  // *out_done = std::move(done);
  // } else {
  done->block(compute_stream);
  comm_stream_.synchronize();
  // }
}

void ProcessGroupImpl::flush_comm_to_current() {
#if defined(USE_NPU)
  auto cur = c10_npu::getCurrentNPUStream();
  c10_npu::NPUEvent fence;
  fence.record(comm_stream_);  // 通信流 -> 事件
  fence.block(cur);            // 事件 -> 当前计算流
#endif
}

}  // namespace xllm