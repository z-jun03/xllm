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

#include "dit_worker.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include "pytorch/adapter/utils/utils.h"
#endif
#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/framework/dit_model_loader.h"
#include "framework/dit_cache/dit_cache.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {
std::vector<int64_t> tensor_to_vector(const torch::Tensor& t) {
  TORCH_CHECK(t.dim() == 1, "tensor_to_vector expects 1-D tensor");
  std::vector<int64_t> out;
  out.reserve(t.size(0));
  auto cpu = t.to(torch::kCPU);
  int64_t n = cpu.size(0);
  for (int64_t i = 0; i < n; ++i) {
    out.push_back(cpu[i].item<int64_t>());
  }
  return out;
}
}  // namespace

DiTWorker::DiTWorker(const ParallelArgs& parallel_args,
                     const torch::Device& device,
                     const runtime::Options& options)
    : device_(device), options_(options), parallel_args_(parallel_args) {}

bool DiTWorker::init_model(const std::string& model_weights_path) {
  CHECK(dit_model_ == nullptr) << "Model is already initialized.";
  int currentDevId = device_.index();
#if defined(USE_NPU)
  int ret = aclrtSetDevice(currentDevId);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id:" << currentDevId
               << " failed, ret:" << ret;
  }
#elif defined(USE_MLU)
// TODO(mlu): implement mlu set device
#endif
  LOG(INFO) << "Loading DiT model weights from: " << model_weights_path;
  // << " device: " << device_;
  auto loader = std::make_unique<DiTModelLoader>(model_weights_path);
  dtype_ = util::parse_dtype(loader->get_torch_dtype(), device_);

  auto tensor_options = torch::dtype(dtype_).device(device_);
  rank_ = parallel_args_.rank_;
  world_size_ = parallel_args_.world_size_;
  auto process_group = parallel_args_.process_group_;
  auto devices = options_.devices();

  init_model_parallel(rank_, world_size_, devices);

  context_ = DiTModelContext(parallel_args_,
                             std::move(loader->get_model_args()),
                             std::move(loader->get_quant_args()),
                             tensor_options,
                             options_.model_id());

  dit_model_ = create_dit_model(context_);
  CHECK(dit_model_ != nullptr) << "Failed to create model.";
  dit_model_->load_model(std::move(loader));

  dit_model_executor_ =
      std::make_unique<DiTExecutor>(dit_model_.get(), options_);

  DiTCacheConfig cache_config_;

  // TODO: Optimize ditcache configuration initialization.

  cache_config_.selected_policy = PolicyType::TaylorSeer;
  // cache_config_.selected_policy = PolicyType::None;
  cache_config_.taylorseer.n_derivatives = 3;
  cache_config_.taylorseer.skip_interval_steps = 3;
  cache_config_.taylorseer.num_inference_steps = 25;
  cache_config_.taylorseer.warmup_steps = 0;

  bool success = DiTCache::get_instance().init(cache_config_);
  CHECK(success) << "DiTCache init failed";

  return true;
}

bool DiTWorker::init_model_parallel(int rank,
                                    int world_size,
                                    std::vector<torch::Device> devices) {
  // const std::vector<std::unique_ptr<ProcessGroup>>& process_groups) {
  // 读取并行度参数（假设是全局 FLAGS）
  int64_t data_parallel_size = FLAGS_dp_size;
  int64_t tensor_parallel_size = FLAGS_tp_size;
  int64_t sequence_parallel_size = FLAGS_sp_size;
  std::vector<std::vector<int64_t>> tp_group_ranks_;
  std::vector<std::vector<int64_t>> sp_group_ranks_;

  CHECK(world_size ==
        data_parallel_size * tensor_parallel_size * sequence_parallel_size)
      << "world_size mismatch: expected dp*tp*sp == world_size";

  // all_ranks 形状: (num_group_sets, data_parallel_size,
  // sequence_parallel_size, tensor_parallel_size)
  torch::Tensor all_ranks = torch::arange(
      (int64_t)world_size, torch::TensorOptions().dtype(torch::kInt64));
  all_ranks = all_ranks.reshape(
      {-1, data_parallel_size, sequence_parallel_size, tensor_parallel_size});

  // -------- tensor parallel groups: 把 tensor_parallel_size
  // 这维作为组内rank维度，展开 groups -------- tp_group_ranks_.clear();
  if (tensor_parallel_size > 1) {
    // 将 tensor 的维度重新排列为 (num_groups, tensor_parallel_size) 来得到每个
    // tp group 的 ranks 方法一：先把 all_ranks 视作 (-1, tensor_parallel_size)
    // ，然后 unbind 得到每个 group 的 tensor
    torch::Tensor tp_view = all_ranks.view({-1, tensor_parallel_size});
    auto tp_groups = tp_view.unbind(
        0);  // vector<tensor>, 每个 tensor 长度为 tensor_parallel_size
    for (const auto& g : tp_groups) {
      tp_group_ranks_.push_back(tensor_to_vector(g));
    }

    // 如果需要把这些 rank 列表转换为真正的 ProcessGroup，请在下面调用
    // init_model_parallel_group() 例如：
    tp_process_groups_ =
        init_model_parallel_group(tp_group_ranks_, rank, devices);
    // 但该函数依赖工程中 ProcessGroup 的具体接口（见下方 TODO）
  }

  // -------- sequence parallel groups: 把 sequence_parallel_size 作组维度
  // -------- sp_group_ranks_.clear();
  if (sequence_parallel_size > 1) {
    // 先把 all_ranks 视作 (-1, sequence_parallel_size, tensor_parallel_size)?
    // 我们只需要按 sequence_parallel_size 划分 我们可以先 permute 使得
    // sequence_parallel_size 在最后一维，然后 view 以便 unbind 更直接的做法是
    // reshape 为 (-1, sequence_parallel_size)
    // 但需保证维度对齐：我们希望得到每个 sequence-parallel group 的 ranks
    // 列表（长度 = sequence_parallel_size） 为简单可靠，我们先把 all_ranks
    // reshape 为 (-1, data_parallel_size * sequence_parallel_size *
    // tensor_parallel_size) 然后用更明确的方法：遍历 data-parallel 与
    // tensor-parallel 的组合，抽取对应的 sequence 分量
    int64_t num_group_sets = all_ranks.size(0);  // first dim
    // all_ranks shape is (num_group_sets, data_dp, sp, tp)
    for (int64_t s = 0; s < num_group_sets; ++s) {
      // 取出第 s 个块: shape (data_dp, sp, tp)
      torch::Tensor block = all_ranks[s];  // shape (data_dp, sp, tp)
      // 对于 block 中每个 data_parallel 下标 i 和 tensor_parallel 下标
      // k，sequence 这一维上会有 sequence_parallel_size 个 rank
      auto block_sizes = block.sizes();  // {data_dp, sp, tp}
      int64_t data_dp = block_sizes[0];
      int64_t sp_sz = block_sizes[1];
      int64_t tp_sz = block_sizes[2];
      for (int64_t i = 0; i < data_dp; ++i) {
        for (int64_t k = 0; k < tp_sz; ++k) {
          // 取出 sequence 轴上的整个向量: block[i, :, k] -> 长度 sp_sz
          torch::Tensor seq_vec = block.index({i, torch::indexing::Slice(), k});
          sp_group_ranks_.push_back(tensor_to_vector(seq_vec));
        }
      }
    }

    // 同样，如果需要创建 ProcessGroup：
    tp_process_groups_ =
        init_model_parallel_group(sp_group_ranks_, rank, devices);
  }

  // 打印调试信息
  LOG(INFO) << "init_model_parallel: rank=" << rank
            << " world_size=" << world_size;
  LOG(INFO) << "tp groups count: " << tp_group_ranks_.size();
  LOG(INFO) << "sp groups count: " << sp_group_ranks_.size();

  // 至此，我们已经把 tp_group_ranks_ 与 sp_group_ranks_
  // 计算好了并存储在成员变量中。 若你需要返回/保存真正的 ProcessGroup
  // 对象，请在下面调用 init_model_parallel_group() 并实现 TODO 中的部分。
  return true;
}

std::vector<std::unique_ptr<ProcessGroup>> DiTWorker::init_model_parallel_group(
    const std::vector<std::vector<int64_t>>& group_ranks,
    int local_rank,
    std::vector<torch::Device> devices) {
  // const std::vector<std::unique_ptr<ProcessGroup>>& existing_process_groups)
  // {

  std::vector<std::unique_ptr<ProcessGroup>> out_groups;
  std::vector<torch::Device> process_devices;
  std::vector<int64_t> group_rank;
  bool found = false;

  for (const auto& ranks : group_ranks) {
    for (const auto& rank : ranks) {
      if (local_rank == rank) {
        group_rank = ranks;
        found = true;
      }
    }

    // 如果 existing_process_groups 中每个 ProcessGroup 暴露 ranks()
    // 接口，可以尝试匹配： for (const auto& pg_ptr : existing_process_groups) {
    //   if (!pg_ptr) continue;
    // }
  }
  if (found) {
    for (const auto& rank : group_rank) {
      // out_groups.push_back(existing_process_groups[rank]);
      process_devices.push_back(devices[rank]);
    }
    out_groups = parallel_state::create_npu_process_groups(process_devices);
  }
  return out_groups;
}

folly::SemiFuture<bool> DiTWorker::init_model_async(
    const std::string& model_weights_path) {
  LOG(INFO) << "init model async";
  auto sp = std::make_shared<folly::Promise<bool>>();
  auto fut = sp->getSemiFuture();
  threadpool_.schedule([this, model_weights_path, sp]() mutable {
    bool status = this->init_model(model_weights_path);
    sp->setValue(status);
  });
  return fut;
}

std::optional<DiTForwardOutput> DiTWorker::step(const DiTForwardInput& inputs) {
  device_.set_device();
  Timer timer;

  auto output = dit_model_executor_->forward(inputs.to(device_, dtype_));

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  return output;
}

folly::SemiFuture<std::optional<DiTForwardOutput>> DiTWorker::step_async(
    const DiTForwardInput& inputs) {
  auto sp = std::make_shared<folly::Promise<std::optional<DiTForwardOutput>>>();
  auto fut = sp->getSemiFuture();
  threadpool_.schedule([this, inputs, sp]() mutable {
    auto output = this->step(inputs);
    sp->setValue(output);
  });
  LOG(INFO) << "worker step end";
  return fut;
}

void DiTWorker::process_group_test() {
#if defined(USE_NPU)
  c10_npu::SetDevice(device_.index());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu process group test
#endif
  // create random tensors
  const auto options = torch::dtype(torch::kHalf).device(device_);
  torch::Tensor tensor = torch::randn({10, 10}, options);
  // call allreduce
  // parallel_state::reduce(tensor, context_.get_parallel_args());
  // call allgather
  // parallel_state::gather(tensor, context_.get_parallel_args());
}

folly::SemiFuture<folly::Unit> DiTWorker::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    this->process_group_test();
    promise.setValue();
  });
  return future;
}

// init_model_parallel()

// folly::SemiFuture<folly::Unit> DiTWorker::process_group_test_async() {
//   folly::Promise<folly::Unit> promise;
//   auto future = promise.getSemiFuture();
//   threadpool_.schedule([this, promise = std::move(promise)]() mutable {
//     this->process_group_test_async();
//     promise.setValue();
//   });
//   return future;
// }

// prepare input for execution
DiTForwardInput DiTWorker::prepare_inputs(DiTBatch& batch) {
  return dit_model_executor_->prepare_inputs(batch);
}

int64_t DiTWorker::get_active_activation_memory() {
  return DeviceMonitor::get_instance()
      .get_device_stats(device_.index())
      .active_activation_memory;
}

}  // namespace xllm

// 根据parallel_size传入
// bool init_model_parallel(int group_rank, int rank,
// std::vector<std::unique_ptr<ProcessGroup>> process_groups_) {
//   auto devices = options.devices();
//   const int32_t world_size = static_cast<int32_t>(devices.size());
//   int64_t data_parallel_size = FLAGS_dp_size;
//   int64_t tensor_parallel_size = FLAGS_tp_size;
//   int64_t sequence_parallel_size = FLAGS_sp_size;
//   CHECK(world_size =
//   data_parallel_size*tensor_parallel_size*sequence_parallel_size);

//   all_ranks = torch.arange(world_size).reshape(
//         -1, data_parallel_size, sequence_parallel_size,
//         tensor_parallel_size);
//   if (tensor_parallel_size > 1) {
//     auto group_ranks = all_ranks.view(-1,
//     tensor_model_parallel_size).unbind(0); group_ranks = [x.tolist() for x in
//     group_ranks]; tp_process_groups_ = init_model_parallel_group(group_ranks,
//     process_groups_);
//   }

//   if (sequence_parallel_size > 1) {
//     auto group_ranks = all_rank.view(-1, sequence_parallel_size).unbind(0);

//     sp_process_groups_ = init_model_parallel_group(group_ranks,
//     process_groups_);
//   }
//   return true;
// }

// // std::vector<> init_model_parallel_group
// std::vector<std::unique_ptr<ProcessGroup>> init_model_parallel_group(
//     int64_t group_ranks, int local_rank
//     ,std::vector<std::unique_ptr<ProcessGroup>>* process_groups_) {
//   torch::Tensor all_ranks = torch::arange(devices.size()).reshape(-1,
//   data_parallel_size, sequence_parallel_size, tensor_parallel_size);
//   torch::Tensor group_ranks = all_ranks.view(-1,
//   tensor_parallel_size).unbind(0);

//   std::vector<std::unique_ptr<ProcessGroup>> process_group;
//   for(auto group_rank : group_ranks) {
//     if (local_rank in group_rank) {
//       process_group = process_group[local_rank];
//     }
//   }
//   return process_group;
// }