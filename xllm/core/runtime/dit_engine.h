/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gflags/gflags.h>

#include <memory>

#include "common/macros.h"
#include "distributed_runtime/dist_manager.h"
#include "dit_forward_params.h"
#include "dit_worker.h"
#include "framework/batch/dit_batch.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "worker_client.h"

namespace xllm {

class DiTEngine {
 public:
  // DiTEngine(const runtime::Options& options);

  DiTEngine(const runtime::Options& options,
            std::shared_ptr<DistManager> dist_manager = nullptr);

  ~DiTEngine() = default;

  DiTForwardOutput step(std::vector<DiTBatch>& batch);

  const runtime::Options& options() const { return options_; }

  bool init();

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const;

  void process_group_test();

  std::vector<DiTForwardInput> prepare_inputs(std::vector<DiTBatch>& batches);

  void setup_workers(const runtime::Options& options);

 private:
  bool init_model();
  // options
  runtime::Options options_;

  // dtype
  torch::ScalarType dtype_;

  // quantization args
  QuantArgs quant_args_;

  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // // a list of workers, with each worker handling a partial of model
  // std::vector<std::unique_ptr<DiTWorker>> workers_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // common frequently used args
  uint32_t dp_size_;
  uint32_t worker_clients_num_;
  uint32_t dp_local_tp_size_;

  bool layer_forward_interrupted_ = false;

  std::shared_ptr<DistManager> dist_manager_ = nullptr;

  std::unique_ptr<ThreadPool> threadpool_ = nullptr;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<DiTWorkerImpl>> workers_;
};

}  // namespace xllm

// #pragma once

// #include <gflags/gflags.h>

// #include <memory>

// #include "engine.h"
// #include "common/macros.h"
// #include "dit_worker.h"
// #include "framework/batch/dit_batch.h"
// #include "framework/parallel_state/process_group.h"
// #include "framework/quant_args.h"
// #include "core/distributed_runtime/dist_manager.h"
// #include "util/threadpool.h"
// #include "worker.h"
// #include "worker_client.h"

// namespace xllm {

// class DiTEngine : public Engine{
//  public:
//   // DiTEngine(const runtime::Options& options);

//   DiTEngine(const runtime::Options& options,
//     std::shared_ptr<DistManager> dist_manager = nullptr);

//   ~DiTEngine() = default;

//   DiTForwardOutput step(std::vector<DiTBatch>& batch) override;

//   const runtime::Options& options() const { return options_; }

//   bool init() override;

//   // return the active activation memory
//   std::vector<int64_t> get_active_activation_memory() const override;

//  private:
//   void setup_workers(const runtime::Options& options);
//   void process_group_test();

//  private:
//   bool init_model();
//   // options
//   runtime::Options options_;

//   // dtype
//   torch::ScalarType dtype_;

//   // quantization args
//   QuantArgs quant_args_;

//   // a list of process groups, with each process group handling a single
//   device std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

//   // // a list of workers, with each worker handling a partial of model
//   // std::vector<std::unique_ptr<DiTWorker>> workers_;

//   // a list of workers, with each worker handling a partial of model
//   std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

//   // common frequently used args
//   uint32_t dp_size_;
//   uint32_t worker_clients_num_;
//   uint32_t dp_local_tp_size_;

//   bool layer_forward_interrupted_ = false;

//   std::shared_ptr<DistManager> dist_manager_ = nullptr;

//   std::unique_ptr<ThreadPool> threadpool_ = nullptr;
// };

// }  // namespace xllm