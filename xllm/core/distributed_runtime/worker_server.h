#pragma once

#include <brpc/server.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <thread>

#include "common/macros.h"
#include "distributed_runtime/worker_service.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "runtime/executor.h"
#include "runtime/forward_params.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"
#include "worker.pb.h"

namespace xllm {

class WorkerServer {
 public:
  WorkerServer(int local_worker_idx,
               const std::string& master_node_addr,
               std::atomic<bool>& done,
               const ParallelArgs& parallel_args,
               const torch::Device& device,
               const runtime::Options& options,
               WorkerType worker_type);

  virtual ~WorkerServer();

 private:
  DISALLOW_COPY_AND_ASSIGN(WorkerServer);

  void create_server(const runtime::Options& options,
                     std::atomic<bool>& done,
                     const std::string& master_node_addr,
                     const torch::Device& device,
                     int world_sizse,
                     int global_rank,
                     int32_t dp_size,
                     int local_rank,
                     int32_t ep_size);

  bool sync_master_node(const std::string& master_node_addr,
                        proto::AddressInfo& addr_info,
                        proto::CommUniqueIdList& uids);

 private:
  std::unique_ptr<std::thread> worker_thread_;
};

}  // namespace xllm
