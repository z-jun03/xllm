#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "framework/model/embedding_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class EmbedWorkerImpl : public WorkerImpl {
 public:
  EmbedWorkerImpl(const ParallelArgs& parallel_args,
                  const torch::Device& device,
                  const runtime::Options& options);

  ~EmbedWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(torch::ScalarType dtype,
                  const ModelArgs& args,
                  const QuantArgs& quant_args) override;

  std::optional<ForwardOutput> step(const ForwardInput& inputs) override;
};

}  // namespace xllm
