#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class Executor final {
 public:
  Executor(CausalLM* model,
           const ModelArgs& args,
           const torch::Device& device,
           const runtime::Options& options);

  virtual ~Executor() = default;

  ForwardInput prepare_inputs(Batch& batch);

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& params);

 private:
  std::unique_ptr<ExecutorImpl> impl_;
};

}  // namespace xllm
