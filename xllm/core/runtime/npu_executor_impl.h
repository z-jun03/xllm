#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class NpuExecutorImpl : public ExecutorImpl {
 public:
  NpuExecutorImpl(CausalLM* model,
                  const ModelArgs& args,
                  const torch::Device& device,
                  const runtime::Options& options);

  ~NpuExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  torch::Tensor run(const torch::Tensor& tokens,
                    const torch::Tensor& positions,
                    std::vector<KVCache>& kv_caches,
                    const ModelInputParams& params) override;

 private:
  // not own
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
};

}  // namespace xllm
