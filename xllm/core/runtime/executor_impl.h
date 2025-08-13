#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "options.h"

namespace xllm {

class ExecutorImpl {
 public:
  virtual ~ExecutorImpl() = default;

  virtual ForwardInput prepare_inputs(Batch& batch) = 0;

  virtual torch::Tensor run(const torch::Tensor& tokens,
                            const torch::Tensor& positions,
                            std::vector<KVCache>& kv_caches,
                            const ModelInputParams& params) = 0;
};

}  // namespace xllm
