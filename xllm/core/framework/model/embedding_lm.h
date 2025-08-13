#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "causal_lm.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/parallel_state.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "model_args.h"
#include "model_input_params.h"

namespace xllm {

class EmbeddingLM : public CausalLM {
 public:
  ~EmbeddingLM() override = default;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) = 0;
};

template <typename Model>
class EmbeddingLMImpl : public EmbeddingLM {
 public:
  EmbeddingLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;

  torch::TensorOptions options_;
};

}  // namespace xllm
