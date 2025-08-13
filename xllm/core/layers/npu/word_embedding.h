#pragma once

#include "atb/atb_infer.h"
#include "buffer/atb_workspace.h"
#include "framework/context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm::hf {

class AtbEmbeddingImpl : public torch::nn::Module {
 public:
  ~AtbEmbeddingImpl() {};
  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string weight_str) const = 0;
  virtual void merge_loaded_weights() = 0;

  virtual torch::Tensor forward(const torch::Tensor& x,
                                atb::Context* context,
                                AtbWorkspace& workspace,
                                int nodeId) = 0;
};

class AtbWordEmbedding : public torch::nn::ModuleHolder<AtbEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<AtbEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = AtbEmbeddingImpl;

  AtbWordEmbedding(const Context& context);
};

std::shared_ptr<AtbEmbeddingImpl> create_word_embedding_layer(
    const Context& context);

}  // namespace xllm::hf
