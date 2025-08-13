#pragma once

#include "atb/atb_infer.h"
#include "buffer/atb_workspace.h"
#include "framework/context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm::hf {

class LlmHeadImpl : public torch::nn::Module {
 public:
  ~LlmHeadImpl() override = default;

  virtual void load_state_dict(const StateDict& state_dict) = 0;

  virtual void verify_loaded_weights(const std::string weight_str) const = 0;

  virtual void merge_loaded_weights() = 0;

  virtual torch::Tensor forward(const torch::Tensor& hidden_states,
                                const torch::Tensor& seleted_idxes,
                                atb::Context* context,
                                AtbWorkspace& workspace,
                                int nodeId) = 0;
};

class LlmHead : public torch::nn::ModuleHolder<LlmHeadImpl> {
 public:
  using torch::nn::ModuleHolder<LlmHeadImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = LlmHeadImpl;

  LlmHead(const Context& context);
};

std::shared_ptr<LlmHeadImpl> create_llm_head_layer(const Context& context);

}  // namespace xllm::hf
