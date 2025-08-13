#include "llm_head.h"

#include "atb_head_impl.h"

namespace xllm::hf {

std::shared_ptr<LlmHeadImpl> create_llm_head_layer(const Context& context) {
  return std::make_shared<AtbLmHeadImpl>(context);
}

LlmHead::LlmHead(const Context& context)
    : ModuleHolder(create_llm_head_layer(context)) {}

}  // namespace xllm::hf
