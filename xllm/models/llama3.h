#pragma once

#include "llama.h"

namespace xllm::hf {
// register the causal model
REGISTER_CAUSAL_MODEL(llama3, LlamaForCausalLM);
}  // namespace xllm::hf