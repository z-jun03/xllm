#pragma once

#include <torch/torch.h>

#include <string>

#include "core/framework/request/mm_data.h"

namespace xllm {

class InputProcessor {
 public:
  virtual ~InputProcessor() = default;

  virtual void process(std::string& prompt, const MMData& mm_data) = 0;
};

}  // namespace xllm
