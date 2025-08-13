#pragma once

#include "fast_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "tiktoken_tokenizer.h"
#include "tokenizer_args.h"

namespace xllm {

class TokenizerFactory {
 public:
  static std::unique_ptr<Tokenizer> create_tokenizer(
      const std::string& model_weights_path,
      TokenizerArgs tokenizer_args);
};

}  // namespace xllm
