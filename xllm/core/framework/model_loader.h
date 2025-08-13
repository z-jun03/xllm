#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {

class ModelLoader {
 public:
  enum ModelType : uint8_t { HF_MODEL_TYPE = 0, INVALID = 1 };

  virtual ~ModelLoader() = default;

  virtual const ModelArgs& model_args() const { return args_; }
  virtual const QuantArgs& quant_args() const { return quant_args_; }
  virtual const TokenizerArgs& tokenizer_args() const {
    return tokenizer_args_;
  }
  virtual std::unique_ptr<Tokenizer> tokenizer() const = 0;
  virtual std::vector<std::unique_ptr<StateDict>>& get_state_dicts() = 0;

 protected:
  // model args
  ModelArgs args_;
  // quantization args
  QuantArgs quant_args_;
  // tokenizer args
  TokenizerArgs tokenizer_args_;

 public:
  // create a model loader from the given path
  static std::unique_ptr<ModelLoader> create(
      const std::string& model_weights_path);
};

}  // namespace xllm
