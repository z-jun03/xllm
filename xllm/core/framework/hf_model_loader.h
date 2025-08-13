#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "model_loader.h"

namespace xllm {

class HFModelLoader : public ModelLoader {
 public:
  HFModelLoader(const std::string& model_weights_path);

  std::unique_ptr<Tokenizer> tokenizer() const override;

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override;

 private:
  bool load_args(const std::string& model_weights_path);
  bool load_model_args(const std::string& model_weights_path);
  bool load_quant_args(const std::string& model_weights_path);
  bool load_tokenizer_args(const std::string& model_weights_path);
  bool load_image_preprocessor_args(const std::string& model_weights_path);

  std::string model_weights_path_;

  // sorted model weights files
  std::vector<std::string> model_weights_files_;

  // models weights tensors
  std::vector<std::unique_ptr<StateDict>> state_dicts_;
};
}  // namespace xllm
