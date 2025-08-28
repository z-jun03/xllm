/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>

#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state.h"
#include "core/framework/quant_args.h"

namespace xllm {

class Context {
 public:
  Context(const ParallelArgs& input_parallel_args)
      : parallel_args(input_parallel_args) {}

  const ModelArgs& get_model_args() const { return model_args; }
  void set_model_args(const ModelArgs& model_args) {
    this->model_args = model_args;
  }

  const QuantArgs& get_quant_args() const { return quant_args; }
  void set_quant_args(const QuantArgs& quant_args) {
    this->quant_args = quant_args;
  }

  const ParallelArgs& get_parallel_args() const { return parallel_args; }
  //   void set_paralle_args(const ParallelArgs& parallel_args) {
  //     this->parallel_args = parallel_args;
  //   }

  const torch::TensorOptions& get_tensor_options() const {
    return tensor_options;
  }
  void set_tensor_options(const torch::TensorOptions& tensor_options) {
    this->tensor_options = tensor_options;
  }

 private:
  ModelArgs model_args;
  QuantArgs quant_args;
  ParallelArgs parallel_args;
  torch::TensorOptions tensor_options;
};

}  // namespace xllm
