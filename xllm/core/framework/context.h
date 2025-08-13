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
