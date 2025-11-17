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

#include <torch/torch.h>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/linear_impl.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace F = torch::nn::functional;

class DiTLinearImpl : public torch::nn::Module {
 public:
  DiTLinearImpl(int64_t in, int64_t out, bool with_bias = true) {
    weight = register_parameter("weight", torch::empty({out, in}));
    if (with_bias) {
      bias = register_parameter("bias", torch::empty(out));
    } else {
      bias = register_parameter("bias", {}, false);
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return F::linear(x, weight, bias);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight, weight_is_loaded_);
    if (bias.defined()) {
      weight::load_weight(state_dict, "bias", bias, bias_is_loaded_);
    }
  }

  void to(torch::TensorOptions options) {
    weight.set_data(weight.to(options));
    if (bias.defined()) {
      bias.set_data(bias.to(options));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    if (bias.defined()) {
      CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

  torch::Tensor weight;
  torch::Tensor bias;

 private:
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
};

TORCH_MODULE(DiTLinear);

// 基类父类
class DiTColumnParallelLinearImpl : public torch::nn::Module {
 public:
  DiTColumnParallelLinearImpl(int64_t in_features,
                              int64_t out_features,
                              bool bias,
                              bool gather_output,
                              const ParallelArgs& parallel_args,
                              const torch::TensorOptions& options)
      : gather_output_(gather_output),
        parallel_args_(parallel_args),
        device_(options.device()) {
    rank_ = parallel_args_.rank_;
    world_size_ = parallel_args_.world_size_;
    CHECK(out_features % world_size_ == 0)
        << "out_features " << out_features << " not divisible by world_size "
        << world_size_;
    const int64_t out_features_per_partition = out_features / world_size_;
    // Note: torch.nn.functional.linear performs XA^T + b and as a result
    // we allocate the transpose.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features}, options),
        // torch::empty({in_features, out_features_per_partition},options),
        /*requires_grad=*/false);

    if (bias) {
      bias_ = register_parameter(
          "bias",
          torch::empty({out_features_per_partition}, options),
          /*requires_grad=*/false);
    }
  }

  torch::Tensor forward(torch::Tensor input) {
    input = input.to(device_);
    // auto bias = (bias_.defined() && rank_ == 0) ?
    // std::optional<at::Tensor>(bias_)
    //                                             : std::nullopt;
    torch::Tensor bias =
        (bias_.defined() && rank_ == 0) ? bias_ : torch::Tensor();
    // xllm::kernel::MatmulParams matmul_params;
    // matmul_params.a = input;
    // matmul_params.b = weight_;
    // matmul_params.bias = bias;

    // auto output = xllm::kernel::matmul(matmul_params);
    auto output = F::linear(input, weight_, bias);
    if (world_size_ > 1 && gather_output_) {
      output =
          xllm::parallel_state::gather(output, parallel_args_.process_group_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto rank = rank_;
    const auto world_size = world_size_;

    // load sharded weights on dim 0
    LOAD_SHARDED_WEIGHT(weight, 0);

    if (bias_.defined()) {
      // load sharded bias on dim 0
      LOAD_SHARDED_WEIGHT(bias, 0);
    }
  }

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes) {
    const auto rank = rank_;
    const auto world_size = world_size_;

    // load and merge the weights on dim 0
    LOAD_FUSED_WEIGHT(weight, 0);

    if (bias_.defined()) {
      // load and merge the bias on dim 0
      LOAD_FUSED_WEIGHT(bias, 0);
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(weight);
  DEFINE_FUSED_WEIGHT(bias);

  int rank_;
  int world_size_;
  // whether to gather the output
  bool gather_output_;
  at::Device device_;
  // parallel args
  ParallelArgs parallel_args_;
};
TORCH_MODULE(DiTColumnParallelLinear);

class DiTRowParallelLinearImpl : public torch::nn::Module {
 public:
  DiTRowParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           bool bias,
                           bool input_is_parallelized,
                           bool if_reduce_results,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
      : input_is_parallelized_(input_is_parallelized),
        if_reduce_results_(if_reduce_results),
        parallel_args_(parallel_args) {
    rank_ = parallel_args_.rank_;
    world_size_ = parallel_args_.world_size_;
    CHECK(in_features % world_size_ == 0)
        << "in_features " << in_features << " not divisible by world_size "
        << world_size_;
    const int64_t in_features_per_partition = in_features / world_size_;
    // Allocate the transpose since linear performs XA^T.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition}, options),
        // torch::empty({in_features_per_partition,out_features}, options),
        /*requires_grad=*/false);

    if (bias) {
      bias_ = register_parameter("bias",
                                 torch::empty({out_features}, options),
                                 /*requires_grad=*/false);
    }
  }

  torch::Tensor forward(torch::Tensor input) {
    // input = input.to(device_);
    // auto bias = (bias_.defined() && rank_ == 0) ?
    // std::optional<at::Tensor>(bias_)
    //                                             : std::nullopt;
    if (!input_is_parallelized_) {
      input =
          xllm::parallel_state::scatter(input, parallel_args_.process_group_);
    }

    torch::Tensor bias =
        (bias_.defined() && rank_ == 0) ? bias_ : torch::Tensor();
    // xllm::kernel::MatmulParams matmul_params;
    // matmul_params.a = input;
    // matmul_params.b = weight_;
    // matmul_params.bias = bias;

    // auto output = xllm::kernel::matmul(matmul_params);
    auto output = F::linear(input, weight_, bias);
    if (if_reduce_results_ && world_size_ > 1) {
      output =
          xllm::parallel_state::reduce(output, parallel_args_.process_group_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto rank = rank_;
    const auto world_size = world_size_;

    // load sharded weights on dim 1
    LOAD_SHARDED_WEIGHT(weight, 1);

    if (bias_.defined()) {
      LOAD_WEIGHT(bias);
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // whether to reduce the results
  bool if_reduce_results_;

  // parallel args
  ParallelArgs parallel_args_;

  int rank_;
  int world_size_;
};
TORCH_MODULE(DiTRowParallelLinear);

class DiTQKVParallelLinearImpl : public torch::nn::Module {
 public:
  DiTQKVParallelLinearImpl(int64_t hidden_size,
                           int64_t num_heads,
                           int64_t num_kv_heads,
                           int64_t head_size,
                           int64_t num_kv_head_replicas,
                           bool bias,
                           bool gather_output,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
      : hidden_size_(hidden_size),
        num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        head_size_(head_size),
        num_kv_head_replicas_(num_kv_head_replicas),
        gather_output_(gather_output),
        parallel_args_(parallel_args),
        options_(options),
        device_(options.device()) {
    const int32_t QKV_CNT = 3;
    // rank_ = parallel_args_.tp_group_->rank();
    // world_size_ = parallel_args_.tp_group_->world_size();
    // rank = parallel_args_->rank();
    // world_size_ = parallel_args_->world_size();
    rank_ = parallel_args_.rank_;
    world_size_ = parallel_args_.world_size_;
    const int64_t out_features_per_partition =
        (num_heads + 2 * num_kv_heads) * head_size;
    // Note: torch.nn.functional.linear performs XA^T + b and as a result
    // we allocate the transpose.
    qkv_weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size}, options),
        // torch::empty({hidden_size, out_features_per_partition},options),
        /*requires_grad=*/false);
    qkv_weight_list_.resize(QKV_CNT);

    if (bias) {
      qkv_bias_ = register_parameter(
          "bias",
          torch::empty({out_features_per_partition}, options),
          /*requires_grad=*/false);
      qkv_bias_list_.resize(QKV_CNT);
    }
  }

  torch::Tensor forward(torch::Tensor input) {
    input = input.to(device_);
    torch::Tensor bias =
        (qkv_bias_.defined() && rank_ == 0) ? qkv_bias_ : torch::Tensor();

    // auto bias = (qkv_bias_.defined() && rank_ == 0)
    //                 ? std::optional<at::Tensor>(qkv_bias_)
    //                 : std::nullopt;
    // xllm::kernel::MatmulParams matmul_params;
    // matmul_params.a = input;
    // matmul_params.b = qkv_weight_;
    // matmul_params.bias = bias;

    // auto output = xllm::kernel::matmul(matmul_params);
    auto output = F::linear(input, qkv_weight_, bias);
    if (world_size_ > 1 && gather_output_) {
      output =
          xllm::parallel_state::gather(output, parallel_args_.process_group_);
    }
    return output;
  }

  bool load_qkv_weight(const StateDict& state_dict, int32_t index) {
    if (qkv_weight_list_[index].defined() || state_dict.size() == 0) {
      return false;
    }
    DEFINE_WEIGHT(weight);
    int64_t out_feature = num_heads_ * head_size_;
    int32_t rank = rank_;
    int world_size = world_size_;
    if (index > 0) {
      rank = rank_ / num_kv_head_replicas_;
      world_size = world_size_ / num_kv_head_replicas_;
      out_feature = num_kv_heads_ * head_size_;
    }
    weight_ = torch::empty({out_feature, hidden_size_}, options_);
    LOAD_SHARDED_WEIGHT(weight, 0);
    if (weight_is_loaded_) {
      qkv_weight_list_[index] = weight_.clone();
    }
    return weight_is_loaded_;
  }

  void load_state_dict(const StateDict& state_dict) {
    std::vector<std::string> prefixes = {"q_proj.", "k_proj.", "v_proj."};
    if (!qkv_weight_is_loaded_) {
      bool all_loaded = true;
      for (size_t i = 0; i < prefixes.size(); ++i) {
        all_loaded =
            all_loaded &&
            load_qkv_weight(state_dict.get_dict_with_prefix(prefixes[i]), i);
      }
      if (all_loaded) {
        const auto merged_weight = torch::cat(qkv_weight_list_, /*dim=*/0);
        CHECK_EQ(qkv_weight_.sizes(), merged_weight.sizes())
            << "weight size mismatch";
        qkv_weight_.copy_(merged_weight);
        // release the memory for weight_list
        qkv_weight_list_.clear();
        qkv_weight_is_loaded_ = true;
      }
    }
  }

  // to_q shape:[in_dim , q_dim]
  // to_v shape: [in_dim , kv_dim]
  // to_qkv shape: [in_dim*3, q_dim]
  void load_state_dict(const StateDict& state_dict,
                       std::vector<std::string> prefixes) {
    // std::vector<std::string> prefixes = {"q_proj.", "k_proj.", "v_proj."};
    if (!qkv_weight_is_loaded_) {
      bool all_loaded = true;
      for (size_t i = 0; i < prefixes.size(); ++i) {
        all_loaded =
            all_loaded &&
            load_qkv_weight(state_dict.get_dict_with_prefix(prefixes[i]), 0);
      }
      if (all_loaded) {
        const auto merged_weight = torch::cat(qkv_weight_list_, /*dim=*/0);
        CHECK_EQ(qkv_weight_.sizes(), merged_weight.sizes())
            << "weight size mismatch";
        qkv_weight_.copy_(merged_weight);
        // release the memory for weight_list
        qkv_weight_list_.clear();
        qkv_weight_is_loaded_ = true;
      }
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(qkv_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!qkv_bias_.defined() || qkv_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const {
    stream << name() << " " << weight().sizes() << " " << weight().device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return qkv_weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  DEFINE_FUSED_WEIGHT(qkv_weight);
  DEFINE_FUSED_WEIGHT(qkv_bias);

  int rank_;
  int world_size_;
  int64_t hidden_size_;
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t head_size_;
  int64_t num_kv_head_replicas_;
  // whether to gather the output
  bool gather_output_;
  at::Device device_;
  // parallel args
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
};
TORCH_MODULE(DiTQKVParallelLinear);

// class DiTColumnParallelLinear : public
// torch::nn::ModuleHolder<layer::ColumnParallelLinearImpl> {
//  public:
//   using
//   torch::nn::ModuleHolder<layer::ColumnParallelLinearImpl>::ModuleHolder;
//   using Impl __attribute__((__unused__)) = layer::ColumnParallelLinearImpl;

//   DiTColumnParallelLinear(int64_t in_features,
//                        int64_t out_features,
//                        bool bias,
//                        bool gather_output,
//                        const ParallelArgs& parallel_args,
//                        const torch::TensorOptions& options)
//       :
//       ModuleHolder(std::make_shared<layer::ColumnParallelLinearImpl>(in_features,
//                                                                 out_features,
//                                                                 bias,
//                                                                 gather_output,
//                                                                 parallel_args,
//                                                                 options)) {}
// };

// // TORCH_MODULE(DiTColumnParallelLinear);

// class DiTQKVParallelLinear
//     : public torch::nn::ModuleHolder<layer::QKVParallelLinearImpl> {
//  public:
//   using torch::nn::ModuleHolder<layer::QKVParallelLinearImpl>::ModuleHolder;
//   using Impl __attribute__((__unused__)) = layer::QKVParallelLinearImpl;

//   DiTQKVParallelLinear(int64_t hidden_size,
//                     int64_t num_heads,
//                     int64_t num_kv_heads,
//                     int64_t head_size,
//                     int64_t num_kv_head_replicas,
//                     bool bias,
//                     bool gather_output,
//                     const ParallelArgs& parallel_args,
//                     const torch::TensorOptions& options)
//       : ModuleHolder(
//             std::make_shared<layer::QKVParallelLinearImpl>(hidden_size,
//                                                     num_heads,
//                                                     num_kv_heads,
//                                                     head_size,
//                                                     num_kv_head_replicas,
//                                                     bias,
//                                                     gather_output,
//                                                     parallel_args,
//                                                     options)) {}
// };

// TORCH_MODULE(DiTQKVParallelLinear);

}  // namespace xllm
