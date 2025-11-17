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

#include "npu_flux_single_encoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <iostream>
#include <map>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 18;

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {

};

static std::map<int, int> WEIGHT_SHARD = {

};

void NpuFluxSingleEncoderLayerImpl::param_from_args(
    atb_speed::flux::EncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank = parallel_args.n_heads() / params.worldSize;
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue =
      static_cast<int>(args.n_heads() / params.worldSize);
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.enableLogN = false;
}

NpuFluxSingleEncoderLayerImpl::NpuFluxSingleEncoderLayerImpl(
    const ModelContext& context)
    : NpuBaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  at_weight_tensors_.resize();
  atb_weight_tensors_.resize();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void NpuFluxSingleEncoderLayerImpl::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

int64_t NpuFluxSingleEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::flux::EncoderLayerParam& param);

}  // namespace layer

void NpuFluxSingleEncoderLayerImpl::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
  get_weights_col_packed_qkv();
}

int64_t NpuFluxSingleEncoderLayerImpl::init_layer() {
  name_ = "flux_single_encoder_layer";
  model_name_ = "flux_";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t NpuFluxSingleEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::flux::EncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::flux::EncoderLayer(param, &operation);
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  return atb::NO_ERROR;
}

}  // namespace xllm