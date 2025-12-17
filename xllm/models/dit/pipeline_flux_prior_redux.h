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
#include "core/layers/pos_embedding.h"
#include "core/layers/rotary_embedding.h"
#include "pipeline_flux_base.h"
#include "processors/siglip_image_processor.h"
#include "siglip_vision_model.h"

namespace xllm {

class ReduxImageEncoderImpl : public torch::nn::Module {
 public:
  explicit ReduxImageEncoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    act_ = register_module("act", torch::nn::Functional(torch::silu));

    redux_up_ = register_module("redux_up",
                                DiTLinear(model_args.mm_hidden_size(),
                                          model_args.mm_intermediate_size() * 3,
                                          true));
    redux_down_ =
        register_module("redux_down",
                        DiTLinear(model_args.mm_intermediate_size() * 3,
                                  model_args.mm_intermediate_size(),
                                  true));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return redux_down_(act_(redux_up_(hidden_states)));
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      redux_up_->load_state_dict(state_dict->get_dict_with_prefix("redux_up."));
      redux_up_weight_loaded_ = true;
      redux_up_bias_loaded_ = true;
      redux_down_->load_state_dict(
          state_dict->get_dict_with_prefix("redux_down."));
      redux_down_weight_loaded_ = true;
      redux_down_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(redux_up_weight_loaded_)
        << "weight is not loaded for " << prefix + "redux_up.weight";
    CHECK(redux_up_bias_loaded_)
        << "weight is not loaded for " << prefix + "redux_up.bias";
    CHECK(redux_down_weight_loaded_)
        << "weight is not loaded for " << prefix + "redux_down.weight";
    CHECK(redux_down_bias_loaded_)
        << "weight is not loaded for " << prefix + "redux_down.bias";
  }

 private:
  DiTLinear redux_up_{nullptr};
  DiTLinear redux_down_{nullptr};

  torch::nn::Functional act_ = nullptr;
  bool redux_up_weight_loaded_ = false;
  bool redux_up_bias_loaded_ = false;
  bool redux_down_weight_loaded_ = false;
  bool redux_down_bias_loaded_ = false;
};
TORCH_MODULE(ReduxImageEncoder);

REGISTER_MODEL_ARGS(ReduxImageEncoder, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(mm_hidden_size, "redux_dim", 1152);
  LOAD_ARG_OR(mm_intermediate_size, "txt_in_features", 4096);
});

class FluxPriorReduxPipelineImpl : public FluxPipelineBaseImpl {
 public:
  FluxPriorReduxPipelineImpl(const DiTModelContext& context) {
    auto model_args = context.get_model_args("image_encoder");
    options_ = context.get_tensor_options();
    image_encoder_ =
        SiglipVisionModel(context.get_model_context("image_encoder"));
    // feature_extractor_ =
    // SiglipImageProcessor(context.get_model_context("feature_extractor"));
    image_embedder_ =
        ReduxImageEncoder(context.get_model_context("image_embedder"));
    feature_extractor_ = std::make_unique<SiglipImageProcessor>(
        context.get_model_context("feature_extractor"));
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "FluxPriorReduxPipeline loading model from"
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto image_encoder_loader = loader->take_component_loader("image_encoder");
    auto image_embedder_loader =
        loader->take_component_loader("image_embedder");
    image_encoder_->load_model(std::move(image_encoder_loader));
    image_encoder_->to(options_.device());
    image_embedder_->load_model(std::move(image_embedder_loader));
    image_embedder_->to(options_.device());
    // feature_extractor_->load_image_preprocessor_args(model_path+"/feature_extractor");
  }

  torch::Tensor encode_image(const torch::Tensor& image,
                             int64_t num_images_per_prompt) {
    // auto dtype = next(image_encoder_->parameters()).dtype();
    auto imgs = feature_extractor_->preprocess(image).to(
        options_);  //, /*height=*/std::nullopt, /*width=*/std::nullopt);
    LOG(INFO) << "imgs size: " << imgs.sizes();
    auto encoded = image_encoder_->forward(imgs);
    auto image_enc_hidden_states = encoded;  // assume returns last_hidden_state
    image_enc_hidden_states =
        image_enc_hidden_states.repeat_interleave(num_images_per_prompt, 0);
    return image_enc_hidden_states;
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;
    auto image = input.images.defined() ? std::make_optional(input.images)
                                        : std::nullopt;
    auto prompts = std::make_optional(input.prompts);
    auto prompts_2 = input.prompts_2.empty()
                         ? std::nullopt
                         : std::make_optional(input.prompts_2);

    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto pooled_prompt_embeds =
        input.pooled_prompt_embeds.defined()
            ? std::make_optional(input.pooled_prompt_embeds)
            : std::nullopt;
    auto prompt_embeds_scale = generation_params.prompt_embeds_scale;
    auto pooled_prompt_embeds_scale =
        generation_params.pooled_prompt_embeds_scale;
    // std::tuple<torch::Tensor, torch::Tensor>
    // std::tie(prompt_embeds, pooled_prompt_embeds)
    std::vector<torch::Tensor> output = forward_(image.value(),
                                                 prompts,
                                                 prompts_2,
                                                 prompt_embeds,
                                                 pooled_prompt_embeds,
                                                 generation_params.height,
                                                 generation_params.width,
                                                 prompt_embeds_scale,
                                                 pooled_prompt_embeds_scale);
    DiTForwardOutput out;
    // out.tensors = torch::chunk(output, output[0].size(0), 0);
    out.tensors = output;
    LOG(INFO) << "Output tensor chunks size: " << out.tensors.size();
    return out;
  }

  std::vector<torch::Tensor> forward_(
      torch::Tensor& image,
      std::optional<std::vector<std::string>>& prompt_opt,
      std::optional<std::vector<std::string>>& prompt_2_opt,
      std::optional<torch::Tensor>& prompt_embeds,
      std::optional<torch::Tensor>& pooled_prompt_embeds,
      int64_t height = 384,
      int64_t width = 384,
      float prompt_embeds_scale = 1.0f,
      float pooled_prompt_embeds_scale = 1.0f) {
    torch::NoGradGuard no_grad;
    int64_t batch_size = image.dim() == 4 ? image.size(0) : 1;
    std::vector<std::string> prompt = prompt_opt.has_value()
                                          ? prompt_opt.value()
                                          : std::vector<std::string>{};
    if (!prompt.empty() && prompt.size() == 1 && batch_size > 1) {
      prompt = std::vector<std::string>(batch_size, prompt[0]);
    }

    // auto device = options_.device();
    LOG(INFO) << "image size: " << image.sizes();
    auto image_latents = encode_image(image, 1);
    LOG(INFO) << "image_latents size: " << image_latents.sizes();
    auto image_embeds = image_embedder_->forward(image_latents);
    image_embeds = image_embeds.to(options_);

    prompt_embeds = prompt_embeds.value_or(torch::zeros(
        {batch_size, 512, 4096},
        options_));  // torch::TensorOptions().dtype(image_embeds.dtype).device(options_.device())));
    pooled_prompt_embeds = pooled_prompt_embeds.value_or(torch::zeros(
        {batch_size, 768},
        options_));  //,
                     // torch::TensorOptions().dtype(image_embeds.dtype).device(options_.device())));
    // prompt_embeds = torch::zeros({batch_size, 512, 4096},
    // torch::TensorOptions().dtype(image_embeds.dtype).device(options_.device()));
    // pooled_prompt_embeds = torch::zeros({batch_size,
    // 768},torch::TensorOptions().dtype(image_embeds.dtype).device(options_.device()));
    // encode_prompt(prompt,
    //               prompt_2,
    //               prompt_embeds,
    //               pooled_prompt_embeds,
    //               num_images_per_prompt,
    //               max_sequence_length);

    // if(!prompt_embeds.has_value()) {
    //   prompt_embeds = torch::zeros({batch_size, 512, 4096}, options_);
    // }
    // if(!pooled_prompt_embeds.has_value()) {
    //   pooled_prompt_embeds = torch::zeros({batch_size, 768}, options_);
    // }

    prompt_embeds = torch::cat({prompt_embeds.value(), image_embeds}, 1);

    // auto pet = torch::tensor(prompt_embeds_scale,
    // options_.device()).to(image_embeds.dtype()); auto ppet =
    // torch::tensor(pooled_prompt_embeds_scale,
    // options_.device()).to(image_embeds.dtype());

    auto pet = torch::full({batch_size},
                           prompt_embeds_scale,
                           options_);  //.dtype(image_embeds.dtype));
    auto ppet = torch::full({batch_size},
                            pooled_prompt_embeds_scale,
                            options_);  //, options_.dtype(image_embeds.dtype));

    prompt_embeds = prompt_embeds.value() * pet.view({-1, 1, 1});
    pooled_prompt_embeds = pooled_prompt_embeds.value() * ppet.view({-1, 1});

    prompt_embeds = torch::sum(prompt_embeds.value(), 0, /*keepdim=*/true);
    pooled_prompt_embeds =
        torch::sum(pooled_prompt_embeds.value(), 0, /*keepdim=*/true);

    // auto pe = prompt_embeds.value_or(torch::zeros({batch, 512, 4096}, opts));
    // auto ppe = pooled_prompt_embeds.value_or(torch::zeros({batch, 768},
    // opts)); torch::Tensor pe = prompt_embeds.value(); torch::Tensor ppe =
    // pooled_prompt_embeds.value();

    // pe = torch::sum(pe, 0, true);
    // ppe = torch::sum(ppe, 0, true);

    // prompt_embeds = pe;
    // pooled_prompt_embeds = ppe;

    LOG(INFO) << "finish redux model.";
    return {prompt_embeds.value(), pooled_prompt_embeds.value()};
  }

 private:
  SiglipVisionModel image_encoder_{nullptr};
  // SiglipImageProcessor feature_extractor_{nullptr};
  // std::unique_ptr<ImageProcessor> image_processor_;
  std::unique_ptr<SiglipImageProcessor> feature_extractor_;
  ReduxImageEncoder image_embedder_{nullptr};
};
TORCH_MODULE(FluxPriorReduxPipeline);

REGISTER_DIT_MODEL(fluxredux, FluxPriorReduxPipeline);
}  // namespace xllm
