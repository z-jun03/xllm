
// /* Copyright 2025 The xLLM Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://github.com/jd-opensource/xllm/blob/main/LICENSE

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

// #pragma once

// #include <atb/atb_infer.h>
// #include <c10/core/ScalarType.h>
// #include <torch/torch.h>

// #include <regex>
// #include <unordered_map>

// #include "core/framework/dit_model_loader.h"
// #include "core/framework/kv_cache/kv_cache.h"
// #include "core/framework/model/model_input_params.h"
// #include "core/framework/model_context.h"
// #include "core/layers/siglip_encoder_layer.h"
// #include "dit_linear.h"
// #include "models/model_registry.h"
// #include "processors/clip_image_processor.h"
// #include "processors/input_processor.h"
// #include "processors/pywarpper_image_processor.h"
// #include "xllm_kernels/core/include/atb_speed/log.h"

// namespace xllm {

// class SiglipImageProcessorImpl : public InputProcessor {
//  public:
//     SiglipImageProcessorImpl(ModelContext context,
//                                  bool do_resize = true,
//                                  bool do_rescale = true,
//                                  bool do_normalize = true,
//                                  float rescale_factor = 1.0 / 255.0,
//                                  int64_t latent_channels = 4) {
//     const auto& model_args = context.get_model_args();
//     options_ = context.get_tensor_options();
//     rescale_factor_ = rescale_factor;
//     latent_channels_ = latent_channels;
//     do_resize_ = do_resize;
//     do_normalize_ = do_normalize;
//   }

//   torch::Tensor preprocess(const torch::Tensor& images,
//       std::optional<int64_t> height = std::nullopt,
//       std::optional<int64_t> width = std::nullopt) {
//   int batch_size = images.size(0);
//   std::vector<torch::Tensor> processed_images;

//   auto [target_h, target_w] =
//         get_default_height_width(processed, height, width);
//   for (int i = 0; i < batch_size; ++i) {
//     torch::Tensor image = images[i];

//     if (do_resize_) {
//       image = resize(processed, target_h, target_w);
//     }

//     if (do_rescale_) {
//       image = rescale(image, rescale_factor_);
//     }

//     if (do_normalize_) {
//       image = normalize(image, image_mean_, image_std_);
//     }

//     processed_images.push_back(image);
//   }
//   return torch::stack(processed_images);
//  }

//  private:
//   std::pair<int64_t, int64_t> get_default_height_width(
//       const torch::Tensor& image,
//       std::optional<int64_t> height = std::nullopt,
//       std::optional<int64_t> width = std::nullopt) const {
//     int64_t h, w;
//     if (image.dim() == 3) {
//       h = image.size(1);
//       w = image.size(2);
//     } else if (image.dim() == 4) {
//       h = image.size(2);
//       w = image.size(3);
//     } else {
//       LOG(FATAL) << "Unsupported image dimension: " << image.dim();
//     }

//     int64_t target_h = height.value_or(h);
//     int64_t target_w = width.value_or(w);
//     return adjust_dimensions(target_h, target_w);
//   }

//   torch::Tensor resize(const torch::Tensor& image,
//                        int64_t target_height,
//                        int64_t target_width) const {
//     return torch::nn::functional::interpolate(
//         image,
//         torch::nn::functional::InterpolateFuncOptions()
//             .size(std::vector<int64_t>{target_height, target_width})
//             .mode(torch::kNearest));
//   }

//   torch::Tensor ImageProcessor::rescale(const torch::Tensor& image,
//                                       double scale) {
//   return image * scale;
//  }

//   torch::Tensor normalize(const torch::Tensor& tensor) const {
//     return 2.0 * tensor - 1.0;
//   }

//   std::pair<int64_t, int64_t> adjust_dimensions(int64_t height,
//                                                 int64_t width) const {
//     height = height - (height % scale_factor_);
//     width = width - (width % scale_factor_);
//     return {height, width};
//   }

//  private:
//   float rescale_factor_ = 1.0 / 255.0;
//   int latent_channels_ = 4;
//   bool do_resize_ = true;
//   bool do_rescale_ = true;
//   bool do_normalize_ = true;
//   torch::TensorOptions options_;
// };
// TORCH_MODULE(SiglipImageProcessor);

// class GELUTanhImpl : public torch::nn::Module {
//  public:
//   explicit GELUTanhImpl(bool use_gelu_tanh_python = false)
//       : use_gelu_tanh_python_(use_gelu_tanh_python) {}

//   torch::Tensor forward(torch::Tensor input) {
//     if (use_gelu_tanh_python_) {
//       return gelu_tanh_python(input);
//     } else {
//       return torch::nn::functional::gelu(
//           input,
//           torch::nn::functional::GeluFuncOptions().approximate(torch::kTanh));
//     }
//   }

//  private:
//   bool use_gelu_tanh_python_;

//   torch::Tensor gelu_tanh_python(torch::Tensor input) {
//     const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
//     const float coefficient = 0.044715f;
//     return input * 0.5f *
//            (1.0f + torch::tanh(sqrt_2_over_pi *
//                                (input + coefficient * torch::pow(input,
//                                3))));
//   }
// };
// TORCH_MODULE(GELUTanh);

// class SiglipVisionEmbeddingsImpl : public torch::nn::Module {
//  public:
//   explicit SiglipVisionEmbeddingsImpl(const ModelContext& context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();
//     embed_dim_ = model_args.mm_hidden_size();
//     image_size_ = model_args.mm_image_size();
//     patch_embedding_ = register_module(
//         "patch_embedding",
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(model_args.mm_num_channels(),
//                                                    embed_dim_,
//                                                    model_args.mm_patch_size())
//                               .stride(model_args.mm_patch_size())
//                               .bias(false)));
//     patch_embedding_->weight.set_data(patch_embedding_->weight.to(options));

//     auto num_patches =
//         (model_args.mm_image_size() / model_args.mm_patch_size()) *
//         (model_args.mm_image_size() / model_args.mm_patch_size());
//     auto num_positions = num_patches + 1;
//     position_embedding_ =
//         register_parameter("position_embedding",
//                            torch::randn({num_positions, embed_dim_},
//                            options));
//     position_ids_ = register_buffer(
//         "position_ids",
//         torch::arange(0, num_positions, torch::kLong).unsqueeze(0));
//   }

//   torch::Tensor forward(const torch::Tensor& pixel_values) {
//     int64_t batch_size = pixel_values.size(0);
//     auto embeddings =
//         patch_embedding_->forward(pixel_values).flatten(2).transpose(1, 2);
//     embeddings += position_embedding_.index({position_ids_});
//     return embeddings;
//   }

//   // load the weight from the checkpoint
//   void load_state_dict(const StateDict& state_dict) {
//     const auto pos = state_dict.get_tensor("position_embedding.weight");
//     if (pos.defined()) {
//       CHECK_EQ(pos.sizes(), position_embedding_.sizes())
//           << "position_embedding weight size mismatch for " << name();
//       position_embedding_.data().copy_(pos);
//       is_position_embedding_loaded = true;
//     }

//     const auto weight = state_dict.get_tensor("patch_embedding.weight");
//     if (weight.defined()) {
//       DCHECK_EQ(patch_embedding_->weight.sizes(), weight.sizes())
//           << "patch_embedding weight size mismatch for " << name();
//       patch_embedding_->weight.data().copy_(weight);
//       is_patch_embedding_loaded = true;
//     }
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     CHECK(is_position_embedding_loaded)
//         << "weight is not loaded for " << prefix +
//         "position_embedding.weight";
//     CHECK(is_patch_embedding_loaded)
//         << "weight is not loaded for " << prefix + "patch_embedding.weight";
//   }

//  private:
//   int64_t embed_dim_;
//   int64_t image_size_;
//   bool is_position_embedding_loaded{false};
//   bool is_patch_embedding_loaded{false};
//   torch::Tensor position_ids_;
//   torch::nn::Conv2d patch_embedding_{nullptr};
//   torch::Tensor position_embedding_{nullptr};
// };
// TORCH_MODULE(SiglipVisionEmbeddings);

// class SiglipAttentionImpl : public torch::nn::Module {
//  public:
//   SiglipAttentionImpl(const ModelContext& context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();
//     CHECK(model_args.mm_hidden_size() % model_args.mm_num_attention_heads()
//     ==
//           0);
//     head_dim_ = model_args.mm_head_dim();
//     embed_dim_ = model_args.mm_hidden_size();
//     num_heads_ = model_args.mm_num_attention_heads();
//     const int64_t n_local_heads = num_heads_;

//     qkv_sizes_ = {n_local_heads * model_args.mm_head_dim(),
//                   n_local_heads * model_args.mm_head_dim(),
//                   n_local_heads * model_args.mm_head_dim()};

//     scale_ = 1.0f / std::sqrt(static_cast<float>(model_args.mm_head_dim()));
//     q_proj_ = register_module(
//         "q_proj",
//         DiTLinear(model_args.mm_hidden_size(), num_heads_ * head_dim_,
//         true));
//     k_proj_ = register_module(
//         "k_proj",
//         DiTLinear(model_args.mm_hidden_size(), num_heads_ * head_dim_,
//         true));
//     v_proj_ = register_module(
//         "v_proj",
//         DiTLinear(model_args.mm_hidden_size(), num_heads_ * head_dim_,
//         true));
//     o_proj_ = register_module(
//         "o_proj",
//         DiTLinear(
//             model_args.mm_hidden_size(), model_args.mm_hidden_size(), true));

//     q_proj_->to(options);
//     k_proj_->to(options);
//     v_proj_->to(options);
//     o_proj_->to(options);
//   }

//   torch::Tensor forward(const torch::Tensor& hidden_states) {
//     auto bsz = hidden_states.size(0);
//     auto tgt_len = hidden_states.size(1);

//     auto query_states = q_proj_(hidden_states);
//     auto key_states = k_proj_(hidden_states);
//     auto value_states = v_proj_(hidden_states);

//     // [batch_size, num_heads, seq_len, head_dim]
//     query_states = shape(query_states, tgt_len, bsz);
//     key_states = shape(key_states, -1, bsz);
//     value_states = shape(value_states, -1, bsz);

//     auto src_len = key_states.size(1);
//     auto attn_weights =
//         torch::matmul(query_states, key_states.transpose(-1, -2)) * scale_;
//     attn_weights = torch::softmax(attn_weights, -1);
//     auto attn_output = torch::matmul(attn_weights, value_states);
//     DCHECK_EQ(attn_output.sizes(),
//               torch::IntArrayRef({bsz * num_heads_, tgt_len, head_dim_}));
//     attn_output =
//         attn_output
//             .view(torch::IntArrayRef({bsz, num_heads_, tgt_len, head_dim_}))
//             .transpose(1, 2)
//             .contiguous();
//     attn_output =
//         attn_output.view(torch::IntArrayRef({bsz, tgt_len, embed_dim_}));
//     attn_output = attn_output.transpose(1, 2).contiguous();
//     return o_proj_(attn_output);
//   }

//   void load_state_dict(const StateDict& state_dict) {
//     q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
//     q_proj_weight_loaded_ = true;
//     q_proj_bias_loaded_ = true;
//     k_proj_->load_state_dict(state_dict.get_dict_with_prefix("k_proj."));
//     k_proj_weight_loaded_ = true;
//     k_proj_bias_loaded_ = true;
//     v_proj_->load_state_dict(state_dict.get_dict_with_prefix("v_proj."));
//     v_proj_weight_loaded_ = true;
//     v_proj_bias_loaded_ = true;
//     o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
//     o_proj_weight_loaded_ = true;
//     o_proj_bias_loaded_ = true;
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     CHECK(q_proj_weight_loaded_)
//         << "weight is not loaded for " << prefix + "q_proj.weight";
//     CHECK(q_proj_bias_loaded_)
//         << "weight is not loaded for " << prefix + "q_proj.bias";
//     CHECK(k_proj_weight_loaded_)
//         << "weight is not loaded for " << prefix + "k_proj.weight";
//     CHECK(k_proj_bias_loaded_)
//         << "weight is not loaded for " << prefix + "k_proj.bias";
//     CHECK(v_proj_weight_loaded_)
//         << "weight is not loaded for " << prefix + "v_proj.weight";
//     CHECK(v_proj_bias_loaded_)
//         << "weight is not loaded for " << prefix + "v_proj.bias";
//     CHECK(o_proj_weight_loaded_)
//         << "weight is not loaded for " << prefix + "out_proj.weight";
//     CHECK(o_proj_bias_loaded_)
//         << "weight is not loaded for " << prefix + "out_proj.bias";
//   }

//  private:
//   torch::Tensor shape(torch::Tensor tensor, int64_t seq_len, int64_t bsz) {
//     return tensor.view({bsz, seq_len, num_heads_, head_dim_})
//         .transpose(1, 2)
//         .contiguous();
//   }

//  private:
//   int64_t embed_dim_;
//   int64_t num_heads_;
//   int64_t head_dim_;
//   float scale_;
//   std::vector<int64_t> qkv_sizes_;

//   DiTLinear o_proj_ = nullptr;
//   DiTLinear q_proj_ = nullptr;
//   DiTLinear k_proj_ = nullptr;
//   DiTLinear v_proj_ = nullptr;

//   bool q_proj_weight_loaded_ = false;
//   bool q_proj_bias_loaded_ = false;
//   bool k_proj_weight_loaded_ = false;
//   bool k_proj_bias_loaded_ = false;
//   bool v_proj_weight_loaded_ = false;
//   bool v_proj_bias_loaded_ = false;
//   bool o_proj_weight_loaded_ = false;
//   bool o_proj_bias_loaded_ = false;
// };
// TORCH_MODULE(SiglipAttention);

// class SlglipMLPImpl : public torch::nn::Module {
//  public:
//   explicit SiglipMLPImpl(const ModelContext& context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();
//     // act_ = register_module("act", torch::nn::Functional(quick_gelu));
//     act_ = register_module("act_", GELUTanh())

//     fc1_ = register_module("fc1",
//                            DiTLinear(model_args.mm_hidden_size(),
//                                      model_args.mm_intermediate_size(),
//                                      true));
//     fc2_ = register_module("fc2",
//                            DiTLinear(model_args.mm_intermediate_size(),
//                                      model_args.mm_hidden_size(),
//                                      true));

//     fc1_->to(options);
//     fc2_->to(options);
//   }

//   torch::Tensor forward(const torch::Tensor& hidden_states) {
//     return fc2_(act_(fc1_(hidden_states)));
//   }

//   void load_state_dict(const StateDict& state_dict) {
//     fc1_->load_state_dict(state_dict.get_dict_with_prefix("fc1."));
//     fc1_weight_loaded_ = true;
//     fc1_bias_loaded_ = true;
//     fc2_->load_state_dict(state_dict.get_dict_with_prefix("fc2."));
//     fc2_weight_loaded_ = true;
//     fc2_bias_loaded_ = true;
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     CHECK(fc1_weight_loaded_)
//         << "weight is not loaded for " << prefix + "fc1.weight";
//     CHECK(fc1_bias_loaded_)
//         << "weight is not loaded for " << prefix + "fc1.bias";
//     CHECK(fc2_weight_loaded_)
//         << "weight is not loaded for " << prefix + "fc2.weight";
//     CHECK(fc2_bias_loaded_)
//         << "weight is not loaded for " << prefix + "fc2.bias";
//   }

//  private:
//   // torch::nn::Functional act_ = nullptr;
//   DiTLinear fc1_ = nullptr;
//   DiTLinear fc2_ = nullptr;
//   bool fc1_weight_loaded_ = false;
//   bool fc1_bias_loaded_ = false;
//   bool fc2_weight_loaded_ = false;
//   bool fc2_bias_loaded_ = false;
// };
// TORCH_MODULE(SlglipMLP);

// class SiglipEncoderLayerImpl : public torch::nn::Module {
//  public:
//   explicit SiglipEncoderLayerImpl(const ModelContext& context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();
//     self_attn_ = register_module("self_attn", CLIPAttention(context));
//     layer_norm1_ = register_module(
//         "layer_norm1",
//         torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
//                 .elementwise_affine(true)
//                 .eps(model_args.mm_layer_norm_eps())));
//     layer_norm2_ = register_module(
//         "layer_norm2",
//         torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
//                 .elementwise_affine(true)
//                 .eps(model_args.mm_layer_norm_eps())));
//     layer_norm1_->weight.set_data(layer_norm1_->weight.to(options));
//     layer_norm1_->bias.set_data(layer_norm1_->bias.to(options));
//     layer_norm2_->weight.set_data(layer_norm2_->weight.to(options));
//     layer_norm2_->bias.set_data(layer_norm2_->bias.to(options));
//     mlp_ = register_module("mlp", CLIPMLP(context));
//   }

//   torch::Tensor forward(const torch::Tensor& hidden_states) {
//     auto residual = hidden_states;
//     const auto& layer_norm1 = layer_norm1_(hidden_states);
//     auto h = self_attn_(layer_norm1) + residual;
//     residual = h;
//     h = layer_norm2_(h);
//     h = mlp_(h);
//     h += residual;
//     return h;
//   }

//   void load_state_dict(const StateDict& state_dict) {
//     self_attn_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
//     weight::load_weight(state_dict,
//                         "layer_norm1.weight",
//                         layer_norm1_->weight,
//                         layer_norm1_weight_loaded_);
//     weight::load_weight(state_dict,
//                         "layer_norm1.bias",
//                         layer_norm1_->bias,
//                         layer_norm1_bias_loaded_);
//     weight::load_weight(state_dict,
//                         "layer_norm2.weight",
//                         layer_norm2_->weight,
//                         layer_norm2_weight_loaded_);
//     weight::load_weight(state_dict,
//                         "layer_norm2.bias",
//                         layer_norm2_->bias,
//                         layer_norm2_bias_loaded_);
//     mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     self_attn_->verify_loaded_weights(prefix + "self_attn.");
//     mlp_->verify_loaded_weights(prefix + "mlp.");
//     CHECK(layer_norm1_weight_loaded_)
//         << "weight is not loaded for " << prefix + "layer_norm1.weight";
//     CHECK(layer_norm1_bias_loaded_)
//         << "weight is not loaded for " << prefix + "layer_norm1.bias";
//     CHECK(layer_norm2_weight_loaded_)
//         << "weight is not loaded for " << prefix + "layer_norm2.weight";
//     CHECK(layer_norm2_bias_loaded_)
//         << "weight is not loaded for " << prefix + "layer_norm2.bias";
//   }

//  private:
//   bool layer_norm1_weight_loaded_ = false;
//   bool layer_norm1_bias_loaded_ = false;
//   bool layer_norm2_weight_loaded_ = false;
//   bool layer_norm2_bias_loaded_ = false;

//   torch::nn::LayerNorm layer_norm1_ = nullptr;
//   torch::nn::LayerNorm layer_norm2_ = nullptr;
//   SiglipAttention self_attn_ = nullptr;
//   SiglipVisionEmbeddingsImpl mlp_ = nullptr;
// };
// TORCH_MODULE(SiglipEncoderLayer);

// class SiglipEncoderImpl : public torch::nn::Module {
//  public:
//   explicit SiglipEncoderImpl(const ModelContext& context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();
//     blocks_ = register_module("layers", torch::nn::ModuleList());
//     layers_.reserve(model_args.mm_num_hidden_layers());
//     for (int32_t i = 0; i < model_args.mm_num_hidden_layers(); i++) {
//       auto block = SiglipEncoderLayer(context);
//       layers_.push_back(block);
//       blocks_->push_back(block);
//     }
//   }

//   // Output hidden states for last intermediate layers
//   torch::Tensor forward(const torch::Tensor& embeddings) {
//     bool output_hidden_states = false;
//     bool output_attentions = false;
//     c10::optional<torch::Tensor> attention_mask = c10::nullopt;
//     c10::optional<torch::Tensor> head_mask = c10::nullopt;
//     std::vector<torch::Tensor> all_hidden_states;
//     std::vector<torch::Tensor> all_attentions;
//     std::vector<torch::Tensor> encoder_states;

//     auto hidden_states = embeddings;
//     for (size_t i = 0; i < layers_.size(); ++i) {
//       encoder_states.emplace_back(hidden_states);
//       auto& layer = layers_[i];
//       hidden_states = layer(hidden_states);
//     }
//     if (output_hidden_states) encoder_states.emplace_back(hidden_states);

//     std::vector<torch::Tensor> outputs = {hidden_states};
//     if (output_hidden_states) {
//       // todo
//     }
//     if (output_attentions) {
//       // todo
//     }
//     return outputs[0];
//   }

//   void load_state_dict(const StateDict& state_dict) {
//     for (size_t i = 0; i < layers_.size(); ++i) {
//       layers_[i]->load_state_dict(
//           state_dict.get_dict_with_prefix("layers." + std::to_string(i) +
//           "."));
//     }
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     for (size_t i = 0; i < layers_.size(); ++i) {
//       layers_[i]->verify_loaded_weights(prefix + "layers." +
//       std::to_string(i) +
//                                         ".");
//     }
//   }

//  private:
//   torch::nn::ModuleList blocks_ = nullptr;
//   std::vector<SiglipEncoderLayer> layers_ = {};
// };
// TORCH_MODULE(SiglipEncoder);

// class SiglipMultiheadAttentionPoolingHeadImpl : public torch::nn::Module {
//  public:
//   explicit SiglipMultiheadAttentionPoolingHeadImpl(const ModelContext&
//   context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();

//     // 初始化参数
//     int64_t hidden_size = model_args.mm_hidden_size();
//     int64_t num_attention_heads = model_args.mm_num_attention_heads();
//     float layer_norm_eps = model_args.mm_layer_norm_eps();

//     // 注册probe参数（对应Python的nn.Parameter）
//     probe_ = register_parameter(
//         "probe",
//         torch::randn({1, 1, hidden_size}, options));

//     // 初始化MultiheadAttention（batch_first=True对应Python参数）
//     attention_ = register_module(
//         "attention",
//         torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(hidden_size,
//             num_attention_heads)
//                 .batch_first(true)  // 匹配Python的batch_first=True
//                 .bias(true)         // 默认启用bias，与Python一致
//                 .dropout(0.0f)));    // 无dropout（Python默认0.0）

//     // 初始化LayerNorm
//     layernorm_ = register_module(
//         "layernorm",
//         torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({hidden_size})
//                 .elementwise_affine(true)
//                 .eps(layer_norm_eps)));
//     // 设置LayerNorm的tensor选项（设备、数据类型）
//     layernorm_->weight.set_data(layernorm_->weight.to(options));
//     layernorm_->bias.set_data(layernorm_->bias.to(options));

//     // 初始化MLP（复用已有SiglipMLP实现，修正原代码笔误SlglipMLP）
//     mlp_ = register_module("mlp", SiglipMLP(context));
//   }

//   torch::Tensor forward(const torch::Tensor& hidden_state) {
//     CHECK(hidden_state.dim() == 3)
//         << "hidden_state must be 3D tensor (batch_size, seq_len,
//         hidden_size), got " << hidden_state.dim() << "D";

//     int64_t batch_size = hidden_state.size(0);

//     // Probe重复batch_size次（对应Python的probe.repeat(batch_size, 1, 1)）
//     auto probe = probe_.repeat({batch_size, 1, 1});

//     // MultiheadAttention前向传播（返回tuple，取第一个元素为输出）
//     auto attention_output = attention_(probe, hidden_state,
//     hidden_state).output;

//     // 残差连接 + LayerNorm + MLP
//     auto residual = attention_output;
//     auto hidden_state_norm = layernorm_(attention_output);
//     auto hidden_state = residual + mlp_(hidden_state_norm);

//     // 返回第一个token的输出（对应Python的hidden_state[:, 0]）
//     return hidden_state.slice(1, 0, 1).squeeze(1);
//   }

//   // 加载权重（参考原有类的权重加载逻辑）
//   void load_state_dict(const StateDict& state_dict) {
//     // 加载probe参数
//     const auto probe_tensor = state_dict.get_tensor("probe");
//     if (probe_tensor.defined()) {
//       CHECK_EQ(probe_tensor.sizes(), probe_.sizes())
//           << "probe weight size mismatch for " << name();
//       probe_.data().copy_(probe_tensor);
//       is_probe_loaded_ = true;
//     }

//     // 加载Attention权重
//     attention_->load_state_dict(state_dict.get_dict_with_prefix("attention."));
//     is_attention_loaded_ = true;

//     // 加载LayerNorm权重
//     weight::load_weight(
//         state_dict,
//         "layernorm.weight",
//         layernorm_->weight,
//         is_layernorm_weight_loaded_);
//     weight::load_weight(
//         state_dict,
//         "layernorm.bias",
//         layernorm_->bias,
//         is_layernorm_bias_loaded_);

//     // 加载MLP权重
//     mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
//     is_mlp_loaded_ = true;
//   }

//   // 验证权重是否加载完整
//   void verify_loaded_weights(const std::string& prefix) const {
//     CHECK(is_probe_loaded_)
//         << "weight is not loaded for " << prefix + "probe";
//     CHECK(is_attention_loaded_)
//         << "weight is not loaded for " << prefix + "attention.";
//     CHECK(is_layernorm_weight_loaded_)
//         << "weight is not loaded for " << prefix + "layernorm.weight";
//     CHECK(is_layernorm_bias_loaded_)
//         << "weight is not loaded for " << prefix + "layernorm.bias";
//     CHECK(is_mlp_loaded_)
//         << "weight is not loaded for " << prefix + "mlp.";
//     mlp_->verify_loaded_weights(prefix + "mlp.");
//   }

//  private:
//   // 网络组件
//   torch::Tensor probe_;
//   torch::nn::MultiheadAttention attention_{nullptr};
//   torch::nn::LayerNorm layernorm_{nullptr};
//   SiglipMLP mlp_{nullptr};

//   // 权重加载状态标志
//   bool is_probe_loaded_{false};
//   bool is_attention_loaded_{false};
//   bool is_layernorm_weight_loaded_{false};
//   bool is_layernorm_bias_loaded_{false};
//   bool is_mlp_loaded_{false};
// };
// TORCH_MODULE(SiglipMultiheadAttentionPoolingHead);

// class SiglipVisionTransformerImpl : public torch::nn::Module {
//  public:
//   explicit SiglipVisionTransformerImpl(const ModelContext& context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();
//     embeddings_ = register_module("embeddings", CLIPTextEmbedding(context));
//     final_layer_norm_ = register_module(
//         "final_layer_norm",
//         torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
//                 .elementwise_affine(true)
//                 .eps(model_args.mm_layer_norm_eps())));
//     head_ = register_module(
//         "head", SiglipMultiheadAttentionPoolingHead(context));
//     final_layer_norm_->weight.set_data(final_layer_norm_->weight.to(options));
//     final_layer_norm_->bias.set_data(final_layer_norm_->bias.to(options));
//     encoder_ = register_module("encoder", CLIPEncoder(context));
//     eos_token_id = model_args.eos_token_id();
//   }

//   torch::Tensor forward(const torch::Tensor& input_ids) {
//     if (!input_ids.defined()) {
//       LOG(FATAL) << "input_ids is undefined.";
//     }
//     auto input_shape = input_ids.sizes();
//     auto reshaped_input_ids = input_ids.view({-1, input_shape.back()});
//     auto hidden_states = embeddings_->forward(reshaped_input_ids);
//     auto encoder_output = encoder_->forward(hidden_states);
//     auto last_hidden_state = final_layer_norm_->forward(encoder_output);
//     return last_hidden_state;
//   }

//   // load the weight from the checkpoint
//   void load_state_dict(const StateDict& state_dict) {
//     embeddings_->load_state_dict(
//         state_dict.get_dict_with_prefix("embeddings."));
//     encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
//     weight::load_weight(state_dict,
//                         "final_layer_norm.weight",
//                         final_layer_norm_->weight,
//                         final_layer_norm_weight_loaded_);
//     weight::load_weight(state_dict,
//                         "final_layer_norm.bias",
//                         final_layer_norm_->bias,
//                         final_layer_norm_bias_loaded_);
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     embeddings_->verify_loaded_weights(prefix + "embeddings.");
//     encoder_->verify_loaded_weights(prefix + "encoder.");
//     CHECK(final_layer_norm_weight_loaded_)
//         << "weight is not loaded for " << prefix + "final_layer_norm.weight";
//     CHECK(final_layer_norm_bias_loaded_)
//         << "weight is not loaded for " << prefix + "final_layer_norm.bias";
//   }

//  private:
//   int64_t eos_token_id;
//   bool final_layer_norm_weight_loaded_ = false;
//   bool final_layer_norm_bias_loaded_ = false;
//   SiglipMultiheadAttentionPoolingHead head_ = nullptr;
//   CLIPTextEmbedding embeddings_ = nullptr;
//   CLIPEncoder encoder_ = nullptr;
//   torch::nn::LayerNorm final_layer_norm_ = nullptr;
// };
// TORCH_MODULE(SiglipVisionTransformer);

// class SiglipVisionModelImpl : public torch::nn::Module {
//  public:
//   explicit SiglipVisionModelImpl(const ModelContext& context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();
//     eos_token_id = model_args.eos_token_id();
//     transformer_ = register_module("transformer",
//     CLIPTextTransformer(context));
//   }

//   torch::Tensor forward(const torch::Tensor& input_ids) {
//     auto last_hidden_states = transformer_->forward(input_ids);
//     int64_t batch_size = last_hidden_states.size(0);
//     auto device = last_hidden_states.device();
//     torch::Tensor batch_indices = torch::arange(batch_size, device);
//     torch::Tensor end_pos;
//     if (eos_token_id == 2) {
//       auto argmax_result = input_ids.to(device).max(1);
//       end_pos = std::get<1>(argmax_result);
//     } else {
//       torch::Tensor eos_mask =
//           (input_ids == eos_token_id).to(device, torch::kInt);
//       auto argmax_result = eos_mask.max(1);
//       end_pos = std::get<1>(argmax_result);
//     }
//     torch::Tensor pooled_output =
//         last_hidden_states.index({batch_indices, end_pos});
//     pooled_output = head_(pooled_output);
//     return pooled_output;
//   }

//   void load_state_dict(const StateDict& state_dict) {
//     transformer_->load_state_dict(
//         state_dict.get_dict_with_prefix("text_model."));
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     transformer_->verify_loaded_weights(prefix + ".");
//   }

//   void load_model(std::unique_ptr<DiTFolderLoader> loader) {
//     LOG(INFO) << "Loading CLIPTextModel from ModelLoader...";
//     for (const auto& state_dict : loader->get_state_dicts()) {
//       transformer_->load_state_dict(
//           state_dict->get_dict_with_prefix("text_model."));
//     }

//     // verify
//     transformer_->verify_loaded_weights("text_model.");
//     LOG(INFO) << "clip_text_model loaded successfully.";
//   }

//  private:
//   int64_t eos_token_id;
//   CLIPTextTransformer transformer_ = nullptr;
// };
// TORCH_MODULE(SiglipVisionModel);

// REGISTER_MODEL_ARGS(CLIPTextModel, [&] {
//   LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
//   LOAD_ARG_OR(vocab_size, "vocab_size", 49408);
//   LOAD_ARG_OR(mm_hidden_size, "hidden_size", 768);
//   LOAD_ARG_OR(mm_intermediate_size, "intermediate_size", 3072);
//   LOAD_ARG_OR(mm_projection_dim, "projection_dim", 768);
//   LOAD_ARG_OR(mm_num_hidden_layers, "num_hidden_layers", 12);
//   LOAD_ARG_OR(mm_num_attention_heads, "num_attention_heads", 12);
//   LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 77);
//   LOAD_ARG_OR(mm_hidden_act, "hidden_act", "quick_gelu");
//   LOAD_ARG_OR(mm_layer_norm_eps, "layer_norm_eps", 1e-5f);
//   LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
//   LOAD_ARG_OR(mm_head_dim, "head_dim", 64);
// });

// }  // namespace xllm

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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <regex>
#include <unordered_map>

#include "core/framework/dit_model_loader.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/siglip_encoder_layer.h"
#include "dit_linear.h"
#include "models/model_registry.h"
#include "processors/clip_image_processor.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm {

// class SiglipImageProcessorImpl : public torch::nn::Module {//public
// InputProcessor {
//  public:
//   SiglipImageProcessorImpl(ModelContext context) {
//                           //  bool do_resize = true,
//                           //  bool do_rescale = true,
//                           //  bool do_normalize = true,
//                           //  float rescale_factor = 1.0 / 255.0,
//                           //  int64_t latent_channels = 4) {
//     const auto& model_args = context.get_model_args();
//     options_ = context.get_tensor_options();
//     // rescale_factor_ = rescale_factor;
//     // latent_channels_ = latent_channels;
//     // do_resize_ = do_resize;
//     // do_rescale_ = do_rescale;
//     // do_normalize_ = do_normalize;

//     // if (model_args.mm_patch_size() > 0) {
//     //   scale_factor_ = model_args.mm_patch_size();
//     // } else {
//     //   scale_factor_ = 1;
//     // }

//     // image_mean_ = torch::tensor({0.485, 0.456, 0.406}, options_);
//     // image_std_ = torch::tensor({0.229, 0.224, 0.225}, options_);
//     image_mean_ = torch::Tensor({0.5, 0.5, 0.5}, options_);
//     image_std_ = torch::Tensor({0.5, 0.5, 0.5}, options_);

//   }

//   torch::Tensor preprocess(const torch::Tensor& images,
//       std::optional<int64_t> height = std::nullopt,
//       std::optional<int64_t> width = std::nullopt) {
//     int64_t batch_size = images.size(0);
//     std::vector<torch::Tensor> processed_images;

//     auto hw = get_default_height_width(images, height, width);
//     int64_t target_h = hw.first;
//     int64_t target_w = hw.second;

//     for (int64_t i = 0; i < batch_size; ++i) {
//       torch::Tensor image = images[i];

//       if (do_resize_) {
//         // image = resize(image, target_h, target_w);
//         image = resize(image, size, resample_);
//       }

//       if (do_rescale_) {
//         image = rescale(image, rescale_factor_);
//       }

//       if (do_normalize_) {
//         image = normalize(image, image_mean_, image_std_);
//       }

//       processed_images.push_back(image);
//     }
//     return torch::stack(processed_images);
//   }

//  private:
//   std::pair<int64_t, int64_t> get_default_height_width(
//       const torch::Tensor& image,
//       std::optional<int64_t> height = std::nullopt,
//       std::optional<int64_t> width = std::nullopt) const {
//     int64_t h, w;
//     if (image.dim() == 3) {
//       h = image.size(1);
//       w = image.size(2);
//     } else if (image.dim() == 4) {
//       h = image.size(2);
//       w = image.size(3);
//     } else {
//       LOG(FATAL) << "Unsupported image dimension: " << image.dim();
//     }

//     int64_t target_h = height.value_or(h);
//     int64_t target_w = width.value_or(w);
//     return adjust_dimensions(target_h, target_w);
//   }

//   torch::Tensor resize(const torch::Tensor& image,
//                        int64_t target_height,
//                        int64_t target_width) const {
//     torch::Tensor img = image;
//     bool added_batch = false;
//     if (img.dim() == 3) {
//       img = img.unsqueeze(0);
//       added_batch = true;
//     }
//     auto out = torch::nn::functional::interpolate(
//         img,
//         torch::nn::functional::InterpolateFuncOptions()
//             .size(std::vector<int64_t>{target_height, target_width})
//             .mode(torch::kNearest));
//     if (added_batch) out = out.squeeze(0);
//     return out;
//   }

//   // torch::Tensor rescale(const torch::Tensor& image,
//   //                                     double scale) {
//   //   return image * scale;
//   // }

//   torch::Tensor resize(const torch::Tensor& image,
//                                       const std::vector<int64_t>& size,
//                                       int resample,
//                                       bool antialias) {
//     if (image.dim() != 3) {
//       LOG(FATAL) << "Input image must be a 3D tensor (C x H x W).";
//     }
//     auto options = torch::nn::functional::InterpolateFuncOptions()
//                       .size(size)
//                       .align_corners(false)
//                       .antialias(antialias);
//     switch (resample) {
//       case 1:
//         options.mode(torch::kNearest);
//         break;
//       case 2:
//         options.mode(torch::kBilinear);
//         break;
//       case 3:
//         options.mode(torch::kBicubic);
//         break;
//       default:
//         LOG(FATAL) << "Invalid resample value. Must be one of 1, 2, or 3.";
//     }
//     return torch::nn::functional::interpolate(image.unsqueeze(0), options)
//         .squeeze(0)
//         .clamp(0, 255)
//         .to(torch::kUInt8);
//   }

//   torch::Tensor normalize(const torch::Tensor& tensor,
//                           const torch::Tensor& mean,
//                           const torch::Tensor& std) const {
//     auto t = tensor;
//     if (t.dim() == 3 && mean.numel() == t.size(0)) {
//       return (t - mean.view({-1, 1, 1})) / std.view({-1, 1, 1});
//     } else {
//       return 2.0 * t - 1.0;
//     }
//   }

//   std::pair<int64_t, int64_t> adjust_dimensions(int64_t height,
//                                                 int64_t width) const {
//     int64_t h = height - (height % scale_factor_);
//     int64_t w = width - (width % scale_factor_);
//     if (h <= 0) h = scale_factor_;
//     if (w <= 0) w = scale_factor_;
//     return {h, w};
//   }

//  private:
//   float rescale_factor_ = 0.00392156862745098;
//   int64_t latent_channels_ = 4;
//   bool do_resize_ = true;
//   bool do_rescale_ = true;
//   bool do_normalize_ = true;
//   torch::TensorOptions options_;
//   torch::Tensor image_mean_, image_std_;
//   int64_t scale_factor_ = 1;
//   int64_t resample_ = 3;
//   // size
//   std::vector<int64_t> size = {384,384};
// };
// TORCH_MODULE(SiglipImageProcessor);

// class GELUTanhImpl : public torch::nn::Module {
//  public:
//   explicit GELUTanhImpl(bool use_gelu_tanh_python = false)
//       : use_gelu_tanh_python_(use_gelu_tanh_python) {}

//   torch::Tensor forward(torch::Tensor input) {
//     if (use_gelu_tanh_python_) {
//       return gelu_tanh_python(input);
//     } else {
//       return torch::nn::functional::gelu(
//           input,
//           //
//           torch::nn::functional::GELUFuncOptions().approximate(torch::kTanh));
//           torch::nn::functional::GELUFuncOptions().approximate(true));
//     }
//   }

//  private:
//   bool use_gelu_tanh_python_;

//   torch::Tensor gelu_tanh_python(torch::Tensor input) {
//     const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
//     const float coefficient = 0.044715f;
//     return input * 0.5f *
//            (1.0f + torch::tanh(sqrt_2_over_pi *
//                                (input + coefficient * torch::pow(input,
//                                3))));
//   }
// };
// TORCH_MODULE(GELUTanh);

torch::Tensor GELUTanh(torch::Tensor input) {
  const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
  const float coefficient = 0.044715f;
  return input * 0.5f *
         (1.0f + torch::tanh(sqrt_2_over_pi *
                             (input + coefficient * torch::pow(input, 3))));
}

class SiglipVisionEmbeddingsImpl : public torch::nn::Module {
 public:
  explicit SiglipVisionEmbeddingsImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embed_dim_ = model_args.mm_hidden_size();
    image_size_ = model_args.mm_image_size();
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(model_args.mm_num_channels(),
                                                   embed_dim_,
                                                   model_args.mm_patch_size())
                              .stride(model_args.mm_patch_size())
                              .bias(false)));
    patch_embedding_->weight.set_data(patch_embedding_->weight.to(options));

    auto num_patches =
        (model_args.mm_image_size() / model_args.mm_patch_size()) *
        (model_args.mm_image_size() / model_args.mm_patch_size());
    auto num_positions = num_patches;
    position_embedding_ =
        register_parameter("position_embedding",
                           torch::randn({num_positions, embed_dim_}, options));
    position_ids_ = register_buffer(
        "position_ids",
        torch::arange(0, num_positions, torch::kLong).unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& pixel_values) {
    int64_t batch_size = pixel_values.size(0);
    int64_t seq_length = pixel_values.size(1);
    LOG(INFO) << "pixel_value: " << pixel_values.sizes();
    auto patch_embeds =
        patch_embedding_->forward(pixel_values).flatten(2).transpose(1, 2);
    LOG(INFO) << "patch_embeds: " << patch_embeds.sizes();
    torch::Tensor embeddings =
        patch_embeds + position_embedding_.index({position_ids_});
    return embeddings;
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto pos = state_dict.get_tensor("position_embedding.weight");
    if (pos.defined()) {
      CHECK_EQ(pos.sizes(), position_embedding_.sizes())
          << "position_embedding weight size mismatch for " << name();
      position_embedding_.data().copy_(pos);
      is_position_embedding_loaded = true;
    }

    const auto weight = state_dict.get_tensor("patch_embedding.weight");
    if (weight.defined()) {
      DCHECK_EQ(patch_embedding_->weight.sizes(), weight.sizes())
          << "patch_embedding weight size mismatch for " << name();
      patch_embedding_->weight.data().copy_(weight);
      is_patch_embedding_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_position_embedding_loaded)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
    CHECK(is_patch_embedding_loaded)
        << "weight is not loaded for " << prefix + "patch_embedding.weight";
  }

 private:
  int64_t embed_dim_;
  int64_t image_size_;
  bool is_position_embedding_loaded{false};
  bool is_patch_embedding_loaded{false};
  torch::Tensor position_ids_;
  torch::nn::Conv2d patch_embedding_{nullptr};
  torch::Tensor position_embedding_{nullptr};
};
TORCH_MODULE(SiglipVisionEmbeddings);

class SiglipAttentionImpl : public torch::nn::Module {
 public:
  SiglipAttentionImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    CHECK(model_args.mm_hidden_size() % model_args.mm_num_attention_heads() ==
          0);
    // head_dim_ = model_args.mm_head_dim();
    embed_dim_ = model_args.mm_hidden_size();
    num_heads_ = model_args.mm_num_attention_heads();
    head_dim_ = embed_dim_ / num_heads_;

    q_proj_ = register_module(
        "q_proj",
        DiTLinear(
            model_args.mm_hidden_size(), model_args.mm_hidden_size(), true));
    k_proj_ = register_module(
        "k_proj",
        DiTLinear(
            model_args.mm_hidden_size(), model_args.mm_hidden_size(), true));
    v_proj_ = register_module(
        "v_proj",
        DiTLinear(
            model_args.mm_hidden_size(), model_args.mm_hidden_size(), true));
    o_proj_ = register_module(
        "o_proj",
        DiTLinear(
            model_args.mm_hidden_size(), model_args.mm_hidden_size(), true));

    q_proj_->to(options);
    k_proj_->to(options);
    v_proj_->to(options);
    o_proj_->to(options);

    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto bsz = hidden_states.size(0);
    auto tgt_len = hidden_states.size(1);

    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    query_states = query_states.view({bsz, tgt_len, num_heads_, head_dim_})
                       .transpose(1, 2)
                       .contiguous();
    key_states = key_states.view({bsz, tgt_len, num_heads_, head_dim_})
                     .transpose(1, 2)
                     .contiguous();
    value_states = value_states.view({bsz, tgt_len, num_heads_, head_dim_})
                       .transpose(1, 2)
                       .contiguous();

    auto attn_weights =
        torch::matmul(query_states, key_states.transpose(-1, -2)) * scale_;
    attn_weights = torch::softmax(attn_weights, -1);

    auto attn_output = torch::matmul(attn_weights, value_states);

    attn_output = attn_output.transpose(1, 2).contiguous().view(
        {bsz, tgt_len, embed_dim_});

    return o_proj_(attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
    q_proj_weight_loaded_ = true;
    q_proj_bias_loaded_ = true;
    k_proj_->load_state_dict(state_dict.get_dict_with_prefix("k_proj."));
    k_proj_weight_loaded_ = true;
    k_proj_bias_loaded_ = true;
    v_proj_->load_state_dict(state_dict.get_dict_with_prefix("v_proj."));
    v_proj_weight_loaded_ = true;
    v_proj_bias_loaded_ = true;
    o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
    o_proj_weight_loaded_ = true;
    o_proj_bias_loaded_ = true;
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(q_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "q_proj.weight";
    CHECK(q_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "q_proj.bias";
    CHECK(k_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "k_proj.weight";
    CHECK(k_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "k_proj.bias";
    CHECK(v_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "v_proj.weight";
    CHECK(v_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "v_proj.bias";
    CHECK(o_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "out_proj.weight";
    CHECK(o_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "out_proj.bias";
  }

 private:
  int64_t embed_dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  float scale_;

  DiTLinear o_proj_ = nullptr;
  DiTLinear q_proj_ = nullptr;
  DiTLinear k_proj_ = nullptr;
  DiTLinear v_proj_ = nullptr;

  bool q_proj_weight_loaded_ = false;
  bool q_proj_bias_loaded_ = false;
  bool k_proj_weight_loaded_ = false;
  bool k_proj_bias_loaded_ = false;
  bool v_proj_weight_loaded_ = false;
  bool v_proj_bias_loaded_ = false;
  bool o_proj_weight_loaded_ = false;
  bool o_proj_bias_loaded_ = false;
};
TORCH_MODULE(SiglipAttention);

class SiglipMLPImpl : public torch::nn::Module {
 public:
  explicit SiglipMLPImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    act_ = register_module("act", torch::nn::Functional(GELUTanh));

    fc1_ = register_module("fc1",
                           DiTLinear(model_args.mm_hidden_size(),
                                     model_args.mm_intermediate_size(),
                                     true));
    fc2_ = register_module("fc2",
                           DiTLinear(model_args.mm_intermediate_size(),
                                     model_args.mm_hidden_size(),
                                     true));

    fc1_->to(options);
    fc2_->to(options);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    // return fc2_->forward(act_->forward(fc1_->forward(hidden_states)));
    return fc2_(act_(fc1_(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {
    fc1_->load_state_dict(state_dict.get_dict_with_prefix("fc1."));
    fc1_weight_loaded_ = true;
    fc1_bias_loaded_ = true;
    fc2_->load_state_dict(state_dict.get_dict_with_prefix("fc2."));
    fc2_weight_loaded_ = true;
    fc2_bias_loaded_ = true;
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(fc1_weight_loaded_)
        << "weight is not loaded for " << prefix + "fc1.weight";
    CHECK(fc1_bias_loaded_)
        << "weight is not loaded for " << prefix + "fc1.bias";
    CHECK(fc2_weight_loaded_)
        << "weight is not loaded for " << prefix + "fc2.weight";
    CHECK(fc2_bias_loaded_)
        << "weight is not loaded for " << prefix + "fc2.bias";
  }

 private:
  // GELUTanh act_{nullptr};
  torch::nn::Functional act_ = nullptr;
  DiTLinear fc1_ = nullptr;
  DiTLinear fc2_ = nullptr;
  bool fc1_weight_loaded_ = false;
  bool fc1_bias_loaded_ = false;
  bool fc2_weight_loaded_ = false;
  bool fc2_bias_loaded_ = false;
};
TORCH_MODULE(SiglipMLP);

class SiglipEncoderLayerImpl : public torch::nn::Module {
 public:
  explicit SiglipEncoderLayerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attn_ = register_module("self_attn", SiglipAttention(context));
    layer_norm1_ = register_module(
        "layer_norm1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    layer_norm2_ = register_module(
        "layer_norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    layer_norm1_->weight.set_data(layer_norm1_->weight.to(options));
    layer_norm1_->bias.set_data(layer_norm1_->bias.to(options));
    layer_norm2_->weight.set_data(layer_norm2_->weight.to(options));
    layer_norm2_->bias.set_data(layer_norm2_->bias.to(options));
    mlp_ = register_module("mlp", SiglipMLP(context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto residual = hidden_states;
    auto ln1_out = layer_norm1_->forward(hidden_states);
    auto h = self_attn_->forward(ln1_out) + residual;
    residual = h;
    h = layer_norm2_->forward(h);
    h = mlp_->forward(h);
    h += residual;
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attn_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
    weight::load_weight(state_dict,
                        "layer_norm1.weight",
                        layer_norm1_->weight,
                        layer_norm1_weight_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm1.bias",
                        layer_norm1_->bias,
                        layer_norm1_bias_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm2.weight",
                        layer_norm2_->weight,
                        layer_norm2_weight_loaded_);
    weight::load_weight(state_dict,
                        "layer_norm2.bias",
                        layer_norm2_->bias,
                        layer_norm2_bias_loaded_);
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    CHECK(layer_norm1_weight_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm1.weight";
    CHECK(layer_norm1_bias_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm1.bias";
    CHECK(layer_norm2_weight_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm2.weight";
    CHECK(layer_norm2_bias_loaded_)
        << "weight is not loaded for " << prefix + "layer_norm2.bias";
  }

 private:
  bool layer_norm1_weight_loaded_ = false;
  bool layer_norm1_bias_loaded_ = false;
  bool layer_norm2_weight_loaded_ = false;
  bool layer_norm2_bias_loaded_ = false;

  torch::nn::LayerNorm layer_norm1_ = nullptr;
  torch::nn::LayerNorm layer_norm2_ = nullptr;
  SiglipAttention self_attn_ = nullptr;
  SiglipMLP mlp_ = nullptr;
};
TORCH_MODULE(SiglipEncoderLayer);

class SiglipEncoderImpl : public torch::nn::Module {
 public:
  explicit SiglipEncoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.mm_num_hidden_layers());
    for (int32_t i = 0; i < model_args.mm_num_hidden_layers(); i++) {
      auto block = SiglipEncoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  torch::Tensor forward(const torch::Tensor& embeddings) {
    bool output_hidden_states = false;
    bool output_attentions = false;
    std::vector<torch::Tensor> encoder_states;

    auto hidden_states = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
      encoder_states.emplace_back(hidden_states);
      auto& layer = layers_[i];
      hidden_states = layer->forward(hidden_states);
    }
    if (output_hidden_states) encoder_states.emplace_back(hidden_states);

    std::vector<torch::Tensor> outputs = {hidden_states};
    return outputs[0];
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  torch::nn::ModuleList blocks_ = nullptr;
  std::vector<SiglipEncoderLayer> layers_ = {};
};
TORCH_MODULE(SiglipEncoder);

// class SiglipMultiheadAttentionPoolingHeadImpl : public torch::nn::Module {
//  public:
//   explicit SiglipMultiheadAttentionPoolingHeadImpl(const ModelContext&
//   context) {
//     auto model_args = context.get_model_args();
//     auto options = context.get_tensor_options();

//     int64_t hidden_size = model_args.mm_hidden_size();
//     int64_t num_attention_heads = model_args.mm_num_attention_heads();
//     float layer_norm_eps = model_args.mm_layer_norm_eps();

//     probe_ = register_parameter(
//         "probe",
//         torch::randn({1, 1, hidden_size}, options));

//     attention_ = register_module(
//         "attention",
//         torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(hidden_size,
//             num_attention_heads)
//                 .batch_first(true)
//                 .bias(true)
//                 .dropout(0.0f)));

//     layernorm_ = register_module(
//         "layernorm",
//         torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({hidden_size})
//                 .elementwise_affine(true)
//                 .eps(layer_norm_eps)));
//     layernorm_->weight.set_data(layernorm_->weight.to(options));
//     layernorm_->bias.set_data(layernorm_->bias.to(options));

//     mlp_ = register_module("mlp", SiglipMLP(context));
//   }

//   torch::Tensor forward(const torch::Tensor& hidden_states) {
//     CHECK(hidden_states.dim() == 3)
//         << "hidden_states must be 3D tensor (batch_size, seq_len,
//         hidden_size)";
//     int64_t batch_size = hidden_states.size(0);

//     auto probe = probe_.repeat({batch_size, 1, 1});

//     auto attn_out = attention_->forward(probe, hidden_states, hidden_states);
//     auto attention_output = std::get<0>(attn_out);

//     auto residual = attention_output;
//     auto hidden_norm = layernorm_->forward(attention_output);
//     auto out = residual + mlp_->forward(hidden_norm);

//     return out.select(1, 0);
//   }

//   void load_state_dict(const StateDict& state_dict) {
//     const auto probe_tensor = state_dict.get_tensor("probe");
//     if (probe_tensor.defined()) {
//       CHECK_EQ(probe_tensor.sizes(), probe_.sizes())
//           << "probe weight size mismatch for " << name();
//       probe_.data().copy_(probe_tensor);
//       is_probe_loaded_ = true;
//     }

//     //
//     attention_->load_state_dict(state_dict.get_dict_with_prefix("attention."));
//     // 先加载 probe、layernorm、mlp（如之前）
// // const auto probe_tensor = state_dict.get_tensor("probe");
// // if (probe_tensor.defined()) {
// //   CHECK_EQ(probe_tensor.sizes(), probe_.sizes());
// //   probe_.data().copy_(probe_tensor);
// //   is_probe_loaded_ = true;
// // }

// // layernorm
//   // weight::load_weight(state_dict, "layernorm.weight", layernorm_->weight,
//   is_layernorm_weight_loaded_);
//   // weight::load_weight(state_dict, "layernorm.bias", layernorm_->bias,
//   is_layernorm_bias_loaded_);

//   // // mlp 已有 load_state_dict
//   // mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
//   // is_mlp_loaded_ = true;

//   // 手动加载 attention 的核心参数（示例）
//   {
//     const auto in_proj_w = state_dict.get_tensor("attention.in_proj_weight");
//     if (in_proj_w.defined()) {
//       // in_proj_weight 在 libtorch 的实现可能是一个参数 named
//       "in_proj_weight" auto params = attention_->named_parameters(); if
//       (params.contains("in_proj_weight")) {
//         params["in_proj_weight"].data().copy_(in_proj_w.to(params["in_proj_weight"].device()));
//         // 如果需要，标记加载标志
//       } else {
//         LOG(WARNING) << "MultiheadAttention has no named parameter
//         in_proj_weight";
//       }
//     }

//     const auto in_proj_b = state_dict.get_tensor("attention.in_proj_bias");
//     if (in_proj_b.defined()) {
//       auto params = attention_->named_parameters();
//       if (params.contains("in_proj_bias")) {
//         params["in_proj_bias"].data().copy_(in_proj_b.to(params["in_proj_bias"].device()));
//       }
//     }

//     const auto out_w = state_dict.get_tensor("attention.out_proj.weight");
//     if (out_w.defined()) {
//       // out_proj 是一个 Linear 子模块
//       if (attention_->out_proj) {
//         attention_->out_proj->weight.data().copy_(out_w.to(attention_->out_proj->weight.device()));
//       } else {
//         LOG(WARNING) << "MultiheadAttention has no out_proj module
//         accessible";
//       }
//     }

//     const auto out_b = state_dict.get_tensor("attention.out_proj.bias");
//     if (out_b.defined()) {
//       if (attention_->out_proj) {
//         attention_->out_proj->bias.data().copy_(out_b.to(attention_->out_proj->bias.device()));
//       }
//     }
//   }

//     is_attention_loaded_ = true;

//     weight::load_weight(
//         state_dict,
//         "layernorm.weight",
//         layernorm_->weight,
//         is_layernorm_weight_loaded_);
//     weight::load_weight(
//         state_dict,
//         "layernorm.bias",
//         layernorm_->bias,
//         is_layernorm_bias_loaded_);

//     mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
//     is_mlp_loaded_ = true;
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     CHECK(is_probe_loaded_)
//         << "weight is not loaded for " << prefix + "probe";
//     CHECK(is_attention_loaded_)
//         << "weight is not loaded for " << prefix + "attention.";
//     CHECK(is_layernorm_weight_loaded_)
//         << "weight is not loaded for " << prefix + "layernorm.weight";
//     CHECK(is_layernorm_bias_loaded_)
//         << "weight is not loaded for " << prefix + "layernorm.bias";
//     CHECK(is_mlp_loaded_)
//         << "weight is not loaded for " << prefix + "mlp.";
//     mlp_->verify_loaded_weights(prefix + "mlp.");
//   }

//  private:
//   torch::Tensor probe_;
//   torch::nn::MultiheadAttention attention_{nullptr};
//   torch::nn::LayerNorm layernorm_{nullptr};
//   SiglipMLP mlp_{nullptr};

//   bool is_probe_loaded_{false};
//   bool is_attention_loaded_{false};
//   bool is_layernorm_weight_loaded_{false};
//   bool is_layernorm_bias_loaded_{false};
//   bool is_mlp_loaded_{false};
// };
// TORCH_MODULE(SiglipMultiheadAttentionPoolingHead);

class SiglipVisionTransformerImpl : public torch::nn::Module {
 public:
  explicit SiglipVisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embeddings_ =
        register_module("embeddings", SiglipVisionEmbeddings(context));
    post_layer_norm_ = register_module(
        "post_layer_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({model_args.mm_hidden_size()})
                .elementwise_affine(true)
                .eps(model_args.mm_layer_norm_eps())));
    post_layer_norm_->weight.set_data(post_layer_norm_->weight.to(options));
    post_layer_norm_->bias.set_data(post_layer_norm_->bias.to(options));
    encoder_ = register_module("encoder", SiglipEncoder(context));
    // head_ = register_module("head",
    // SiglipMultiheadAttentionPoolingHead(context))
  }

  torch::Tensor forward(const torch::Tensor& pixel_values) {
    // if (!input_ids.defined()) {
    //   LOG(FATAL) << "input_ids is undefined.";
    // }
    // auto input_shape = input_ids.sizes();
    // auto reshaped_input_ids = input_ids.view({-1, input_shape.back()});
    // auto hidden_states = embeddings_->forward(reshaped_input_ids);
    auto hidden_states = embeddings_->forward(pixel_values);
    auto encoder_output = encoder_->forward(hidden_states);
    auto last_hidden_state = post_layer_norm_->forward(encoder_output);
    // auto pool_output
    return last_hidden_state;
  }

  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    weight::load_weight(state_dict,
                        "post_layernorm.weight",
                        post_layer_norm_->weight,
                        post_layer_norm_weight_loaded_);

    weight::load_weight(state_dict,
                        "post_layernorm.bias",
                        post_layer_norm_->bias,
                        post_layer_norm_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
    CHECK(post_layer_norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "post_layernorm.weight";
    CHECK(post_layer_norm_bias_loaded_)
        << "weight is not loaded for " << prefix + "post_layernorm.bias";
  }

 private:
  int64_t eos_token_id;
  bool post_layer_norm_weight_loaded_ = false;
  bool post_layer_norm_bias_loaded_ = false;
  SiglipVisionEmbeddings embeddings_ = nullptr;
  SiglipEncoder encoder_ = nullptr;
  torch::nn::LayerNorm post_layer_norm_ = nullptr;
};
TORCH_MODULE(SiglipVisionTransformer);

class SiglipVisionModelImpl : public torch::nn::Module {
 public:
  explicit SiglipVisionModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    eos_token_id = model_args.eos_token_id();
    transformer_ =
        register_module("transformer", SiglipVisionTransformer(context));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    auto last_hidden_states = transformer_->forward(input_ids);
    return last_hidden_states;
    // int64_t batch_size = last_hidden_states.size(0);
    // auto device = last_hidden_states.device();
    // torch::Tensor batch_indices = torch::arange(batch_size, device);
    // torch::Tensor end_pos;
    // if (eos_token_id == 2) {
    //   auto argmax_result = input_ids.to(device).max(1);
    //   end_pos = std::get<1>(argmax_result);
    // } else {
    //   torch::Tensor eos_mask =
    //       (input_ids == eos_token_id).to(device, torch::kInt);
    //   auto argmax_result = eos_mask.max(1);
    //   end_pos = std::get<1>(argmax_result);
    // }
    // torch::Tensor pooled_output =
    //     last_hidden_states.index({batch_indices, end_pos});
    // pooled_output = head_->forward(pooled_output);
    // return pooled_output;
  }

  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(
        state_dict.get_dict_with_prefix("vision_model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    transformer_->verify_loaded_weights(prefix + ".");
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    LOG(INFO) << "Loading SiglipVisionModel from ModelLoader...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      transformer_->load_state_dict(
          state_dict->get_dict_with_prefix("vision_model."));
    }

    transformer_->verify_loaded_weights("vision_model.");
    LOG(INFO) << "SiglipVisionModel loaded successfully.";
  }

 private:
  int64_t eos_token_id;
  SiglipVisionTransformer transformer_ = nullptr;
};
TORCH_MODULE(SiglipVisionModel);

REGISTER_MODEL_ARGS(SiglipVisionModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(mm_hidden_size, "hidden_size", 1152);
  LOAD_ARG_OR(mm_image_size, "image_size", 384);
  LOAD_ARG_OR(mm_intermediate_size, "intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_hidden_layers, "num_hidden_layers", 27);
  LOAD_ARG_OR(mm_num_attention_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "num_channels", 3);
  LOAD_ARG_OR(mm_patch_size, "patch_size", 14);
  LOAD_ARG_OR(mm_hidden_act, "hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_layer_norm_eps, "layer_norm_eps", 1e-6f);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(mm_head_dim, "head_dim", 64);
});

}  // namespace xllm
