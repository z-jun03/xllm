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

// #include "taylorseer.h"

// namespace xllm {

// namespace {
// double factorial(int k) { return std::tgamma(static_cast<double>(k) + 1.0); }

// // Hermite polynomial (physicists’ version)
// double hermite_poly(double x, int n) {
//     if (n == 0) return 1.0;
//     if (n == 1) return 2.0 * x;

//     double H_nm2 = 1.0;        // H_0
//     double H_nm1 = 2.0 * x;    // H_1
//     double H_n = 0.0;

//     for (int k = 2; k <= n; ++k) {
//         H_n = 2.0 * x * H_nm1 - 2.0 * (k - 1) * H_nm2;
//         H_nm2 = H_nm1;
//         H_nm1 = H_n;
//     }

//     return H_n;
// }

// // Scaled Hermite polynomial
// double scaled_hermite_poly(double x, int n, double sigma) {
//     double x_scaled = sigma * x;
//     double Hn = hermite_poly(x_scaled, n);
//     return std::pow(sigma, static_cast<double>(n)) * Hn;
// }

// }  // namespace

// void TaylorSeer::init(const DiTCacheConfig& cfg) {
//   CHECK_GT(cfg.taylorseer.num_inference_steps, 0)
//       << "num_inference_steps must be > 0";
//   CHECK_GE(cfg.taylorseer.warmup_steps, 0) << "warmup_steps must be >= 0";
//   CHECK_GE(cfg.taylorseer.skip_interval_steps, 0)
//       << "skip_interval_steps must be >= 0";
//   CHECK_GE(cfg.taylorseer.n_derivatives, 0) << "n_derivatives must be >= 0";

//   CHECK_LT(cfg.taylorseer.skip_interval_steps,
//            cfg.taylorseer.num_inference_steps)
//       << "skip_interval_steps must be less than num_inference_steps";
//   CHECK_LT(cfg.taylorseer.warmup_steps, cfg.taylorseer.num_inference_steps)
//       << "warmup_steps must be less than num_inference_steps";

//   num_inference_steps_ = cfg.taylorseer.num_inference_steps;
//   warmup_steps_ = cfg.taylorseer.warmup_steps;
//   skip_interval_steps_ = cfg.taylorseer.skip_interval_steps;
//   n_derivatives_ = cfg.taylorseer.n_derivatives;
//   order_ = n_derivatives_ + 1;

//   // 如果 cfg 中有这些超参，可以从 cfg 读取；否则使用类的默认值。
//   // 例如 (伪代码):
//   // alpha_ = cfg.taylorseer.alpha; ...
//   // 这里先保持默认值（可在头文件中修改默认）。

//   reset_cache();
// }

// void TaylorSeer::reset_cache() {
//   dY_prev_.assign(order_, torch::Tensor());
//   dY_current_.assign(order_, torch::Tensor());
//   dY_smoothed_.assign(order_, torch::Tensor()); // EMA 平滑容器
//   valid_prev_.assign(order_, false);
//   valid_current_.assign(order_, false);
//   current_step_ = 1;
//   last_non_approximated_step_ = 1;
//   use_cache_ = false;
// }

// std::pair<std::vector<torch::Tensor>, std::vector<bool>>
// TaylorSeer::approximate_derivative(const torch::Tensor& Y) {
//   std::vector<torch::Tensor> dY(order_);
//   std::vector<bool> valid(order_, false);

//   dY[0] = Y;
//   valid[0] = true;

//   int elapsed_steps = current_step_ - last_non_approximated_step_;
//   if (elapsed_steps <= 0) return {dY, valid};

//   for (int i = 0; i < n_derivatives_; ++i) {
//     if (i >= static_cast<int>(valid_prev_.size()) || !valid_prev_[i] ||
//         !dY_prev_[i].defined()) {
//       break;
//     }
//     dY[i + 1] = (dY[i] - dY_prev_[i]) / static_cast<double>(elapsed_steps);
//     valid[i + 1] = true;
//   }

//   return {dY, valid};
// }

// // torch::Tensor TaylorSeer::approximate_value() {
// //   if (!dY_smoothed_[0].defined()) return torch::Tensor(); //
// 使用平滑后的导数
// //   int elapsed_steps = current_step_ - last_non_approximated_step_;
// //   if (elapsed_steps < 0) return torch::Tensor();

// //   // 归一化时间尺度，避免 t^i 数值爆炸
// //   double t = static_cast<double>(elapsed_steps) * alpha_;

// //   torch::Tensor output = torch::zeros_like(dY_smoothed_[0]);
// //   for (int i = 0; i < order_; ++i) {
// //     if (!valid_current_[i]) break;
// //     // 标准泰勒系数：t^i / i!，并加上对高阶的阻尼
// //     double coef = std::pow(t, i) / factorial(i);
// //     coef *= std::pow(damping_, i);

// //     // 可选的截断，防止极端系数
// //     const double kMaxCoef = 1e3;
// //     if (coef > kMaxCoef) coef = kMaxCoef;
// //     if (coef < -kMaxCoef) coef = -kMaxCoef;

// //     output += dY_smoothed_[i] * coef;
// //   }
// //   return output;
// // }

// torch::Tensor TaylorSeer::approximate_value() {
//   if (!dY_current_[0].defined()) return torch::Tensor();

//   int elapsed_steps = current_step_ - last_non_approximated_step_;
//   if (elapsed_steps < 0) return torch::Tensor();

//   torch::Tensor output = torch::zeros_like(dY_smoothed_[0]);
//   for (int i = 0; i < order_; ++i) {
//     if (!valid_current_[i]) break;
//     double coef =
//         std::pow(static_cast<double>(elapsed_steps), i) / factorial(i+1);
//         //有问题，
//     // double coef = scaled_hermite_poly(elapsed_steps, i ,0.5) /
//     factorial(i); output += dY_smoothed_[i] * coef;
//   }
//   return output;
// }

// //
// void TaylorSeer::update(const torch::Tensor& Y) {
//   dY_prev_ = dY_current_;
//   valid_prev_ = valid_current_;

//   if (!dY_current_[0].defined()) {
//     for (int i = 0; i < order_; ++i) {
//       dY_current_[i] = (i == 0) ? Y : torch::zeros_like(Y);
//       valid_current_[i] = (i == 0);
//       // 初始化平滑张量与当前保持一致（避免未定义）
//       dY_smoothed_[i] = dY_current_[i].defined() ? dY_current_[i].clone()
//                                                 : torch::Tensor();
//     }
//   } else {
//     auto [new_dY, new_valid] = approximate_derivative(Y);
//     dY_current_ = std::move(new_dY);
//     valid_current_ = std::move(new_valid);
//     // 对每一阶导数做 EMA 平滑（只有在 new_valid[i] 为 true 时才更新）
//     for (int i = 0; i < order_; ++i) {
//       if (!valid_current_[i]) {
//         // 若当前无效，保持原来的平滑值或设为未定义
//         continue;
//       }
//       if (!dY_smoothed_[i].defined()) {
//         // 第一次赋值时直接设为当前估计
//         dY_smoothed_[i] = dY_current_[i].clone();
//       } else {
//         // EMA: s = beta * s + (1 - beta) * x
//         if (i==0)
//           dY_smoothed_[i] = dY_current_[i].clone();
//         else
//         dY_smoothed_[i].mul_(ema_beta_).add_(dY_current_[i] * (1.0 -
//         ema_beta_));
//       }
//     }
//   }
//   last_non_approximated_step_ = current_step_;
// }

// // void TaylorSeer::init(const DiTCacheConfig& cfg) {
// //   CHECK_GT(cfg.taylorseer.num_inference_steps, 0)
// //       << "num_inference_steps must be > 0";
// //   CHECK_GE(cfg.taylorseer.warmup_steps, 0) << "warmup_steps must be >= 0";
// //   CHECK_GE(cfg.taylorseer.skip_interval_steps, 0)
// //       << "skip_interval_steps must be >= 0";
// //   CHECK_GE(cfg.taylorseer.n_derivatives, 0) << "n_derivatives must be >=
// 0";

// //   CHECK_LT(cfg.taylorseer.skip_interval_steps,
// //            cfg.taylorseer.num_inference_steps)
// //       << "skip_interval_steps must be less than num_inference_steps";
// //   CHECK_LT(cfg.taylorseer.warmup_steps,
// cfg.taylorseer.num_inference_steps)
// //       << "warmup_steps must be less than num_inference_steps";

// //   num_inference_steps_ = cfg.taylorseer.num_inference_steps;
// //   warmup_steps_ = cfg.taylorseer.warmup_steps;
// //   skip_interval_steps_ = cfg.taylorseer.skip_interval_steps;
// //   n_derivatives_ = cfg.taylorseer.n_derivatives;
// //   order_ = n_derivatives_ + 1;
// //   reset_cache();
// // }

// // void TaylorSeer::reset_cache() {
// //   dY_prev_.assign(order_, torch::Tensor());
// //   dY_current_.assign(order_, torch::Tensor());
// //   valid_prev_.assign(order_, false);
// //   valid_current_.assign(order_, false);
// //   current_step_ = 1;
// //   last_non_approximated_step_ = 1;
// //   use_cache_ = false;
// // }

// void TaylorSeer::mark_step_begin() { ++current_step_; }

// // std::pair<std::vector<torch::Tensor>, std::vector<bool>>
// // TaylorSeer::approximate_derivative(const torch::Tensor& Y) {
// //   std::vector<torch::Tensor> dY(order_);
// //   std::vector<bool> valid(order_, false);

// //   dY[0] = Y;
// //   valid[0] = true;

// //   int elapsed_steps = current_step_ - last_non_approximated_step_;
// //   if (elapsed_steps <= 0) return {dY, valid};

// //   for (int i = 0; i < n_derivatives_; ++i) {
// //     if (i >= static_cast<int>(valid_prev_.size()) || !valid_prev_[i] ||
// //         !dY_prev_[i].defined()) {
// //       break;
// //     }
// //     dY[i + 1] = (dY[i] - dY_prev_[i]) /
// static_cast<double>(elapsed_steps);
// //     valid[i + 1] = true;
// //   }

// //   return {dY, valid};
// // }

// // torch::Tensor TaylorSeer::approximate_value() {
// //   if (!dY_current_[0].defined()) return torch::Tensor();

// //   int elapsed_steps = current_step_ - last_non_approximated_step_;
// //   if (elapsed_steps < 0) return torch::Tensor();

// //   torch::Tensor output = torch::zeros_like(dY_current_[0]);
// //   for (int i = 0; i < order_; ++i) {
// //     if (!valid_current_[i]) break;
// //     // double coef =
// //         // std::pow(static_cast<double>(elapsed_steps), i) / factorial(i);
// //有问题，
// //     double coef = scaled_hermite_poly(elapsed_steps, i ,0.5) /
// factorial(i);
// //     output += dY_current_[i] * coef;
// //   }
// //   return output;
// // }

// // void TaylorSeer::update(const torch::Tensor& Y) {
// //   dY_prev_ = dY_current_;
// //   valid_prev_ = valid_current_;

// //   if (!dY_current_[0].defined()) {
// //     for (int i = 0; i < order_; ++i) {
// //       dY_current_[i] = (i == 0) ? Y : torch::zeros_like(Y);
// //       valid_current_[i] = (i == 0);
// //     }
// //   } else {
// //     auto [new_dY, new_valid] = approximate_derivative(Y);
// //     dY_current_ = std::move(new_dY);
// //     valid_current_ = std::move(new_valid);
// //   }
// //   last_non_approximated_step_ = current_step_;
// // }

// bool TaylorSeer::on_before_block(const CacheBlockIn&) { return false; }

// CacheBlockOut TaylorSeer::on_after_block(const CacheBlockIn& blockin) {
//   TensorMap out_map;
//   out_map["hidden_states"] =
//       get_tensor_or_empty(blockin.tensors, "hidden_states");
//   out_map["encoder_hidden_states"] =
//       get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
//   return CacheBlockOut(out_map);
// }

// bool TaylorSeer::on_before_step(const CacheStepIn& stepin) {
//   current_step_ = stepin.step_id;

//   if (current_step_ == 1 || current_step_ <= warmup_steps_) {
//     if (current_step_ == 1) reset_cache();
//     use_cache_ = false;
//     return use_cache_;
//   }

//   use_cache_ =
//       ((current_step_ - warmup_steps_ + 1) % skip_interval_steps_ != 0);
//   return use_cache_;
// }

// CacheStepOut TaylorSeer::on_after_step(const CacheStepIn& stepin) {
//   if (!use_cache_) {
//     update(stepin.tensors.at("hidden_states"));
//     return CacheStepOut(stepin.tensors);
//   }

//   TensorMap result{{"hidden_states", approximate_value()}};
//   update(result.at("hidden_states"));
//   return CacheStepOut(result);
// }

// }  // namespace xllm
