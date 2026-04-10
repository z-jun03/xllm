/* Copyright 2025 The xLLM Authors. All Rights Reserved.
... (Apache 2.0 License header) ...
==============================================================================*/

#include "dicache.h"

namespace xllm {

void DiCache::init(const DiTCacheConfig& cfg) {
  probe_depth_ = cfg.dicache.probe_depth;
  rel_l1_thresh_ = cfg.dicache.rel_l1_thresh;
  ret_ratio_ = cfg.dicache.ret_ratio;
  reset_cache();
}

void DiCache::reset_cache() {
  current_step_ = 0;
  accumulated_error_ = 0.0;
  force_full_calc_ = false;
  // approximate_final_hidden_ = torch::Tensor();

  previous_input_ = torch::Tensor();
  previous_probe_states_ = torch::Tensor();
  previous_residual_ = torch::Tensor();
  previous_probe_residual_ = torch::Tensor();
  residual_window_.clear();
  probe_residual_window_.clear();

  original_hidden_states_ = torch::Tensor();
  probe_hidden_states_ = torch::Tensor();
  current_block_id_ = 0;
  use_cache_ = false;
}

double DiCache::relative_l1_distance(const torch::Tensor& lsh,
                                     const torch::Tensor& rhs) {
  if (!lsh.defined() || !rhs.defined()) return 0.0;
  auto diff = (lsh - rhs).abs().mean();
  auto base = rhs.abs().mean();
  if (base.item<double>() == 0.0) return 0.0;
  return (diff / base).item<double>();
}

torch::Tensor DiCache::decide_and_prepare_skip(
    const torch::Tensor& hidden_states) {
  if (!probe_hidden_states_.defined() || !previous_input_.defined() ||
      !previous_probe_states_.defined()) {
    use_cache_ = false;
    // previous_probe_states_ = std::move(probe_hidden_states_);//.clone();
    // previous_input_ = std::move(original_hidden_states_);//.clone();
    // previous_probe_states_ = probe_hidden_states_.clone();
    // previous_input_ = original_hidden_states_.clone();
    // if (probe_hidden_states_.defined()) {
    previous_probe_states_ = probe_hidden_states_.clone();
    // }
    // if (original_hidden_states_.defined()) {
    previous_input_ = original_hidden_states_.clone();
    // }

    return hidden_states;
  }

  // torch::Tensor approx = std::move(hidden_states);//.clone();
  // torch::Tensor approx;
  double error =
      relative_l1_distance(probe_hidden_states_, previous_probe_states_);
  accumulated_error_ += error;
  if (accumulated_error_ < rel_l1_thresh_) {
    use_cache_ = true;
    // approx = original_hidden_states_.clone();
    // approx = std::move(original_hidden_states_);//.clone();
    torch::Tensor approx = original_hidden_states_.clone();
    if (residual_window_.size() >= 2) {
      torch::Tensor current_residual_indicator =
          probe_hidden_states_ - original_hidden_states_;
      torch::Tensor denom =
          probe_residual_window_.back() - probe_residual_window_.front();
      auto gamma =
          ((current_residual_indicator - probe_residual_window_.front())
               .abs()
               .mean() /
           denom.abs().mean())
              .clip(1.0, 1.5);
      approx += residual_window_.front() +
                gamma * (residual_window_.back() - residual_window_.front());
    } else if (!previous_residual_.defined()) {
      approx += previous_residual_;
    }
    previous_probe_states_ = std::move(probe_hidden_states_);
    previous_input_ = original_hidden_states_.clone();
    return approx;
    // approximate_final_hidden_ = approx;
  } else {
    use_cache_ = false;
    accumulated_error_ = 0.0;
    // approx = hidden_states;
    previous_probe_states_ = std::move(probe_hidden_states_);
    previous_input_ = original_hidden_states_.clone();
    return hidden_states;
  }

  // previous_probe_states_ = std::move(probe_hidden_states_);
  // previous_input_ = std::move(original_hidden_states_);
  // previous_input_ = original_hidden_states_.clone();
  // return approx;
}

bool DiCache::on_before_step(const CacheStepIn& stepin) {
  current_step_ = stepin.step_id;
  if (current_step_ == 1) {
    reset_cache();
  }
  use_cache_ = false;
  int warmup_end = static_cast<int>(ret_ratio_ * num_steps_);
  bool is_last_step = (current_step_ == num_steps_);
  if (current_step_ <= warmup_end || is_last_step) {
    force_full_calc_ = true;
    accumulated_error_ = 0.0;
  } else {
    force_full_calc_ = false;
  }

  original_hidden_states_ =
      std::move(get_tensor_or_empty(stepin.tensors, "hidden_states"));

  current_block_id_ = 0;
  return false;
}

CacheStepOut DiCache::on_after_step(const CacheStepIn& stepin) {
  if (!use_cache_) {
    auto hidden_states = get_tensor_or_empty(stepin.tensors, "hidden_states");
    if (original_hidden_states_.defined() && hidden_states.defined()) {
      torch::Tensor residual = hidden_states - original_hidden_states_;
      // previous_residual_ = std::move(residual);//.clone();
      // residual_window_.push_back(std::move(residual));
      previous_residual_ = residual.clone();
      residual_window_.push_back(residual.clone());
      if (residual_window_.size() > 2) residual_window_.pop_front();

      if (probe_hidden_states_.defined()) {
        torch::Tensor probe_residual =
            probe_hidden_states_ - original_hidden_states_;
        previous_probe_residual_ = probe_residual.clone();
        probe_residual_window_.push_back(probe_residual.clone());
        if (probe_residual_window_.size() > 2)
          probe_residual_window_.pop_front();
      }
    }
    previous_input_ = std::move(original_hidden_states_);
  }

  TensorMap out_map;
  auto hidden_states = get_tensor_or_empty(stepin.tensors, "hidden_states");
  if (hidden_states.defined()) {
    out_map["hidden_states"] = std::move(hidden_states);
  }
  return CacheStepOut(out_map);
}

bool DiCache::on_before_block(const CacheBlockIn& blockin) {
  current_block_id_ = blockin.block_id;
  LOG(INFO) << "on_before_block: step " << current_step_ << ", block "
            << current_block_id_;

  if (use_cache_) {
    LOG(INFO) << "skip step " << current_step_ << " block " << current_block_id_
              << " due to previous decision, returning approximate final "
                 "hidden states";
    return true;
  }

  if (current_block_id_ < probe_depth_) return false;
  return false;
}

CacheBlockOut DiCache::on_after_block(const CacheBlockIn& blockin) {
  auto hidden_states = get_tensor_or_empty(blockin.tensors, "hidden_states");
  auto encoder_hidden_states =
      get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");

  TensorMap out_map;
  if (use_cache_) {
    out_map["hidden_states"] =
        get_tensor_or_empty(blockin.tensors, "hidden_states");
    out_map["encoder_hidden_states"] =
        get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
    return CacheBlockOut(out_map);
  }

  if (current_block_id_ < probe_depth_ && encoder_hidden_states.defined()) {
    if (current_block_id_ == probe_depth_ - 1) {
      probe_hidden_states_ = hidden_states.clone();
    }

    if (current_block_id_ == probe_depth_ - 1 && !force_full_calc_) {
      // TensorMap out_map;
      LOG(INFO) << "force_full_calc_";
      out_map["hidden_states"] = decide_and_prepare_skip(hidden_states);
      out_map["encoder_hidden_states"] = encoder_hidden_states;
      return CacheBlockOut(out_map);
    }
  }

  out_map["hidden_states"] = hidden_states;
  out_map["encoder_hidden_states"] = encoder_hidden_states;
  return CacheBlockOut(out_map);
}

}  // namespace xllm

// dicache.cpp

// #include "dicache.h"

// namespace xllm {

// void DiCache::init(const DiTCacheConfig& cfg) {
//     probe_depth_ = cfg.dicache.probe_depth;
//     rel_l1_thresh_ = cfg.dicache.rel_l1_thresh;
//     ret_ratio_ = cfg.dicache.ret_ratio;
//     num_steps_ = cfg.num_steps;
//     reset();
// }

// void DiCache::reset() {
//     current_step_ = 0;
//     accumulated_error_ = 0.0;
//     force_full_calc_ = false;
//     use_cache_ = false;
//     previous_input_ = torch::Tensor();
//     previous_probe_states_ = torch::Tensor();
//     previous_residual_ = torch::Tensor();
//     residual_window_.clear();
//     probe_residual_window_.clear();
//     original_hidden_states_ = torch::Tensor();
//     probe_hidden_states_ = torch::Tensor();
//     current_block_id_ = 0;
// }

// double DiCache::relative_l1_distance(const torch::Tensor& a, const
// torch::Tensor& b) {
//     if (!a.defined() || !b.defined()) return 0.0;
//     auto diff = (a - b).abs().mean();
//     auto base = b.abs().mean();
//     if (base.item<double>() == 0.0) return 0.0;
//     return (diff / base).item<double>();
// }

// torch::Tensor DiCache::decide_and_prepare_skip(const torch::Tensor&
// hidden_states) {
//     // 历史数据不足 -> 不能跳过，但更新历史
//     if (!probe_hidden_states_.defined() || !previous_input_.defined() ||
//     !previous_probe_states_.defined()) {
//         use_cache_ = false;
//         previous_probe_states_ = probe_hidden_states_.clone();   //
//         深拷贝保存 previous_input_ = original_hidden_states_.clone(); return
//         hidden_states;
//     }

//     double delta_y = relative_l1_distance(probe_hidden_states_,
//     previous_probe_states_); accumulated_error_ += delta_y;

//     if (accumulated_error_ < rel_l1_thresh_) {
//         // 跳过剩余块，计算外推值
//         use_cache_ = true;
//         torch::Tensor approx = original_hidden_states_.clone();

//         if (residual_window_.size() >= 2) {
//             torch::Tensor current_residual_indicator = probe_hidden_states_ -
//             original_hidden_states_; torch::Tensor denom =
//             probe_residual_window_.back() - probe_residual_window_.front();
//             double gamma = ((current_residual_indicator -
//             probe_residual_window_.front()).abs().mean() /
//                             denom.abs().mean()).item<double>();
//             gamma = std::clamp(gamma, 1.0, 1.5);
//             approx += residual_window_.front() + gamma *
//             (residual_window_.back() - residual_window_.front());
//         } else if (previous_residual_.defined()) {
//             approx += previous_residual_;
//         }

//         // 更新历史（使用 move 转移所有权，避免拷贝）
//         previous_probe_states_ = std::move(probe_hidden_states_);
//         previous_input_ = original_hidden_states_.clone();
//         return approx;
//     } else {
//         // 误差过大，不跳过，清零累积误差
//         use_cache_ = false;
//         accumulated_error_ = 0.0;
//         // 更新历史
//         previous_probe_states_ = std::move(probe_hidden_states_);
//         previous_input_ = original_hidden_states_.clone();
//         return hidden_states;
//     }
// }

// void DiCache::on_step_begin(int step_id, const torch::Tensor& hidden_states)
// {
//     current_step_ = step_id;
//     if (current_step_ == 1) reset();
//     use_cache_ = false;

//     int warmup_end = static_cast<int>(ret_ratio_ * num_steps_);
//     bool is_last_step = (current_step_ == num_steps_);
//     if (current_step_ <= warmup_end || is_last_step) {
//         force_full_calc_ = true;
//         accumulated_error_ = 0.0;
//     } else {
//         force_full_calc_ = false;
//     }

//     // 保存当前步输入的副本（因外部可能修改原 tensor）
//     original_hidden_states_ = hidden_states.clone();
//     current_block_id_ = 0;
// }

// bool DiCache::on_block_begin(int block_id) {
//     current_block_id_ = block_id;
//     if (use_cache_) return true;   // 跳过所有剩余块
//     return false;
// }

// torch::Tensor DiCache::on_block_end(int block_id,
//                                     const torch::Tensor& hidden_states,
//                                     const torch::Tensor&
//                                     /*encoder_hidden_states*/) {
//     if (use_cache_) {
//         // 不应发生，因为 on_block_begin 已返回 true
//         跳过计算；但为了安全返回原值 return hidden_states;
//     }

//     if (block_id < probe_depth_) {
//         if (block_id == probe_depth_ - 1) {
//             // 保存探针输出
//             probe_hidden_states_ = hidden_states.clone();
//         }
//         if (block_id == probe_depth_ - 1 && !force_full_calc_) {
//             // 在最后一个探针块后做决策
//             return decide_and_prepare_skip(hidden_states);
//         }
//     }
//     return hidden_states;
// }

// torch::Tensor DiCache::on_step_end(const torch::Tensor& final_hidden_states)
// {
//     if (use_cache_) {
//         // 跳过步：无需更新残差窗口，直接返回外推值（已经由
//         decide_and_prepare_skip 返回） return final_hidden_states;
//     }

//     // 正常步：更新残差窗口
//     if (original_hidden_states_.defined() && final_hidden_states.defined()) {
//         torch::Tensor residual = final_hidden_states -
//         original_hidden_states_; previous_residual_ = residual.clone();
//         residual_window_.push_back(residual.clone());
//         if (residual_window_.size() > 2) residual_window_.pop_front();

//         if (probe_hidden_states_.defined()) {
//             torch::Tensor probe_residual = probe_hidden_states_ -
//             original_hidden_states_;
//             probe_residual_window_.push_back(probe_residual.clone());
//             if (probe_residual_window_.size() > 2)
//             probe_residual_window_.pop_front();
//         }
//     }
//     previous_input_ = original_hidden_states_.clone();
//     return final_hidden_states;
// }

// } // namespace xllm