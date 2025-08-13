#pragma once

#include "scheduler/continuous_scheduler.h"

namespace xllm {

struct SequenceStatus {
  SequenceStatus(uint32_t need_to_use, uint32_t release_block)
      : num_block_need_to_use_(need_to_use),
        num_release_block_(release_block) {}

  void print() const {
    LOG(INFO) << "SequenceStatus { "
              << "need_to_use: " << num_block_need_to_use_ << ", "
              << "release: " << num_release_block_ << " }";
  }

  uint32_t num_block_need_to_use_;
  uint32_t num_release_block_;
};

template <typename Func>
class ResourceGuard {
 public:
  explicit ResourceGuard(Func&& release_func)
      : release_func_(std::forward<Func>(release_func)) {}

  ~ResourceGuard() {
    if (is_necessory_release_) {
      release_func_();
    }
  }

  void success() { is_necessory_release_ = false; }

 private:
  bool is_necessory_release_ = true;
  Func release_func_;
};

class BlockCapacityGuard {
 public:
  BlockCapacityGuard(BlockManagerPool* block_manager);

  bool if_accept_candidate_sequences(
      const std::vector<Sequence*>& candidate_sequences,
      const std::deque<std::shared_ptr<Request>>& running_queue,
      const std::vector<Sequence*>& running_sequences);

 private:
  uint32_t block_size() const {
    const auto& option = block_manager_->options();
    return option.block_size();
  }

  uint32_t num_block() const {
    const auto& option = block_manager_->options();
    return option.num_blocks();
  }

  bool is_prefix_cache() const {
    const auto& option = block_manager_->options();
    return option.enable_prefix_cache();
  }

  uint32_t num_blocks_in_useless() const {
    // TODO for mutil dp
    return num_block() - block_manager_->num_used_blocks()[0];
  }

  bool simulate_is_satisfied_for_candidate_sequences();

  void compute_reserved_block_num();

  void prefix_cache_for_candidate_sequences();

  uint32_t get_needed_block_num_for_prefill();

  std::vector<SequenceStatus> get_all_sequence_status();

  uint32_t num_block_need_to_use_for(const Sequence* sequence);

  uint32_t num_release_block_for(Sequence* sequence);

  std::vector<SequenceStatus> get_running_sequence_status();

 private:
  BlockManagerPool* block_manager_;

  std::vector<Sequence*> candidate_sequences_;
  std::vector<Sequence*> running_queue_;
  std::vector<Sequence*> running_sequences_;

  uint32_t num_reserved_block_for_prefill_;
};

class ZeroEvictionScheduler final : public ContinuousScheduler {
 public:
  ZeroEvictionScheduler(Engine* engine, const Options& options);
  virtual ~ZeroEvictionScheduler();

  std::vector<Batch> prepare_batch_test() { return prepare_batch(); }

  uint32_t get_waiting_requests_num() const override {
    return waiting_priority_queue_.size();
  };

 private:
  void handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests) override;

  // build a batch of requests from the priority queue
  std::vector<Batch> prepare_batch() override;

  bool try_allocate_block_for(std::shared_ptr<Request> request,
                              std::vector<Sequence*>* prefill_sequences,
                              std::vector<size_t>* prefill_sequences_budget,
                              size_t* allocated_tokens,
                              size_t* allocated_seqs,
                              size_t remaining_token_budget,
                              size_t remaining_seq_budget);

  // is last step handle prefill requests
  bool last_step_prefill_ = false;

  std::unique_ptr<BlockCapacityGuard> block_capacity_guard_;

  bool is_satisfied_for_prefill = true;
};

}  // namespace xllm
