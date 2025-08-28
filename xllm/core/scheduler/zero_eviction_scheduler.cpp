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

#include "scheduler/zero_eviction_scheduler.h"

#include "common/metrics.h"
#include "framework/batch/batch_factory.h"
#include "util/timer.h"
#include "util/utils.h"

DEFINE_int32(max_decode_token_per_sequence,
             200,
             "max decode token per sequence");

namespace xllm {

namespace {

uint32_t ceiling_div(uint32_t left, uint32_t right) {
  return (left + right - 1) / right;
}

std::vector<Sequence*> get_running_sequences(
    const std::deque<std::shared_ptr<Request>>& running_queue) {
  std::vector<Sequence*> running_sequences;

  for (auto& running_request : running_queue) {
    if (Request* request = running_request.get()) {
      for (auto& sequence : request->sequences()) {
        // skip finished sequence.
        if (sequence->finished()) {
          continue;
        }
        running_sequences.emplace_back(sequence.get());
      }
    }
  }
  return running_sequences;
}

void update_request_status_each_step(
    std::vector<SequenceStatus>* sequence_status_vec,
    uint32_t* num_remaining_block_after_prefill) {
  for (auto it = sequence_status_vec->begin();
       it != sequence_status_vec->end();) {
    if (it->num_block_need_to_use_ == 0) {
      *num_remaining_block_after_prefill += it->num_release_block_;
      it = sequence_status_vec->erase(it);

    } else {
      it->num_block_need_to_use_--;
      it->num_release_block_++;
      ++it;
    }
  }
}

template <typename T, typename... Args>
std::vector<T> merge_vec(const std::vector<T>& first, const Args&... rest) {
  std::vector<T> result;

  result.reserve((first.size() + ... + rest.size()));

  result.insert(result.end(), first.begin(), first.end());
  (result.insert(result.end(), rest.begin(), rest.end()), ...);

  return result;
}

uint32_t get_max_iter_num(const std::vector<SequenceStatus>& sequence_status) {
  CHECK(!sequence_status.empty());
  uint32_t max_iter_num = sequence_status[0].num_block_need_to_use_;
  for (std::size_t i = 1; i < sequence_status.size(); ++i) {
    max_iter_num = sequence_status[i].num_block_need_to_use_ > max_iter_num
                       ? sequence_status[i].num_block_need_to_use_
                       : max_iter_num;
  }
  return max_iter_num;
}

uint32_t get_num_reserved_full_blocks(
    const std::vector<SequenceStatus>& sequence_status) {
  uint32_t reserved_full_blocks = 0;
  for (const auto& status : sequence_status) {
    reserved_full_blocks += status.num_block_need_to_use_;
  }
  return reserved_full_blocks;
}

template <typename Func>
auto resource_guard(Func&& func) {
  return ResourceGuard<Func>(std::forward<Func>(func));
}

}  // namespace

BlockCapacityGuard::BlockCapacityGuard(BlockManagerPool* block_manager) {
  block_manager_ = block_manager;
}

void BlockCapacityGuard::compute_reserved_block_num() {
  CHECK_GT(block_size(), 0);

  for (const auto& sequence : candidate_sequences_) {
    uint32_t num_prefill_block =
        ceiling_div(sequence->num_tokens(), block_size());

    num_reserved_block_for_prefill_ += num_prefill_block;
  }
}

void BlockCapacityGuard::prefix_cache_for_candidate_sequences() {
  if (is_prefix_cache()) {
    for (auto* sequence : candidate_sequences_) {
      block_manager_->allocate_shared(sequence);
    }
  }
}

uint32_t BlockCapacityGuard::get_needed_block_num_for_prefill() {
  if (is_prefix_cache()) {
    return num_reserved_block_for_prefill_;
  }

  uint32_t total_num_prefix_cache_block = 0;
  for (auto* sequence : candidate_sequences_) {
    uint32_t num_prefix_cache_block = sequence->kv_state().num_kv_blocks();
    total_num_prefix_cache_block += num_prefix_cache_block;
  }
  CHECK_GE(num_reserved_block_for_prefill_, total_num_prefix_cache_block);
  return num_reserved_block_for_prefill_ - total_num_prefix_cache_block;
}

std::vector<SequenceStatus> BlockCapacityGuard::get_all_sequence_status() {
  std::vector<Sequence*> running_sequences_for_test =
      merge_vec(running_sequences_, candidate_sequences_, running_queue_);

  std::vector<SequenceStatus> result;
  result.reserve(running_sequences_for_test.size());

  for (auto* sequence : running_sequences_for_test) {
    uint32_t num_block_need_to_use = num_block_need_to_use_for(sequence);
    uint32_t num_release_block = num_release_block_for(sequence);

    result.emplace_back(num_block_need_to_use, num_release_block);
  }
  return result;
}

uint32_t BlockCapacityGuard::num_block_need_to_use_for(
    const Sequence* sequence) {
  uint32_t remaining_decode_token_num =
      FLAGS_max_decode_token_per_sequence -
      (sequence->num_tokens() - sequence->num_prompt_tokens());
  uint32_t remaining_decode_block_num =
      ceiling_div(remaining_decode_token_num, block_size());
  return remaining_decode_block_num;
}

uint32_t BlockCapacityGuard::num_release_block_for(Sequence* sequence) {
  uint32_t num_tokens_block = ceiling_div(sequence->num_tokens(), block_size());
  uint32_t num_release_block =
      num_tokens_block - sequence->kv_state().num_kv_blocks();
  CHECK_GE(num_release_block, 0);

  Slice<Block> blocks = sequence->kv_state().kv_blocks();

  for (std::size_t i = 0; i < blocks.size(); ++i) {
    uint32_t ref_count = blocks[i].ref_count();

    if (ref_count <= 2) {
      num_release_block += blocks.size() - i;
      break;
    }
  }

  return num_release_block;
}

bool BlockCapacityGuard::simulate_is_satisfied_for_candidate_sequences() {
  prefix_cache_for_candidate_sequences();

  uint32_t num_needed_block_for_prefill = get_needed_block_num_for_prefill();

  CHECK_GE(num_needed_block_for_prefill, 0);

  uint32_t num_remaining_block_after_prefill =
      num_blocks_in_useless() - num_needed_block_for_prefill;

  CHECK_GT(num_remaining_block_after_prefill, 0);

  std::vector<SequenceStatus> sequence_status = get_all_sequence_status();

  uint32_t max_iter_num = get_max_iter_num(sequence_status);

  for (std::size_t i = 0; i < max_iter_num; ++i) {
    uint32_t num_needed_block_each_step = sequence_status.size();
    if (num_remaining_block_after_prefill < num_needed_block_each_step) {
      return false;
    }
    num_remaining_block_after_prefill -= num_needed_block_each_step;

    CHECK_GE(num_remaining_block_after_prefill, 0);

    update_request_status_each_step(&sequence_status,
                                    &num_remaining_block_after_prefill);

    uint32_t num_reserved_full_blocks =
        get_num_reserved_full_blocks(sequence_status);

    if (num_remaining_block_after_prefill > num_reserved_full_blocks) {
      return true;
    }
  }

  return true;
}

bool BlockCapacityGuard::if_accept_candidate_sequences(
    const std::vector<Sequence*>& candidate_sequences,
    const std::deque<std::shared_ptr<Request>>& running_queue,
    const std::vector<Sequence*>& running_sequences) {
  num_reserved_block_for_prefill_ = 0;

  candidate_sequences_ = candidate_sequences;
  running_queue_ = get_running_sequences(running_queue);
  running_sequences_ = running_sequences;

  compute_reserved_block_num();

  return simulate_is_satisfied_for_candidate_sequences();
}

ZeroEvictionScheduler::ZeroEvictionScheduler(Engine* engine,
                                             const Options& options)
    : ContinuousScheduler(engine, options) {
  block_capacity_guard_ = std::make_unique<BlockCapacityGuard>(block_manager_);
}

ZeroEvictionScheduler::~ZeroEvictionScheduler() {
  // release all requests in the priority queue
  while (!waiting_priority_queue_.empty()) {
    waiting_priority_queue_.pop();
  }

  // release all requests in the running priority queue
  while (!running_queue_.empty()) {
    running_queue_.pop_front();
  }
}

bool ZeroEvictionScheduler::try_allocate_block_for(
    std::shared_ptr<Request> request,
    std::vector<Sequence*>* prefill_sequences,
    std::vector<size_t>* prefill_sequences_budget,
    size_t* allocated_tokens,
    size_t* allocated_seqs,
    size_t remaining_token_budget,
    size_t remaining_seq_budget) {
  size_t ori_allocated_tokens = *allocated_tokens;
  size_t ori_allocated_seqs = *allocated_seqs;
  // for RAII block release guard.
  auto guard = resource_guard([&prefill_sequences,
                               &prefill_sequences_budget,
                               &allocated_tokens,
                               &allocated_seqs,
                               ori_allocated_tokens,
                               ori_allocated_seqs,
                               this]() {
    // deallocation is triggered only when prefix_cache is enabled. This is
    // because if prefix_cache is not enabled, blocks are not actually allocated
    // here.
    if (this->enable_prefix_cache_) {
      for (auto* seq : *prefill_sequences) {
        // release shared blocks
        this->block_manager_->deallocate(seq);
      }
    }

    // reset the var values.
    prefill_sequences->clear();
    prefill_sequences_budget->clear();

    *allocated_tokens = ori_allocated_tokens;
    *allocated_seqs = ori_allocated_seqs;
  });

  for (auto& prefill_sequence : request->sequences()) {
    if (prefill_sequence->finished()) {
      continue;
    }

    size_t num_tokens = prefill_sequence->num_tokens();
    if (remaining_token_budget < *allocated_tokens + num_tokens ||
        remaining_seq_budget < *allocated_seqs + 1) {
      return false;
    }

    prefill_sequences->emplace_back(prefill_sequence.get());
  }

  CHECK(!prefill_sequences->empty());

  if (!block_capacity_guard_->if_accept_candidate_sequences(
          *prefill_sequences, running_queue_, running_sequences_)) {
    return false;
  }

  for (auto* prefill_sequence : *prefill_sequences) {
    // if scheduling is unsuccessful, everything will be deallocated together,
    // so no additional deallocation is needed here.
    if (!block_manager_->allocate(prefill_sequence)) {
      return false;
    }
    size_t num_tokens = prefill_sequence->num_tokens();
    prefill_sequences_budget->emplace_back(num_tokens);

    *allocated_tokens += num_tokens;
    *allocated_seqs += 1;
  }

  guard.success();
  return true;
}

void ZeroEvictionScheduler::handle_prefill_requests(
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  // Handle new request prompt first.
  // Include those requests that are preempted by others.
  //
  // schedule the prefill requests in the waiting priority queue until budgets
  // are exhausted.
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  //
  // NOTE: preempted requests will be pushed in waiting_priority_queue,
  // they may contian many sequences, so we should check here.

  while (!waiting_priority_queue_.empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 &&
         block_manager_->kv_cache_utilization() <
             FLAGS_prefill_scheduling_memory_usage_threshold) {
    std::shared_ptr<Request> request(waiting_priority_queue_.top());
    if (request->finished() || request->cancelled()) {
      block_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the priority queue
      waiting_priority_queue_.pop();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1)
          << "Waiting request should have only one sequence.";
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());

    if (!try_allocate_block_for(request,
                                &prefill_sequences,
                                &prefill_sequences_budget,
                                &allocated_tokens,
                                &allocated_seqs,
                                remaining_token_budget,
                                remaining_seq_budget)) {
      is_satisfied_for_prefill = false;
      break;
    }

    if (prefill_sequences.empty()) {
      continue;
    }

    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    waiting_priority_queue_.pop();
    running_requests_.emplace_back(request);
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
  }

  if (running_sequences_.empty() && !waiting_priority_queue_.empty() &&
      running_queue_.empty() && block_manager_->kv_cache_utilization() == 0) {
    LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                  "a single sequence.";
    // no enough memory to schedule single sequence, just finish the request
    std::shared_ptr<Request> request(waiting_priority_queue_.top());
    waiting_priority_queue_.pop();
    block_manager_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough memory to schedule single sequence"});
  }

  if (!running_sequences_.empty()) {
    last_step_prefill_ = true;
  }
}

std::vector<Batch> ZeroEvictionScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to waiting_priority_queue_
  // Include those requests that are preempted by others.
  std::shared_ptr<Request> request;
  // read from request queue then push to waiting priority queue
  while (request_queue_.read(request)) {
    CHECK(request);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
      // expand sequences to the target number
      request->expand_sequences(false);
    }

    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      waiting_priority_queue_.push(request);
    } else {
      // request from prefill instance in disagge pd mode.
      running_requests_.emplace_back(request);
    }
  }

  // handle finished/cancelled requests
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    std::shared_ptr<Request> request = *it;
    if (request->finished() || request->cancelled()) {
      block_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
      // if a decode request ends, its kv_cache will be released, and prefill
      // can try to reschedule requests.
      is_satisfied_for_prefill = true;
    }
  }

  // insert running requests back to the running queue, iterating from
  // the highest priority to the lowest
  // insert running requests back to the running queue, iterating from
  // the highest priority to the lowest
  if (last_step_prefill_) {
    // insert all requests to the back of running_queue_
    // 1. last step is prefill step:
    // new prefill has high priority, but these requests has lower priority
    // then existed requests in running_queue_ in decoding stage.
    // so we need to push them to the back of running_queue_.
    for (auto it = running_requests_.begin(); it != running_requests_.end();
         ++it) {
      // finished request is set to nullptr
      if (*it == nullptr) {
        continue;
      }
      handle_running_requests(*it);
      running_queue_.push_back(*it);
    }
  } else {
    // insert all requests to the front of running_queue_
    // 2. last step is decode step:
    // We need to traverse running_requests_ array in reverse order.
    // Because there may be some unexecuted requests with
    // lower priorities remaining in the running_queue_.
    // For the requests in running_requests_,
    // their priorities are all higher than those of the
    // remaining requests. Therefore, the `push_front`
    // method needs to be used.
    //
    for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
         ++it) {
      // finished request is set to nullptr
      if (*it == nullptr) {
        continue;
      }
      handle_running_requests(*it);
      running_queue_.push_front(*it);
    }
  }

  // clear previous batch
  last_step_prefill_ = false;
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;

  if (is_satisfied_for_prefill) {
    handle_prefill_requests(
        remaining_token_budget, remaining_seq_budget, finished_requests);
  }

  handle_decode_requests(
      remaining_token_budget, remaining_seq_budget, num_preempted_requests);

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(running_sequences_, running_sequences_budgets_);

  if (!batches[0].empty()) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            waiting_priority_queue_.size() + running_queue_.size());
  GAUGE_SET(num_preempted_requests, num_preempted_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc, block_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(block_manager_->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(block_manager_->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(block_manager_->num_used_blocks()));
  return batches;
}

}  // namespace xllm
