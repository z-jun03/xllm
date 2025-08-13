#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "core/framework/sampling/sampling_params.h"
#include "mm_data.h"
#include "sequence.h"
#include "stopping_checker.h"

namespace xllm {

class SequencesGroup {
 public:
  SequencesGroup(const std::string& prompt,
                 const std::vector<int32_t>& prompt_tokens,
                 const torch::Tensor& input_embedding,
                 const MMData& mm_data,
                 const SequenceParams& sequence_params);

  bool finished() const;

  bool expand_sequences(bool share_prefix);

  void generate_outputs(std::vector<SequenceOutput>& outputs,
                        const Tokenizer& tokenizer);

  std::vector<std::unique_ptr<Sequence>>& sequences() { return sequences_; }

 private:
  void add();

 private:
  const std::string& prompt_;                  // ref from request
  const std::vector<int32_t>& prompt_tokens_;  // ref from request
  const torch::Tensor& input_embedding_;       // ref from request
  const MMData& mm_data_;                      // ref from request
  SequenceParams sequence_params_;

 private:
  std::vector<std::unique_ptr<Sequence>> sequences_;
};

}  // namespace xllm
