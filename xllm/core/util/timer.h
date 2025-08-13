#pragma once

#include <absl/time/time.h>

namespace xllm {

class Timer final {
 public:
  Timer();

  // reset the timer
  void reset();

  // get the elapsed time in seconds
  double elapsed_seconds() const;

 private:
  // the start time of the timer
  absl::Time start_;
};

}  // namespace xllm