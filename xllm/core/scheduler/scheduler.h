#pragma once

#include <absl/time/time.h>

#include <memory>
#include <string>

#include "framework/request/request.h"

namespace xllm {

class Scheduler {
 public:
  virtual ~Scheduler() = default;

  // add a new request to scheduler.
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;

  // scheduler forward execute
  virtual void step(const absl::Duration& timeout) = 0;

  // offline running
  virtual void generate() = 0;

  // incr/decr pending requests
  virtual void incr_pending_requests(size_t count) {}
  virtual void decr_pending_requests() {}
  virtual size_t num_pending_requests() { return 0; }

  virtual uint32_t get_waiting_requests_num() const = 0;
};

}  // namespace xllm
