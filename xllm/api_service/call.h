#pragma once

#include <brpc/controller.h>

#include <string>

namespace xllm {

class Call {
 public:
  Call(brpc::Controller* controller);
  virtual ~Call() = default;

  std::string get_x_request_id() { return x_request_id_; }
  std::string get_x_request_time() { return x_request_time_; }

 protected:
  void init();

 protected:
  brpc::Controller* controller_;

  std::string x_request_id_;
  std::string x_request_time_;
};

}  // namespace xllm
