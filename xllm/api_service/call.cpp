#include "call.h"

namespace xllm {

Call::Call(brpc::Controller* controller) : controller_(controller) { init(); }

void Call::init() {
  if (controller_->http_request().GetHeader("x-request-id")) {
    x_request_id_ = *controller_->http_request().GetHeader("x-request-id");
  } else if (controller_->http_request().GetHeader("x-ms-client-request-id")) {
    x_request_id_ =
        *controller_->http_request().GetHeader("x-ms-client-request-id");
  }

  if (controller_->http_request().GetHeader("x-request-time")) {
    x_request_time_ = *controller_->http_request().GetHeader("x-request-time");
  } else if (controller_->http_request().GetHeader("x-request-timems")) {
    x_request_time_ =
        *controller_->http_request().GetHeader("x-request-timems");
  }
}

}  // namespace xllm
