#pragma once

#include <acl/acl.h>

#include <atomic>
#include <vector>

namespace xllm {

class NPULayerSynchronizerImpl {
 public:
  NPULayerSynchronizerImpl(const int64_t num_layers);
  virtual ~NPULayerSynchronizerImpl();

  aclrtEvent* get_event(const int64_t layer_index);
  std::atomic<bool>* get_event_flag(const int64_t layer_index);
  bool synchronize_layer(const int64_t layer_index);

 private:
  std::vector<aclrtEvent> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
};

}  // namespace xllm
