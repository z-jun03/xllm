#pragma once
#include <absl/container/flat_hash_set.h>

#include <memory>

#include "call.h"
#include "core/runtime/llm_master.h"

namespace xllm {

template <typename T>
class APIServiceImpl {
 public:
  APIServiceImpl(LLMMaster* master, const std::vector<std::string>& models)
      : master_(master), models_(models.begin(), models.end()) {
    CHECK(master != nullptr);
    CHECK(!models_.empty());
  }
  virtual ~APIServiceImpl() = default;

  void process_async(std::shared_ptr<Call> call) {
    std::shared_ptr<T> call_cast = std::dynamic_pointer_cast<T>(call);
    process_async_impl(call_cast);
  }

  virtual void process_async_impl(std::shared_ptr<T> call) = 0;

 protected:
  LLMMaster* master_;
  absl::flat_hash_set<std::string> models_;
};

}  // namespace xllm
