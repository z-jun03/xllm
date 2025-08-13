#pragma once

#include <mutex>
#include <unordered_map>

#include "xllm_server.h"

namespace xllm {

class ServerRegistry {
 public:
  static ServerRegistry& get_instance() {
    static ServerRegistry instance;
    return instance;
  }

  XllmServer* register_server(const std::string& name);
  XllmServer* get_server(const std::string& name);

 private:
  ServerRegistry() = default;
  ~ServerRegistry() = default;
  DISALLOW_COPY_AND_ASSIGN(ServerRegistry);

  std::unordered_map<std::string, std::unique_ptr<XllmServer>> servers_;
  std::mutex mutex_;
};

}  // namespace xllm
