#include "xllm_server_registry.h"

namespace xllm {

XllmServer* ServerRegistry::register_server(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (servers_.find(name) != servers_.end()) {
      LOG(ERROR) << "Register server failed, " << name << " is existed already";
      return servers_[name].get();
    }

    servers_[name] = XllmServerFactory::create_xllm_server();
    return servers_[name].get();
  }
}

XllmServer* ServerRegistry::get_server(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (servers_.find(name) == servers_.end()) {
      LOG(ERROR) << "Server " << name << " not existed.";
      return nullptr;
    }

    return servers_[name].get();
  }
}

}  // namespace xllm
