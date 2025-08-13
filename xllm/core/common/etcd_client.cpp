#include "etcd_client.h"

#include <glog/logging.h>

#include <etcd/Client.hpp>
#include <etcd/Response.hpp>
#include <nlohmann/json.hpp>

namespace xllm {

EtcdClient::EtcdClient(const std::string& etcd_addr)
    : client_(etcd_addr), etcd_addr_(etcd_addr) {
  auto response = client_.put("XLLM_PING", "PING");
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

EtcdClient::~EtcdClient() { stop_watch(); }

void EtcdClient::add_watch(const std::string& key_prefix,
                           Callback callback,
                           bool recursive) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  if (watchers_.find(key_prefix) != watchers_.end()) {
    watchers_[key_prefix].watcher->Cancel();
  }
  auto watcher = std::make_unique<etcd::Watcher>(
      client_,
      key_prefix,
      [callback](etcd::Response response) { callback(response); },
      recursive);

  watchers_[key_prefix] = {std::move(watcher), callback};
}

void EtcdClient::remove_watch(const std::string& key_prefix) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  auto it = watchers_.find(key_prefix);
  if (it != watchers_.end()) {
    it->second.watcher->Cancel();
    watchers_.erase(it);
  }
}

void EtcdClient::stop_watch() {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  for (auto& pair : watchers_) {
    pair.second.watcher->Cancel();
  }

  watchers_.clear();
}

bool EtcdClient::get_master_service(const std::string& key,
                                    std::string* values) {
  auto response = client_.get(key);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }
  *values = response.value().as_string();
  return true;
}

bool EtcdClient::register_instance(const std::string& key,
                                   const std::string& value,
                                   const int ttl) {
  auto keep_alive = std::make_shared<etcd::KeepAlive>(&client_, ttl);
  auto response = client_.put(key, value, keep_alive->Lease());
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd set " << key << " failed: " << response.error_message();
    keep_alive->Cancel();
    return false;
  }
  keep_alives_.emplace_back(std::move(keep_alive));

  return true;
}

}  // namespace xllm
