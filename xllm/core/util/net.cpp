#include "net.h"

#include <arpa/inet.h>
#include <glog/logging.h>
#include <netdb.h>
#include <sys/socket.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_set>

namespace xllm {
namespace net {

static std::mutex g_port_mutex;
static std::unordered_set<int> g_allocated_port_map;

// TODO: return private ip
std::string get_local_ip_addr() {
  char ip[INET_ADDRSTRLEN]{'\0'};
  char hostname[256];
  int ret = gethostname(hostname, sizeof(hostname));
  if (ret != 0) {
    LOG(ERROR) << "gethostname failed";
    return "";
  }
  struct addrinfo* info = nullptr;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  ret = getaddrinfo(hostname, NULL, &hints, &info);
  if (ret != 0) {
    LOG(ERROR) << "getaddrinfo failed";
    return "";
  }
  auto guard = std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)>(
      info, freeaddrinfo);
  auto* addr = (struct sockaddr_in*)info->ai_addr;
  auto* result = inet_ntop(addr->sin_family, &addr->sin_addr, ip, sizeof(ip));

  return std::string(ip);
}

int get_local_free_port() {
  std::lock_guard<std::mutex> lock(g_port_mutex);
  int port;
  do {
    port = 0;
    struct sockaddr_in addr;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return -1;
    }
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr*)&addr, &len) == -1) {
      return -1;
    }
    port = ntohs(addr.sin_port);

    ::close(fd);
  } while (g_allocated_port_map.find(port) != g_allocated_port_map.end());

  g_allocated_port_map.insert(port);

  return port;
}

uint64_t convert_ip_port_to_uint64(const std::string& ip, uint16_t port) {
  in_addr ip_addr;
  CHECK(inet_pton(AF_INET, ip.c_str(), &ip_addr) == 1)
      << "Invalid IPv4 address format : " << ip;

  uint32_t ip_network = ip_addr.s_addr;
  return (static_cast<uint64_t>(ip_network) << 32) | port;
}

}  // namespace net
}  // namespace xllm
