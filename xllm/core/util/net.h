#pragma once

#include <string>
#include <vector>

namespace xllm {
namespace net {

std::string get_local_ip_addr();
int get_local_free_port();
uint64_t convert_ip_port_to_uint64(const std::string& ip, uint16_t port);

}  // namespace net
}  // namespace xllm
