#pragma once

#include <string>

namespace xllm {
namespace util {

bool get_bool_env(const std::string& key, bool defaultValue);

}  // namespace util
}  // namespace xllm
