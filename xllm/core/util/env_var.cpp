#include "env_var.h"

#include <cstdlib>

namespace xllm {
namespace util {

bool get_bool_env(const std::string& key, bool defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  std::string strVal(val);
  return (strVal == "1" || strVal == "true" || strVal == "TRUE" ||
          strVal == "True");
}

}  // namespace util
}  // namespace xllm
