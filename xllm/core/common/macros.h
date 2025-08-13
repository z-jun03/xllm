#pragma once

namespace xllm {

#define PROPERTY(T, property)                                                 \
 public:                                                                      \
  [[nodiscard]] const T& property() const& noexcept { return property##_; }   \
  [[nodiscard]] T& property() & noexcept { return property##_; }              \
  [[nodiscard]] T&& property() && noexcept { return std::move(property##_); } \
                                                                              \
  auto property(const T& value) & -> decltype(*this) {                        \
    property##_ = value;                                                      \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  auto property(T&& value) & -> decltype(*this) {                             \
    property##_ = std::move(value);                                           \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  void property(const T& value) && = delete;                                  \
  void property(T&& value) && = delete;                                       \
                                                                              \
  T property##_

#define SAFE_CONCAT(a, b) (a##b)

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) ((void)(x))
#endif

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete
#endif

// Define a macro to simplify adding elements from a vector to a repeated field
#define ADD_VECTOR_TO_PROTO(proto_field, vec) \
  do {                                        \
    proto_field->Reserve(vec.size());         \
    for (const auto& value : vec) {           \
      *proto_field->Add() = value;            \
    }                                         \
  } while (0)

#define CALLBACK_WITH_ERROR(CODE, MSG) callback(Status{CODE, MSG});

}  // namespace xllm
