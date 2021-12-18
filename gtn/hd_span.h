#pragma once

#include "gtn/cuda/cuda.h"

#if defined(CUDA)
#define HDTAG __host__ __device__
#else
#define HDTAG
#endif

namespace gtn {
namespace detail {

template <class T>
class HDSpan {

 private:

  T* allocAndCopy(const T* other) {
    T* data;
    if (isCuda_) {
      data = static_cast<T*>(
          gtn::cuda::detail::allocate(sizeof(T) * space_, device_));
    } else {
      data = new T[space_];
    }
    gtn::cuda::detail::copy(
      static_cast<void*>(data),
      static_cast<const void*>(other),
      sizeof(T) * size_);
    return data;
  };

  void free() {
    if (data() == nullptr) {
      return;
    }
    if (isCuda_) {
      gtn::cuda::detail::free(data());
    } else {
      delete[] data();
    }
    data_ = nullptr;
  }

 public:

  explicit HDSpan() {};
  explicit HDSpan(int size, T val, bool isCuda = false, int device = 0)
      : isCuda_(isCuda), device_(device) {
    resize(size);
    if (isCuda) {
      gtn::cuda::detail::fill(data(), val, size);
    } else {
      std::fill(data(), data() + size, val);
    }
  };
  explicit HDSpan(int size, bool isCuda = false, int device = 0)
    : isCuda_(isCuda), device_(device) {
      resize(size);
  };
  explicit HDSpan(bool isCuda, int device = 0)
    : isCuda_(isCuda), device_(device) {};

  HDTAG
  const T operator[](size_t idx) const {
    return data_[idx];
  };

  HDTAG
  T& operator[](size_t idx) {
    return data_[idx];
  };

  void copy(const T* other) {
    gtn::cuda::detail::copy(
      static_cast<void*>(data_),
      static_cast<const void*>(other),
      sizeof(T) * size_);
  };

  HDSpan& operator=(const HDSpan& other) {
    if (this->data_ == other.data_) {
      return *this;
    }
    size_ = other.size();
    space_ = size_;
    auto newData = allocAndCopy(other.data());
    free();
    data_ = newData;
  };

  void resize(size_t size) {
    // Smaller or same size is a no-op with space_ unchanged.
    if (size <= size_) {
      size_ = size;
      return;
    }
    space_ = size;
    auto newData = allocAndCopy(data_);
    size_ = size;
    free();
    data_ = newData;
  };

  void push_back(T val) {
    assert(!isCuda_);
    if (size_ == space_) {
      auto oldSize_  = size_;
      resize(space_ ? space_ << 1 : 1);
      size_ = oldSize_;
    }
    data_[size_++] = val;
  };

  T* begin() {
    return data();
  };

  T* end() {
    return data() + size();
  };

  HDTAG
  const T* data() const {
    return data_;
  };

  HDTAG
  T* data() {
    return data_;
  };

  HDTAG
  size_t size() const {
    return size_;
  };

  bool isCuda() const {
    return isCuda_;
  };

  int device() const {
    return device_;
  };

  void clear() {
    free();
    size_ = 0;
    space_ = 0;
  }

 private:
  T* data_{nullptr};
  size_t size_{0};
  size_t space_{0};
  bool isCuda_{false};
  int device_{0};
};

} // namespace detail
} // namespace gtn
