#pragma once

#include <algorithm>

#include "device.h"
#include "gtn/cuda/cuda.h"

#if defined(_CUDA_)
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
    if (isCuda()) {
      data = static_cast<T*>(
          gtn::cuda::detail::allocate(sizeof(T) * space_, device_.index));
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
    if (isCuda()) {
      gtn::cuda::detail::free(data());
    } else {
      delete[] data();
    }
    data_ = nullptr;
  }

 public:

  explicit HDSpan() {};
  explicit HDSpan(int size, T* data, Device device)
    : size_(size), space_(size), data_(data), device_(device) {
  };
  explicit HDSpan(int size, T val, Device device = Device::CPU)
      : device_(device) {
    resize(size, val);
  };
  explicit HDSpan(int size, Device device = Device::CPU)
    : device_(device) {
      resize(size);
  };
  explicit HDSpan(Device device)
    : device_(device) {};

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

  void resize(size_t size, T val) {
    // Smaller or same size is a no-op with space_ unchanged.
    auto os = size_;
    resize(size);
    if (size > os) {
      if (isCuda()) {
        gtn::cuda::detail::fill(data() + os, val, size - os);
      } else {
        std::fill(data() + os, data() + size, val);
      }
    }
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
    assert(!isCuda());
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

  const T* begin() const {
    return data();
  };

  T* end() {
    return data() + size();
  };

  const T* end() const {
    return data() + size();
  };


  T back() const {
    return data()[size() - 1];
  }
  T& back() {
    return data()[size() - 1];
  }

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
    return device_.isCuda();
  };

  Device device() const {
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
  Device device_{Device::CPU};
};

namespace {

template <typename T>
bool isSameDevice(const HDSpan<T>& a, const HDSpan<T>& b) {
  return a.isCuda() == b.isCuda() && a.device() == b.device();
}

template <typename T>
bool isSameDevice(const HDSpan<T>& a, const HDSpan<T>& b, const HDSpan<T>& c) {
  return a.isCuda() == b.isCuda() && b.isCuda() == c.isCuda() &&
    a.device() == b.device() && b.device() == c.device();
}

} // namespace

template <typename T>
bool operator==(const HDSpan<T>& lhs, const HDSpan<T>& rhs) {
  if (!isSameDevice(lhs, rhs)) {
    throw std::logic_error("Cannot compare two HDSpans on different devices");
  }
  if (lhs.size() != rhs.size()) {
    return false;
  }
  if (lhs.isCuda()) {
    return cuda::detail::equal(lhs.data(), rhs.data(), lhs.size());
  } else {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
  }
}

template <typename T>
bool operator!=(const HDSpan<T>& lhs, const HDSpan<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
void negate(const HDSpan<T>& in, HDSpan<T>& out) {
  if (!isSameDevice(in, out)) {
    throw std::logic_error("Cannot negate HDSpan on different device");
  }
  if (in.size() != out.size()) {
    throw std::logic_error("Cannot negate HDSpans of different sizes");
  }
  if (in.isCuda()) {
    cuda::detail::negate(in.data(), out.data(), in.size());
  } else {
    std::transform(in.begin(), in.end(), out.begin(), std::negate<>());
  }
}

template <typename T>
void add(const HDSpan<T>& lhs, const HDSpan<T>& rhs, HDSpan<T>& out) {
  if (!isSameDevice(lhs, rhs, out)) {
    throw std::logic_error("Cannot add HDSpans on different devices");
  }
  if (lhs.size() != rhs.size()) {
    throw std::logic_error("Cannot add HDSpans of different sizes");
  }
  if (lhs.isCuda()) {
    cuda::detail::add(lhs.data(), rhs.data(), out.data(), lhs.size());
  } else {
    std::transform(
        lhs.begin(), lhs.end(), rhs.begin(), out.begin(), std::plus<>());
  }
}

template <typename T>
void subtract(const HDSpan<T>& lhs, const HDSpan<T>& rhs, HDSpan<T>& out) {
  if (!isSameDevice(lhs, rhs, out)) {
    throw std::logic_error("Cannot subtract HDSpans on different devices");
  }
  if (lhs.size() != rhs.size()) {
    throw std::logic_error("Cannot subtract HDSpans of different sizes");
  }
  if (lhs.isCuda()) {
    cuda::detail::subtract(lhs.data(), rhs.data(), out.data(), lhs.size());
  } else {
    std::transform(
        lhs.begin(), lhs.end(), rhs.begin(), out.begin(), std::minus<>());
  }
}

} // namespace detail
} // namespace gtn
