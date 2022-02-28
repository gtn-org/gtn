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

/**
 * A generic host-device span object which is capable of storing and managing
 * host or device arrays.
 *
 * This object is used to store most underlying `Graph` data for either host or
 * device graphs. Two reasons to use the HDSpan class over more standard
 * alternatives (`std::vector` and `thrust::device_vector`) are:
 * - HDSpan is easy to compile wihtout CUDA
 * - In order to pass HDSpan by value to device kernels they do not do resource
 *   management. The caller is responsible for clearing the memory.
 * Host array can be appended to using `push_back`. Both host and device array
 * support setting and getting of values at a given index using `mySpan[idx] =
 * myVal`. (*NB* accessing a host/device array in device/host code does not throw,
 * and will likely segfault.)
 *
 *
 * Example usage:
 * ```
 * // host array of size 10 initialized with 0s
 * HDSpan<int> hostSpan(10, 0, Device::CPU);
 *
 * // uninitialized device array
 * HDSpan<int> deviceSpan(Device::CUDA);
 *
 * // Copies between host and device are managed automatically
 * deviceSpan = hostSpan;
 *
 * // Cleanup
 * hostSpan.clear();
 * deviceSpan.clear();
 * ```
 */
template <class T>
class HDSpan {

 private:

  T* allocAndCopy(const T* other) {
    T* data;
    if (isCuda()) {
      data = static_cast<T*>(
          gtn::cuda::detail::allocate(sizeof(T) * capacity_, device_.index));
    } else {
      data = new T[capacity_];
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
    : size_(size), capacity_(size), data_(data), device_(device) {
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
  HDSpan(const HDSpan<T>& other) :
    capacity_(other.capacity()),
    size_(other.size()),
    data_(other.data_),
    device_(other.device()) { };

  HDTAG
  HDSpan(HDSpan<T>&& other) :
    capacity_(other.capacity()),
    size_(other.size()),
    data_(other.data_),
    device_(other.device()) {
      other.size_ = 0;
      other.capacity_ = 0;
      other.data_ = nullptr;
  };

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
    capacity_ = size_;
    auto newData = allocAndCopy(other.data());
    free();
    data_ = newData;
    return *this;
  };

  HDSpan& operator=(HDSpan&& other) {
    if (this->data_ == other.data_) {
      return *this;
    }
    size_ = other.size();
    capacity_ = other.capacity();
    free();
    data_ = other.data();
    device_ = other.device();
    other.size_ = 0;
    other.capacity_ = 0;
    other.data_ = nullptr;
    return *this;
  };

  void resize(size_t size, T val) {
    // Smaller or same size is a no-op with capacity_ unchanged.
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
    // If size is smaller than capacity just update the size.
    if (size <= capacity_) {
      size_ = size;
      return;
    }
    capacity_ = size;
    auto newData = allocAndCopy(data_);
    size_ = size;
    free();
    data_ = newData;
  };

  void reserve(size_t size) {
    if (size <= capacity_) {
      return;
    }
    capacity_ = size;
    auto newData = allocAndCopy(data_);
    free();
    data_ = newData;
  };

  void push_back(T val) {
    assert(!isCuda());
    if (size_ == capacity_) {
      auto oldSize_  = size_;
      resize(capacity_ ? capacity_ << 1 : 1);
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
  size_t capacity() const {
    return capacity_;
  };

  HDTAG
  size_t size() const {
    return size_;
  };

  bool isCuda() const {
    return device_.isCuda();
  };

  HDTAG
  Device device() const {
    return device_;
  };

  void clear() {
    free();
    size_ = 0;
    capacity_ = 0;
  }

  template <typename U>
  friend void swap(HDSpan<U>& lhs, HDSpan<U>& rhs);

 private:
  T* data_{nullptr};
  size_t size_{0};
  size_t capacity_{0};
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
void swap(HDSpan<T>& lhs, HDSpan<T>& rhs) {
  if (!isSameDevice(lhs, rhs)) {
    throw std::invalid_argument(
        "[swap] Objects must be on the same device");
  }
  std::swap(lhs.data_, rhs.data_);
  std::swap(lhs.size_, rhs.size_);
  std::swap(lhs.capacity_, rhs.capacity_);
}


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

/** Negate an HDSpan. Only float arrays are supported. */
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

/** Add one HDSpan to another.
 *
 * Only float arrays are supported. */
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

/** Subtract one HDSpan from another.
 *
 * Only float arrays are supported. */
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
