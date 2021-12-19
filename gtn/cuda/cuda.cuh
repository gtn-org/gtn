#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace gtn {
namespace cuda {
namespace detail {

template <typename T>
void fill(T* dst, T val, size_t size) {
  thrust::fill<T*, T>(dst, dst + size, val);
}

} // namespace detail
} // namespace cuda
} // namespace gtn
