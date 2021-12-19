#include <stdexcept>

namespace gtn {
namespace cuda {
namespace detail {

template <typename T>
void fill(T* dst, T val, size_t size) {
  throw std::logic_error("[cuda::fill] CUDA not available.");
}

} // namespace detail
} // namespace cuda
} // namespace gtn
