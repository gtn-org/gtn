#include "cuda.h"

namespace gtn {
namespace cuda {

bool isAvailable() {
#if defined(CUDA)
  return deviceCount() > 0;
#else
  return false;
#endif
}

}
}
