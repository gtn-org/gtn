#include "catch.hpp"

#include "gtn/cuda/cuda.h"
#include "gtn/device.h"

using namespace gtn;

TEST_CASE("test device", "[device]") {
  Device cpu0 = Device{Device::CPU, 0};
  Device cpu1 = Device{Device::CPU, 1};
  Device cuda0 = Device{Device::CUDA, 0};
  Device cuda1 = Device{Device::CUDA, 1};
  CHECK(cpu0 == Device::CPU);
  CHECK_FALSE(cpu0 == cpu1);
  CHECK_FALSE(cuda0 == cuda1);
  CHECK_FALSE(cpu0.isCuda());
  CHECK(cuda0.isCuda());
  CHECK(cpu0 != cpu1);
  CHECK(cpu0 != cuda0);
  if (cuda::isAvailable()) {
    // Using default device index only works when cuda is available
    CHECK(cuda0 == Device::CUDA);
    CHECK_FALSE(Device::CUDA == Device::CPU);
    CHECK(Device::CPU != Device::CUDA);
  }
}
