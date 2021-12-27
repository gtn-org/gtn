#include "device.h"

#include "cuda/cuda.h"

namespace gtn {

const DeviceType Device::CPU = DeviceType::CPU;
const DeviceType Device::CUDA = DeviceType::CUDA;

Device::Device(DeviceType type, int index) :
    type(type), index(index) { }

Device::Device(DeviceType type) :
    type(type),
    index(type == Device::CUDA ? cuda::getDevice() : 0) { }

bool operator==(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

bool operator!=(const Device& lhs, const Device& rhs) {
  return !(lhs == rhs);
}

} // namespace gtn
