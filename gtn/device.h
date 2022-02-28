#pragma once

namespace gtn {

enum class DeviceType {
  CPU, CUDA
};

struct Device {
  static const DeviceType CPU;
  static const DeviceType CUDA;

  Device(DeviceType type, int index);
  Device(DeviceType type);
  DeviceType type;
  int index;
  bool isCuda() const {
    return type == CUDA;
  };
};

bool operator==(const Device& lhs, const Device& rhs);
bool operator!=(const Device& lhs, const Device& rhs);

} // namespace gtn
