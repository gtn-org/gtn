/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <numeric>

#include "gtn/creations.h"

#include "gtn/cpu/creations.h"
#include "gtn/cuda/creations.h"

namespace gtn {

Graph scalarGraph(
    float val, bool calcGrad /* = true */, Device device /* = Device::CPU */) {
  if (device.isCuda()) {
    return cuda::scalarGraph(val, calcGrad, device);
  } else {
    return cpu::scalarGraph(val, calcGrad);
  }
}

Graph scalarGraph(float val, Device device) {
  return scalarGraph(val, true, device);
}

Graph linearGraph(
    int M, int N,
    bool calcGrad /* = true */,
    Device device /* = Device::CPU */) {
  if (device.isCuda()) {
    return cuda::linearGraph(M, N, calcGrad, device);
  } else {
    return cpu::linearGraph(M, N, calcGrad);
  }
}

Graph linearGraph(
    int M, int N,
    Device device) {
  return linearGraph(M, N, true, device);
}

} // namespace gtn
