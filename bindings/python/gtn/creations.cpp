/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(creations, m) {
  m.def(
      "scalar_graph",
      [](float weight, Device device, bool calcGrad) {
        py::gil_scoped_release release;
        return scalarGraph(weight, device, calcGrad);
      },
      "weight"_a,
      "device"_a = Device(DeviceType::CPU),
      "calc_grad"_a = true);

  m.def(
      "linear_graph",
      [](int M, int N, Device device, bool calcGrad) {
        py::gil_scoped_release release;
        return linearGraph(M, N, device, calcGrad);
      },
      "M"_a, "N"_a,
      "device"_a = Device(DeviceType::CPU),
      "calc_grad"_a = true);
}
