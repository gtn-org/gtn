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

PYBIND11_MODULE(cuda, m) {
  m.def("is_available", &gtn::cuda::isAvailable);
  m.def("device_count", &gtn::cuda::deviceCount);
  m.def("get_device", &gtn::cuda::getDevice);
  m.def("set_device", &gtn::cuda::setDevice, "device"_a);
}
