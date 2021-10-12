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

PYBIND11_MODULE(_experimental, m) {
  m.def(
      "pcompose",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return compose(g1, g2);
      },
      "g1"_a,
      "g2"_a);
}
