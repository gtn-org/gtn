/*
 * Copyright (c) Meta, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;
using namespace gtn::criterion;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(criterion, m) {
  m.def(
      "ctc_loss",
      [](const Graph& logProbs, const std::vector<int>& target, int blankIdx) {
        py::gil_scoped_release release;
        return ctcLoss(logProbs, target, blankIdx);
      },
      "log_probs"_a,
      "target"_a,
      "blank_idx"_a);
}
