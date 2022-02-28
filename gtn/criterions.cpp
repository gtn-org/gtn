/*
 * Copyright (c) Meta, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtn/criterions.h"

#include <algorithm>
#include <queue>
#include <set>

#include "gtn/functions.h"

namespace gtn {
namespace criterion {

Graph ctcLoss(
    const Graph& logProbs,
    const std::vector<int>& target,
    const int blankIdx) {
  Graph gLabel{false};
  int L = target.size();
  int S = 2 * L + 1;
  for (int l = 0; l < S; ++l) {
    int idx = (l - 1) / 2;
    gLabel.addNode(l == 0, l == S - 1 or l == S - 2);
    int label = l % 2 ? target[idx] : blankIdx;
    gLabel.addArc(l, l, label);
    if (l > 0) {
      gLabel.addArc(l - 1, l, label);
    }
    if (l % 2 and l > 1 and label != target[idx - 1]) {
      gLabel.addArc(l - 2, l, label);
    }
  }
  gLabel = gLabel.to(logProbs.device());
  return negate(forwardScore(intersect(gLabel, logProbs)));
}
} // namespace criterion
} // namespace gtn
