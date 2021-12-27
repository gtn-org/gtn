/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <numeric>
#include "gtn/cpu/creations.h"

namespace gtn {
namespace cpu {

Graph scalarGraph(float val, bool calcGrad) {
  Graph g1(calcGrad);
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 1, epsilon, epsilon, val);
  return g1;
}

Graph linearGraph(int M, int N, bool calcGrad) {
  Graph g(calcGrad);
  auto& gData = g.getData();

  // Set start and accept
  gData.numNodes = M + 1;
  gData.start.resize(M + 1, false);
  gData.start[0] = true;
  gData.accept.resize(M + 1, false);
  gData.accept.back() = true;
  gData.startIds.resize(1, 0);
  gData.acceptIds.resize(1, M);

  // Set arcs and offsets
  int numArcs = M * N;
  gData.numArcs = numArcs;;
  gData.inArcOffset.resize(M + 2);
  gData.outArcOffset.resize(M + 2);
  gData.inArcs.resize(numArcs);
  gData.outArcs.resize(numArcs);
  gData.ilabels.resize(numArcs);
  gData.olabels.resize(numArcs);
  gData.srcNodes.resize(numArcs);
  gData.dstNodes.resize(numArcs);
  std::iota(gData.inArcs.begin(), gData.inArcs.end(), 0);
  std::iota(gData.outArcs.begin(), gData.outArcs.end(), 0);
  for (int m = 0; m < M; ++m) {
    int s = m * N;
    int e = s + N;
    std::fill(gData.srcNodes.begin() + s, gData.srcNodes.begin() + e, m);
    std::fill(gData.dstNodes.begin() + s, gData.dstNodes.begin() + e, m + 1);
    std::iota(gData.ilabels.begin() + s, gData.ilabels.begin() + e, 0);
    std::iota(gData.olabels.begin() + s, gData.olabels.begin() + e, 0);
    gData.inArcOffset[m + 1] = s;
    gData.outArcOffset[m] = s;
  }
  gData.inArcOffset[0] = 0;
  gData.outArcOffset[M] = numArcs;
  gData.inArcOffset.back() = numArcs;
  gData.outArcOffset.back() = numArcs;
  gData.compiled = true;
  g.getWeights().resize(numArcs, 0);

  g.markArcSorted();
  g.markArcSorted(true);
  return g;
}

} // namespace cpu
} // namespace gtn
