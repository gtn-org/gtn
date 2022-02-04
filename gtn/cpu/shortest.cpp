/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stack>

#include "gtn/cpu/shortest.h"

namespace gtn {
namespace cpu {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float logadd(float a, float b) {
  if (a == kNegInf) {
    return b;
  }
  if (b == kNegInf) {
    return a;
  }
  return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

void shortestDistanceGrad(
    Graph& g,
    float output,
    const Graph& deltas,
    const std::vector<int>& sortedNodes,
    const std::vector<float>& nodeScores,
    const std::vector<size_t>& maxArcIdxCache,
    bool tropical) {

  std::vector<float> nodeGrads(g.numNodes(), 0.0);
  std::vector<float> arcGrads(g.numArcs(), 0.0);

  for (int i = sortedNodes.size() - 1; i >= 0; --i) {
    auto n = sortedNodes[i];
    for (const auto a : g.out(n)) {
      auto dn = g.dstNode(a);
      float curScore;
      if (tropical) {
        curScore = (a == maxArcIdxCache[dn]) ? nodeGrads[dn] : 0.0f;
      } else if (nodeScores[dn] == kNegInf) {
        curScore = 0.0f;
      } else {
        curScore = nodeGrads[dn] *
            std::exp(nodeScores[n] + g.weight(a) - nodeScores[dn]);
      }
      nodeGrads[n] += curScore;
      arcGrads[a] = curScore;
    }
    if (g.isAccept(n)) {
      if (tropical) {
        nodeGrads[n] += (n == maxArcIdxCache.back()) ? deltas.item() : 0.0f;
      } else if (nodeScores.back() != kNegInf) {
        nodeGrads[n] += std::exp(nodeScores[n] - output) * deltas.item();
      }
    }
  }
  g.addGrad(std::move(arcGrads));
}

std::vector<int> topSort(const Graph& g) {
  std::stack<std::pair<int, int>> nodeStack;
  std::vector<bool> visited(g.numNodes(), false);
  std::vector<int> sortedNodes(g.numNodes());

  int numVisited = 0;
  for (int n = 0; n < g.numNodes(); ++n) {
    if (g.numIn(n) == 0) {
      nodeStack.emplace(n, 0);
      visited[n] = true;
      numVisited++;
    }
  }

  int nodeIdx = g.numNodes() - 1;
  while (!nodeStack.empty()) {
    int n, a;
    std::tie(n, a) = nodeStack.top();
    nodeStack.pop();

    // Stop early if all the nodes have been visited already
    if (numVisited == g.numNodes()) {
      sortedNodes[nodeIdx--] = n;
      continue;
    }

    auto it = g.out(n).begin() + a;
    for (; it < g.out(n).end(); it++) {
      auto dst = g.dstNode(*it);
      if (!visited[dst]) {
        nodeStack.emplace(n, it - g.out(n).begin());
        nodeStack.emplace(dst, 0);
        visited[dst] = true;
        numVisited++;
        break;
      }
    }
    if (it == g.out(n).end()) {
      sortedNodes[nodeIdx--] = n;
    }
  }
  return sortedNodes;
}

} // namespace

Graph shortestDistance(const Graph& g, bool tropical /* = false */) {
  std::vector<float> scores(g.numNodes());
  std::vector<size_t> maxArcIdxCache(g.numNodes() + 1, -1);
  auto sortedNodes = topSort(g);

  auto getScore = [tropical](const std::vector<float>& in, float maxScore) {
    if (in.empty()) {
      return kNegInf;
    }
    if (tropical || maxScore == kInf || maxScore == kNegInf) {
      return maxScore;
    }
    float score = -1.0;
    for (auto s : in) {
      score += std::exp(s - maxScore);
    }
    return maxScore + std::log1p(score);
  };

  for (auto n : sortedNodes) {
    float maxScore = kNegInf;
    std::vector<float> inScores(g.numIn(n) + g.isStart(n));
    int i = 0;
    for (auto a : g.in(n)) {
      auto un = g.srcNode(a);
      inScores[i] = scores[un] + g.weight(a);
      if (inScores[i]  > maxScore) {
        maxScore = inScores[i];
        maxArcIdxCache[n] = a;
      }
      i++;
    }
    if (g.isStart(n)) {
      inScores[i] = 0.0;
      if (inScores[i] > maxScore) {
        maxScore = inScores[i];
        maxArcIdxCache[n] = -1; // an invalid value
      }
    }
    scores[n] = getScore(inScores, maxScore);
  }

  // Accumulate scores at all the accept nodes.
  std::vector<float> inScores;
  float maxScore = kNegInf;
  for (auto n : g.accept()) {
    inScores.push_back(scores[n]);
    if (inScores.back() > maxScore) {
      maxScore = inScores.back();
      maxArcIdxCache.back() = n; // NOTE: Using node idx (instead of arc idx)
    }
  }
  auto score = getScore(inScores, maxScore);

  // clear cache not required for bwd
  if (!tropical) {
    maxArcIdxCache.clear();
  }

  auto gradFunc = [scores = std::move(scores),
                   sortedNodes = std::move(sortedNodes),
                   maxArcIdxCache = std::move(maxArcIdxCache),
                   output = score,
                   tropical](std::vector<Graph>& inputs, Graph deltas) mutable {
    shortestDistanceGrad(
        inputs[0],
        output,
        deltas,
        sortedNodes,
        scores,
        maxArcIdxCache,
        tropical);
  };

  Graph result(gradFunc, {g});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, score);
  return result;
}

Graph shortestPath(const Graph& g) {
  // List of scores and backpointers for each node
  std::vector<int> backPointers(g.numNodes());
  std::vector<float> scores(g.numNodes(), kNegInf);
  auto sortedNodes = topSort(g);

  for (auto n : g.start()) {
    scores[n] = 0.0;
    backPointers[n] = -1;
  }

  for (auto n : sortedNodes) {
    for (auto a : g.in(n)) {
      auto un = g.srcNode(a);
      auto nScore = scores[un] + g.weight(a);
      if (nScore > scores[n]) {
        scores[n] = nScore;
        backPointers[n] = a;
      }
    }
  }

  // Accumulate scores at all the accept nodes.
  float score = kNegInf;
  int best = -1;
  for (auto a : g.accept()) {
    if (scores[a] > score) {
      score = scores[a];
      best = a;
    }
  }

  // Chase the pointers to get the best path (backwards)
  std::vector<int> arcs;
  while (best != -1 && backPointers[best] != -1) {
    auto arc = backPointers[best];
    best = g.srcNode(arc);
    arcs.push_back(arc);
  }

  // Build the best path
  Graph out(nullptr, {g});
  if (best != -1) {
    out.addNode(true, arcs.size() == 0);
  }
  for (auto i = arcs.size(); i > 0; --i) {
    out.addNode(false, i == 1);
    out.addArc(
        arcs.size() - i,
        arcs.size() - i + 1,
        g.ilabel(arcs[i - 1]),
        g.olabel(arcs[i - 1]),
        g.weight(arcs[i - 1]));
  }

  auto gradFunc = [arcs = std::move(arcs)](
                      std::vector<Graph>& inputs, Graph deltas) mutable {
    std::vector<float> grad(inputs[0].numArcs(), 0.0);
    for (auto a = 0; a < deltas.numArcs(); ++a) {
      grad[arcs[a]] += deltas.weight(a);
    }
    inputs[0].addGrad(grad);
  };
  out.setGradFunc(std::move(gradFunc));
  return out;
}

} // namespace cpu
} // namespace gtn
