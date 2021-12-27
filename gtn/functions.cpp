/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <queue>
#include <set>

#include "gtn/cpu/functions.h"
#include "gtn/cuda/functions.h"

namespace gtn {

#define DISPATCH1(fn) \
Graph fn(const Graph& g) { \
  if (g.isCuda()) { \
    return cuda::fn(g); \
  } else { \
    return cpu::fn(g); \
  } \
}

#define DISPATCH2(fn) \
Graph fn(const Graph& g1, const Graph& g2) { \
  deviceCheck(g1, g2, #fn); \
  if (g1.isCuda()) { \
    return cuda::fn(g1, g2); \
  } else { \
    return cpu::fn(g1, g2); \
  } \
}

#define DISPATCHV(fn) \
Graph fn(const std::vector<Graph>& graphs) { \
  if (graphs.empty()) { \
    return cpu::fn(graphs); \
  } \
  deviceCheck(graphs, #fn); \
  if (graphs[0].isCuda()) { \
    return cuda::fn(graphs); \
  } else { \
    return cpu::fn(graphs); \
  } \
}

void deviceCheck(const std::vector<Graph>& graphs, const std::string& name) {
  auto device = graphs[0].device();
  for (auto& g : graphs) {
    if (device != g.device()) {
      throw std::invalid_argument(
        "[gtn::" + name + "] Graphs must be on the same device");
    }
  }
}

void deviceCheck(const Graph& g1, const Graph& g2, const std::string& name) {
  deviceCheck({g1, g2}, name);
}

Graph negate(const Graph& g) {
  if (g.numArcs() != 1) {
    throw std::logic_error("[gtn::negate] input must have only one arc");
  }
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(negate(deltas));
  };
  auto result = Graph::deepCopy(g);
  result.setInputs({g});
  result.setGradFunc(gradFunc);
  detail::negate(g.getWeights(), result.getWeights());
  return result;
}

Graph add(const Graph& g1, const Graph& g2) {
  deviceCheck(g1, g2, "add");
  if (g1.numArcs() != 1 || g2.numArcs() != 1) {
    throw std::logic_error("[gtn::add] inputs must have only one arc");
  }
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
    inputs[1].addGrad(deltas);
  };
  auto result = Graph::deepCopy(g1);
  result.setInputs({g1, g2});
  result.setGradFunc(gradFunc);
  detail::add(g1.getWeights(), g2.getWeights(), result.getWeights());
  return result;
}

Graph subtract(const Graph& g1, const Graph& g2) {
  deviceCheck(g1, g2, "subtract");
  if (g1.numArcs() != 1 || g2.numArcs() != 1) {
    throw std::logic_error("[gtn::subtract] inputs must have only one arc");
  }
  float weight = g1.item() - g2.item();
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
    if (inputs[1].calcGrad()) {
      inputs[1].addGrad(negate(deltas));
    }
  };
  auto result = Graph::deepCopy(g1);
  result.setInputs({g1, g2});
  result.setGradFunc(gradFunc);
  detail::subtract(g1.getWeights(), g2.getWeights(), result.getWeights());
  return result;
}

Graph clone(const Graph& g, Projection projection /* = Projection::NONE */) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas.weights());
  };
  Graph out = Graph::deepCopy(g);
  out.setInputs({g.withoutWeights()});
  out.setGradFunc(gradFunc);
  auto& gData = out.getData();
  if (projection == Projection::OUTPUT) {
    gData.ilabels.copy(gData.olabels.data());
  } else if (projection == Projection::INPUT) {
    gData.olabels.copy(gData.ilabels.data());
  }
  return out;
}

Graph projectInput(const Graph& g) {
  return clone(g, Projection::INPUT);
}

Graph projectOutput(const Graph& g) {
  return clone(g, Projection::OUTPUT);
}

Graph concat(const Graph& g1, const Graph& g2) {
  return concat({g1, g2});
}

DISPATCHV(concat)
DISPATCH1(closure)
DISPATCHV(union_)
DISPATCH2(intersect)
DISPATCH2(compose)

Graph remove(const Graph& g, int label /* = epsilon */) {
  return remove(g, label, label);
}

Graph remove(const Graph& g, int ilabel, int olabel) {
  if (g.isCuda()) {
    return cuda::remove(g, ilabel, olabel);
  } else {
    return cpu::remove(g, ilabel, olabel);
  }
}

DISPATCH1(forwardScore)
DISPATCH1(viterbiScore)
DISPATCH1(viterbiPath)

} // namespace gtn
