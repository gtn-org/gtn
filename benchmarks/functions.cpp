/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <random>

#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

void timeSimpleOps(Device device = Device::CPU) {
  // time clone
  auto graph = linearGraph(1000, 100, device);
  auto cloneForward = [&graph]() { auto cloned = clone(graph); };
  TIME_DEVICE(cloneForward, device);

  auto cloneBackward = [&graph, out = clone(graph)]() {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME_DEVICE(cloneBackward, device);

  // TODO remove when other functions are implemented in CUDA
  if (device.isCuda()) {
    return;
  }

  // time closure
  auto closureForward = [&graph]() { auto closed = closure(graph); };
  TIME(closureForward);

  auto closureBackward = [&graph, out = closure(graph)]() {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME(closureBackward);

  // time union_
  std::vector<Graph> graphs;
  for (int i = 0; i < 100; i++) {
    graphs.push_back(linearGraph(1000, 1));
  }

  auto unionForward = [&graphs]() { auto out = union_(graphs); };
  TIME(unionForward);

  auto unionBackward = [&graphs, out = union_(graphs)]() {
    for (auto& g : graphs) {
      g.zeroGrad();
    }
    backward(out, true);
  };
  TIME(unionBackward);

  // time concatenate
  auto concatForward = [&graphs]() { auto closed = concat(graphs); };
  TIME(concatForward);

  auto concatBackward = [&graphs, out = concat(graphs)]() {
    for (auto& g : graphs) {
      g.zeroGrad();
    }
    backward(out, true);
  };
  TIME(concatBackward);
}

void timeForward(Device device = Device::CPU) {
  auto graph = linearGraph(200, 1000);
  std::vector<float> weights(graph.numArcs());
  std::generate(weights.begin(), weights.end(), std::rand);
  graph.setWeights(weights.data());
  graph = graph.to(device);

  auto forwardScoreLinearForward = [&graph]() {
    auto out = forwardScore(graph);
  };
  TIME_DEVICE(forwardScoreLinearForward, device);

  auto forwardScoreLinearBackward = [&graph, out = forwardScore(graph)] {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME_DEVICE(forwardScoreLinearBackward, device);

  graph = makeRandomDAG(20000, 200000).to(device);
  auto forwardScoreRandDAGForward = [&graph]() {
    auto out = forwardScore(graph);
  };
  TIME_DEVICE(forwardScoreRandDAGForward, device);

  auto forwardScoreRandDAGBackward = [&graph, out = forwardScore(graph)] {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME_DEVICE(forwardScoreRandDAGBackward, device);
}

void timeCompose(Device device = Device::CPU) {
  const int N1 = 100;
  const int N2 = 50;
  const int A1 = 20;
  const int A2 = 500;
  auto first = linearGraph(N1, A1, device);
  auto second = linearGraph(N2, A2);
  for (int i = 0; i < N2; i++) {
    for (int j = 0; j < A2; j++) {
      // add self loops so composition completes
      second.addArc(i, i, j);
    }
  }
  second = Graph::deepCopy(second, device);

  auto composeForward = [&first, &second]() {
    auto out = compose(first, second);
  };
  TIME_DEVICE(composeForward, device);

  auto composeBackward = [&first, &second, out = compose(first, second)] {
    first.zeroGrad();
    second.zeroGrad();
    backward(out, true);
  };
  TIME_DEVICE(composeBackward, device);

  if (device.isCuda()) {
    return;
  }
  first.arcSort(true);
  second.arcSort();
  auto composeForwardSorted = [&first, &second]() {
    auto out = compose(first, second);
  };
  TIME(composeForwardSorted);
}

int main() {
  /* Various function benchmarks. */
  timeSimpleOps();
  timeForward();
  timeCompose();
  if (cuda::isAvailable()) {
    timeSimpleOps(Device::CUDA);
    timeForward(Device::CUDA);
    timeCompose(Device::CUDA);
  }
}
