/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include "catch.hpp"

#include "gtn/autograd.h"
#include "gtn/creations.h"
#include "gtn/functions.h"
#include "gtn/graph.h"
#include "gtn/parallel.h"
#include "gtn/utils.h"

using namespace gtn;

TEST_CASE("test thread pool", "[parallel]") {
  gtn::detail::ThreadPool pool(2);

  {
    std::vector<std::future<int>> results(10);
    for (int i = 0; i < results.size(); ++i) {
      results[i] = pool.enqueue([](int idx){ return idx; }, i);
    }
    for (int i = 0; i < results.size(); ++i) {
      CHECK(results[i].get() == i);
    }
  }

  {
    std::vector<std::future<int>> results(10);
    for (int i = 0; i < results.size(); ++i) {
      results[i] = pool.enqueueIndex(i, [](int idx){ return idx; }, i);
    }
    for (int i = 0; i < results.size(); ++i) {
      CHECK(results[i].get() == i);
    }
  }

  // Check main stream synchronizes with thread pool
  if (cuda::isAvailable()) {
    detail::HDSpan<float> a(1 << 20, 1.0, Device::CUDA);
    detail::HDSpan<float> b(1 << 20, 0.0, Device::CUDA);
    detail::HDSpan<float> c(1 << 20, 1000.0, Device::CUDA);
    for (int i = 0; i < 1000; i++) {
      detail::add(a, b, b);
    }
    std::future<bool> fut = pool.enqueue([&b, &c]() { return b == c; });
    CHECK(fut.get());
  }

  // Check thread pool stream synchronization works
  if (cuda::isAvailable()) {
    detail::HDSpan<float> a(1 << 20, 1.0, Device::CUDA);
    detail::HDSpan<float> b(1 << 20, 0.0, Device::CUDA);
    detail::HDSpan<float> c(1 << 20, 100.0, Device::CUDA);
    for (int i = 0; i < 100; ++i) {
      pool.enqueueIndex(0, [&a, &b]() { detail::add(a, b, b); });
    }
    pool.syncStreams();
    std::future<bool> fut = pool.enqueueIndex(1, [&b, &c]() { return b == c; });
    CHECK(fut.get());
  }

  // Check that the main stream synchronizes with the worker streams
  if (cuda::isAvailable()) {
    detail::HDSpan<float> a(1 << 20, 0.0, Device::CUDA);
    detail::HDSpan<float> b(1 << 20, 100.0, Device::CUDA);
    for (int i = 0; i < 1000; ++i) {
      pool.enqueueIndex(0, [&]() { a = b;});
    }
    pool.syncStreams();
    detail::add(b, b, b);
    detail::HDSpan<float> expected(1 << 20, 100, Device::CUDA);
    CHECK((a == expected));
  }
}

TEST_CASE("test parallel map one arg", "[parallel]") {
  const int B = 4;

  std::vector<Graph> inputs;
  for (size_t i = 0; i < B; ++i) {
    inputs.push_back(scalarGraph(static_cast<float>(i)));
  }
  auto outputs = parallelMap(negate, inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(negate(inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("test parallel map two args", "[parallel]") {
  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2;
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
    inputs2.push_back(scalarGraph(static_cast<float>(2 * i)));
  }

  auto outputs = parallelMap(add, inputs1, inputs2);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(add(inputs1[i], inputs2[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("test parallelmap broadcast", "[parallel]") {
  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2 = {scalarGraph(10.0)};
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
  }

  auto outputs = parallelMap(add, inputs1, inputs2);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    // inputs2[0] should be broadcast
    expectedOutputs.push_back(add(inputs1[i], inputs2[0]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("test parallel map lambda", "[parallel]") {
  auto function = [](const Graph& g1, const Graph& g2, const Graph& g3) {
    return subtract(add(g1, g2), g3);
  };

  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2;
  std::vector<Graph> inputs3;
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
    inputs2.push_back(scalarGraph(static_cast<float>(2 * i)));
    inputs3.push_back(scalarGraph(static_cast<float>(3 * i)));
  }

  auto outputs = parallelMap(function, inputs1, inputs2, inputs3);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    // inputs2[0] should be broadcast
    expectedOutputs.push_back(function(inputs1[i], inputs2[i], inputs3[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("test parallel map vector input", "[parallel]") {
  const int B = 4;

  std::vector<std::vector<Graph>> inputs(B);
  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inputs[i].push_back(scalarGraph(static_cast<float>(i * j)));
    }
  }

  auto outputs = parallelMap(union_, inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(union_(inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("test parallelmap vector input 2", "[parallel]") {
  const int B = 4;

  std::vector<std::vector<Graph>> inputs(B);
  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inputs[i].push_back(scalarGraph(static_cast<float>(i * j)));
    }
  }

  auto outputs = parallelMap(
      static_cast<Graph (*)(const std::vector<Graph>&)>(&concat), inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(concat(inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("test parallel map other typed lambda", "[parallel]") {
  const int B = 4;

  auto function = [](int T, int M, std::vector<float> emissionsScore) -> Graph {
    Graph g = linearGraph(T, M);
    g.setWeights(emissionsScore.data());
    return g;
  };

  int T = 2, M = 4;

  std::vector<std::vector<float>> inputs(B);
  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < T * M; ++j) {
      inputs[i].push_back(static_cast<float>(i * j));
    }
  }

  std::vector<int> t({T});
  std::vector<int> m({M});
  auto outputs = parallelMap(function, t, m, inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(function(T, M, inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("test parallel map backward", "[parallel]") {
  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2;
  std::vector<Graph> inputs1Dup;
  std::vector<Graph> inputs2Dup;
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
    inputs2.push_back(scalarGraph(static_cast<float>(2 * i)));
    inputs1Dup.push_back(scalarGraph(static_cast<float>(i)));
    inputs2Dup.push_back(scalarGraph(static_cast<float>(2 * i)));
  }

  auto outputs = parallelMap(add, inputs1, inputs2);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    auto out = add(inputs1Dup[i], inputs2Dup[i]);
    backward(out);
    expectedOutputs.push_back(out);
  }

  std::vector<bool> retainGraph({false});
  // This cast is needed because backward isn't a complete type before
  // overload resolution
  parallelMap(
      static_cast<void (*)(Graph, bool)>(backward), outputs, retainGraph);

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(inputs1[i].grad(), inputs1Dup[i].grad()));
    CHECK(equal(inputs2[i].grad(), inputs2Dup[i].grad()));
  }
}

TEST_CASE("test parallel map throws", "[parallel]") {
  const int B = 4;

  std::vector<Graph> inputs;
  for (size_t i = 0; i < B; ++i) {
    inputs.push_back(linearGraph(2, 1));
  }

  // Throws - inputs contains graph with more than one arc
  REQUIRE_THROWS_MATCHES(
      parallelMap(negate, inputs),
      std::logic_error,
      Catch::Message("[gtn::negate] input must have only one arc"));
}

TEST_CASE("test parallel map cuda", "[parallel]") {
  if (!cuda::isAvailable()) {
    return;
  }
  {
    const int B = 4;
    std::vector<Graph> inputs;
    std::vector<Graph> expectedOutputs;
    for (size_t i = 0; i < B; ++i) {
      inputs.push_back(scalarGraph(static_cast<float>(i), Device::CUDA));
      expectedOutputs.push_back(negate(inputs[i]));
    }

    auto outputs = parallelMap(negate, inputs);

    for (size_t i = 0; i < B; ++i) {
      CHECK(equal(outputs[i], expectedOutputs[i]));
    }
  }

  {
    std::vector<Graph> inputs;
    inputs.push_back(linearGraph(100, 10000, Device::CUDA));
    inputs.push_back(linearGraph(1, 1, Device::CUDA));
    for (int i = 0; i < 100; ++i) {
      parallelMap(projectInput, inputs);
    }
    std::vector<Graph> inputs1;
    std::vector<Graph> inputs2;
    std::vector<Graph> expectedOutputs;
    inputs1.push_back(scalarGraph(2.0, Device::CUDA));
    inputs1.push_back(scalarGraph(1.0, Device::CUDA));
    inputs2.push_back(scalarGraph(0.0, Device::CUDA));
    inputs2.push_back(scalarGraph(0.0, Device::CUDA));
    expectedOutputs.push_back(scalarGraph(1.0, Device::CUDA));
    expectedOutputs.push_back(scalarGraph(2.0, Device::CUDA));
    auto outputs = parallelMap(add, inputs1, inputs2);
    std::swap(outputs[0], outputs[1]);
    auto results = parallelMap(equal, outputs, expectedOutputs);
    CHECK(results == std::vector<bool>(2, true));
  }

  // Changing the size of the batch shouldn't cause synchronization issues
  {
    std::vector<Graph> inputs1;
    std::vector<Graph> inputs2;
    inputs1.push_back(scalarGraph(1.0, Device::CUDA));
    inputs1.push_back(scalarGraph(1.0, Device::CUDA));
    inputs2.push_back(scalarGraph(0.0, Device::CUDA));
    inputs2.push_back(scalarGraph(0.0, Device::CUDA));
    for (int i = 0; i < 100; i++) {
      inputs2 = parallelMap(add, inputs1, inputs2);
    }
    for (int i = 0; i < 10; i++) {
      inputs1.push_back(scalarGraph(0.0, Device::CUDA));
      inputs2.push_back(scalarGraph(0.0, Device::CUDA));
    }
    auto results = parallelMap(add, inputs1, inputs2);
    CHECK(equal(results[0], scalarGraph(101.0, Device::CUDA)));
    CHECK(equal(results[1], scalarGraph(101.0, Device::CUDA)));
  }
}
