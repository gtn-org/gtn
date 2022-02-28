/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <sstream>

#include "catch.hpp"

#include "common.h"
#include "gtn/graph.h"
#include "gtn/utils.h"

using namespace gtn;

TEST_CASE("test graph", "[graph]") {
  Graph g;
  g.addNode(true);
  g.addNode();
  g.addNode();
  g.addNode();
  g.addNode(false, true);
  CHECK(g.numNodes() == 5);
  CHECK(g.numStart() == 1);
  CHECK(g.numAccept() == 1);

  g.addArc(0, 1, 0);
  g.addArc(0, 2, 1);
  g.addArc(1, 2, 0);
  g.addArc(1, 1, 1, 1, 2.1);
  g.addArc(2, 3, 2);

  CHECK(g.numArcs() == 5);
  CHECK(g.numOut(0) == 2);
  CHECK(g.numIn(1) == 2);
  for (int i = 0; i < g.numIn(1); ++i) {
    CHECK(g.in(1)[i] == g.in(1, i));
  }
  for (int i = 0; i < g.numOut(0); ++i) {
    CHECK(g.out(0)[i] == g.out(0, i));
  }

  // If we (shallow) copy the graph it should have the same structure.
  Graph g_copy = g;
  CHECK(g_copy.numNodes() == 5);
  CHECK(g_copy.numStart() == 1);
  CHECK(g_copy.numAccept() == 1);
  CHECK(g_copy.numArcs() == 5);
  CHECK(g_copy.numOut(0) == 2);
  CHECK(g_copy.numIn(1) == 2);
  CHECK(g_copy.weight(3) == 2.1f);

  // Modifying g should modify g_copy and g_copy2
  g.addNode();
  g.addNode();
  g.addNode();
  g.addNode();
  for (int i = 0; i < 8; i++) {
    g.addArc(i, i, i);
    g.addArc(i, i, i + 1);
    g.addArc(i, i + 1, i);
    g.addArc(i, i + 1, i + 1);
  }
  CHECK(equal(g, g_copy));

  // Check that we can copy a graph and the destination still
  // works when the source graph is out of scope
  Graph g2;
  {
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addArc(0, 1, 0);
    g2 = g1;
  }
  CHECK(g2.numNodes() == 2);
  CHECK(g2.label(0) == 0);
  CHECK(g2.dstNode(0) == 1);
  CHECK(g2.srcNode(0) == 0);

  {
    // We can get a scalar out of a single arc graph.
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1, 1, 3.3);
    CHECK(g.item() == 3.3f);

    // We cannot get a scalar out of a many arc graph.
    g.addArc(0, 1, 2, 2, 3.3);
    CHECK_THROWS(g.item());
  }

  allocations = 0;
  deallocations = 0;
  // We should see flat memory use
  {
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1);
    g.addArc(0, 0, 0);
  }
  CHECK(allocations == deallocations);

  {
    // Check adding transducing arcs
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1, 2);
    CHECK(g.ilabel(0) == 1);
    CHECK(g.olabel(0) == 2);
    g.addArc(0, 1, 1, 0, 2);
    CHECK(g.ilabel(1) == 1);
    CHECK(g.olabel(1) == 0);
  }
}

TEST_CASE("test id", "[graph]") {
  Graph g;
  auto id = g.id();
  g.addNode(true);
  g.addNode(false, true);
  g.addArc(0, 1, 0, 2);
  g.addArc(0, 1, 1, 2);
  g.addArc(0, 1, 2, 2);

  CHECK(g.id() == id);

  auto g1 = g;
  CHECK(g1.id() == g.id());

  std::vector<float> weights = {1, 2, 3};
  g1.setWeights(weights.data());
  CHECK(g1.id() == g.id());

  Graph g2(g);
  CHECK(g2.id() == g.id());

  auto g4 = Graph::deepCopy(g);
  CHECK(g4.id() != g.id());
}

TEST_CASE("test copy", "[graph]") {
  Graph graph =
      loadTxt(std::stringstream("0 1\n"
                                "3 4\n"
                                "0 1 0 2 2\n"
                                "0 2 1 3 1\n"
                                "1 2 0 1 2\n"
                                "2 3 0 0 1\n"
                                "2 3 1 2 1\n"
                                "1 4 0 1 2\n"
                                "2 4 1 1 3\n"
                                "3 4 0 2 2\n"));

  // Test copy
  Graph copied = Graph::deepCopy(graph);
  CHECK(equal(copied, graph));
  CHECK(copied.calcGrad() == graph.calcGrad());
  CHECK(copied.id() != graph.id());
  copied.addArc(0, 3, 0);
  CHECK(!equal(copied, graph));

  graph.arcSort(true);
  copied = Graph::deepCopy(graph);
  CHECK(!copied.ilabelSorted());
  CHECK(copied.olabelSorted());
}

TEST_CASE("test arc weight get and set", "[graph]") {
  std::vector<float> l = {1.1f, 2.2f, 3.3f, 4.4f};

  Graph g;
  g.addNode(true, false);
  g.addNode();
  g.addNode();
  g.addNode();
  g.addNode(false, true);
  g.addArc(0, 1, 0);
  g.addArc(1, 2, 0);
  g.addArc(2, 3, 0);
  g.addArc(3, 4, 0);
  g.setWeights(l.data());

  CHECK(l == std::vector<float>(g.weights(), g.weights() + g.numArcs()));
}

TEST_CASE("test arc label getters", "[graph]") {
  std::vector<int> l = {0, 1, 2, 3};

  Graph g;
  g.addNode(true, false);
  g.addNode();
  g.addNode();
  g.addNode(false, true);
  g.addArc(0, 1, l[0], l[3]);
  g.addArc(0, 2, l[1], l[2]);
  g.addArc(1, 2, l[2], l[1]);
  g.addArc(1, 3, l[3], l[0]);

  CHECK(l == g.labelsToVector(/*ilabel=*/true));
  std::reverse(l.begin(), l.end());
  CHECK(l == g.labelsToVector(/*ilabel=*/false));
}

TEST_CASE("test gradient functionality", "[graph]") {
  {
    // calcGrad is false
    Graph g(false);
    CHECK_THROWS(g.grad());

    g.addGrad(Graph{});
    CHECK_THROWS(g.grad());

    g.addNode();
    g.addArc(0, 0, 0);
    g.addArc(0, 0, 1);

    // this should be a no-op
    g.addGrad({0, 0, 0});
    CHECK_THROWS(g.grad());

    // this should be a no-op
    g.setGradFunc([](std::vector<Graph>&, Graph&) { return Graph{}; });
    CHECK(g.gradFunc() == nullptr);
  }

  {
    // No gradient yet
    Graph g(true);
    CHECK_THROWS(g.grad());

    // Empty gradient
    g.addGrad(Graph{});
    CHECK(equal(g.grad(), Graph{}));

    // No gradient
    g.zeroGrad();
    CHECK_THROWS(g.grad());
    g.addNode();
    g.addNode();
    g.addArc(0, 1, 0);

    Graph grad;
    grad.addNode();
    grad.addNode();
    grad.addArc(0, 1, 0, 0, 1.0);

    g.addGrad(grad);
    CHECK(equal(g.grad(), grad));

    // Grads accumulate properly
    g.addGrad(grad);
    Graph expected;
    expected.addNode();
    expected.addNode();
    expected.addArc(0, 1, 0, 0, 2.0);
    CHECK(equal(g.grad(), expected));

    // Wrong sized grad throws
    grad.addArc(0, 1, 0, 0, 2.0);
    CHECK_THROWS(g.addGrad(grad));
    CHECK_THROWS(g.addGrad({0.0, 1.0}));
  }

  {
    // calcGrad propgates properly
    Graph g1(true);
    Graph g2(false);
    Graph g3(nullptr, {g1, g2});
    CHECK(g3.calcGrad());

    Graph g4(false);
    Graph g5(nullptr, {g2, g4});
    CHECK(!g5.calcGrad());
  }

  // Setting setCalcGrad works
  {
    Graph g1(true);
    g1.addNode();
    g1.addArc(0, 0, 1);

    g1.setCalcGrad(false);
    g1.addGrad(std::vector<float>({1.0}));
    CHECK(!g1.isGradAvailable());

    g1.setCalcGrad(true);
    g1.addGrad(std::vector<float>({1.0}));
    CHECK(g1.isGradAvailable());
  }

  {
    // Check copy vs move
    Graph g;
    g.addNode();
    g.addNode();
    g.addArc(0, 1, 0);

    auto grad = Graph::deepCopy(g);
    grad.setWeight(0, 2.0);

    // this should make a copy of grads weights
    g.addGrad(grad);
    grad.setWeight(0, 4.0);
    CHECK(g.grad().weight(0) == 2.0f);

    // this should make a copy
    std::vector<float> gradsV = {1.0};
    g.addGrad(gradsV);
    g.grad().setWeight(0, 2.0);
    CHECK(gradsV[0] == 1.0);
  }
}

TEST_CASE("test arc sort", "[graph]") {
  // sort on empty graph does nothing
  Graph g;
  g.arcSort();

  g.addNode();
  g.addNode();
  g.addNode();

  g.addArc(0, 1, 1, 3);
  g.addArc(0, 1, 0, 2);
  g.addArc(0, 1, 3, 4);
  g.addArc(1, 1, 3, 0);
  g.addArc(1, 1, 0, 4);
  g.addArc(1, 2, 0, 4);
  g.addArc(1, 2, 1, 1);
  g.addArc(1, 2, 2, 0);

  // sort on ilabel
  g.arcSort();
  auto ilabelCmp = [&g](int a, int b) { return g.ilabel(a) < g.ilabel(b); };
  for (auto n = 0; n < g.numNodes(); ++n) {
    auto arcs = g.in(n);
    CHECK(std::is_sorted(arcs.begin(), arcs.end(), ilabelCmp));
    arcs = g.out(n);
    CHECK(std::is_sorted(arcs.begin(), arcs.end(), ilabelCmp));
  }
  CHECK(g.ilabelSorted());
  CHECK(!g.olabelSorted());

  // sort on olabel
  g.arcSort(true);
  auto olabelCmp = [&g](int a, int b) { return g.olabel(a) < g.olabel(b); };
  for (auto n = 0; n < g.numNodes(); ++n) {
    auto arcs = g.in(n);
    CHECK(std::is_sorted(arcs.begin(), arcs.end(), olabelCmp));
    arcs = g.out(n);
    CHECK(std::is_sorted(arcs.begin(), arcs.end(), olabelCmp));
  }
  CHECK(!g.ilabelSorted());
  CHECK(g.olabelSorted());

  g.markArcSorted();
  CHECK(g.ilabelSorted());

  g.addArc(1, 2, 0, 3);
  CHECK(!g.olabelSorted());

  g.markArcSorted(true);
  CHECK(g.olabelSorted());
}

TEST_CASE("test threaded grad", "[graph]") {
  Graph g;
  g.addNode(true);
  g.addNode(false, true);
  for (int i = 0; i < 1000; i++) {
    g.addArc(0, 1, i);
  }

  auto add_to_grad = [&g]() {
    auto grad = std::vector<float>(g.numArcs(), 1);
    g.addGrad(grad);
  };

  int num_threads = 16;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(add_to_grad));
  }
  for (auto& th : threads) {
    th.join();
  }
  // Check the grad is correct
  CHECK((std::all_of(
      g.grad().weights(),
      g.grad().weights() + g.numArcs(),
      [num_threads](float v) { return v == num_threads; })));
}
