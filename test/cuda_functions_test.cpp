#include <cmath>
#include <sstream>

#include "catch.hpp"
#include "gtn/gtn.h"

using namespace gtn;

namespace {
  constexpr float inf = std::numeric_limits<float>::infinity();
}

// *NB* num_arcs is assumed to be greater than num_nodes.
Graph makeRandomDAG(int num_nodes, int num_arcs) {
  Graph graph;
  graph.addNode(true);
  for (int n = 1; n < num_nodes; n++) {
    graph.addNode(false, n == num_nodes - 1);
    graph.addArc(n - 1, n, 0); // assure graph is connected
  }
  for (int i = 0; i < num_arcs - num_nodes + 1; i++) {
    // To preserve DAG property, select src then select dst to be
    // greater than source.
    // select from [0, num_nodes-2]:
    auto src = rand() % (num_nodes - 1);
    // then select from  [src + 1, num_nodes - 1]:
    auto dst = src + 1 + rand() % (num_nodes - src - 1);
    graph.addArc(src, dst, 0);
  }
  return graph;
}

TEST_CASE("test cuda scalar ops", "[cuda functions]") {
  auto g1 = scalarGraph(3.0).cuda();

  auto result = negate(g1);
  CHECK(result.item() == -3.0);

  auto g2 = scalarGraph(4.0).cuda();
  result = add(g1, g2);
  CHECK(result.item() == 7.0);

  result = subtract(g2, g1);
  CHECK(result.item() == 1.0);
}

TEST_CASE("test device matching", "[cuda functions]") {
  auto g1 = Graph();
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 1, 0);

  auto g2 = Graph();
  g2.addNode(true);
  g2.addNode(false, true);
  g2.addArc(0, 1, 0);

  CHECK_THROWS(compose(g1, g2.cuda()));
  CHECK_THROWS(compose(g1.cuda(), g2));
  if (cuda::deviceCount() > 1) {
    CHECK_THROWS(compose(g1.cuda(Device{Device::CUDA, 0}), g2.cuda(Device{Device::CUDA, 1})));
  }
}

TEST_CASE("test cuda compose", "[cuda functions]") {
  auto check = [](const Graph& g1, const Graph& g2) {
    auto gOut = compose(g1, g2);
    auto gOutP = compose(g1.cuda(), g2.cuda()).cpu();
    return isomorphic(gOut, gOutP);
  };

  // Empty result
  CHECK(check(linearGraph(1, 1), linearGraph(2, 1)));

  // Accepts empty string
  {
    auto g1 = Graph();
    g1.addNode(true, true);
    auto g2 = Graph();
    g2.addNode(true, true);
    CHECK(check(g1, g2));
  }

  // Check some simple chain graphs
  CHECK(check(linearGraph(1, 1), linearGraph(1, 1)));
  CHECK(check(linearGraph(5, 1), linearGraph(5, 1)));
  CHECK(check(linearGraph(5, 2), linearGraph(5, 1)));
  CHECK(check(linearGraph(5, 10), linearGraph(5, 1)));
  CHECK(check(linearGraph(1, 2), linearGraph(1, 2)));
  CHECK(check(linearGraph(5, 2), linearGraph(5, 2)));
  CHECK(check(linearGraph(5, 5), linearGraph(5, 3)));
  CHECK(check(linearGraph(5, 3), linearGraph(5, 5)));

  // Check some graphs with self-loops!
  {
    auto g1 = linearGraph(1, 1);
    auto g2 = linearGraph(1, 1);
    g1.addArc(0, 0, 0, 0);
    g1.addArc(1, 1, 0, 0);
    CHECK(check(g1, g2));

    g2.addArc(0, 0, 0, 0);
    g2.addArc(1, 1, 0, 0);
    CHECK(check(g1, g2));
  }

  // Weights combine properly
  {
    auto g1 = linearGraph(2, 3);
    auto g2 = linearGraph(2, 3);
    std::vector<float> w1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<float> w2 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    g1.setWeights(w1.data());
    g2.setWeights(w2.data());
    CHECK(check(g1, g2));
  }

  // More complex test cases
  {
    // Self-loop in the composed graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 0);
    g1.addArc(0, 1, 1);
    g1.addArc(1, 1, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0);
    g2.addArc(1, 1, 0);
    g2.addArc(1, 2, 1);

    std::stringstream in(
        "0\n"
        "2\n"
        "0 1 0\n"
        "1 1 0\n"
        "1 2 1\n");
    Graph expected = loadTxt(in);
    CHECK(isomorphic(compose(g1.cuda(), g2.cuda()).cpu(), expected));
  }

  {
    // Loop in the composed graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0);
    g1.addArc(1, 1, 1);
    g1.addArc(1, 0, 0);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 0, 0);
    g2.addArc(0, 1, 1);
    g2.addArc(1, 0, 1);

    std::stringstream in(
        "0\n"
        "2\n"
        "0 1 0\n"
        "1 0 0\n"
        "1 2 1\n"
        "2 1 1\n");
    Graph expected = loadTxt(in);
    CHECK(isomorphic(compose(g1.cuda(), g2.cuda()).cpu(), expected));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode();
    g1.addNode();
    g1.addNode(false, true);
    for (int i = 0; i < g1.numNodes() - 1; i++) {
      for (int j = 0; j < 3; j++) {
        g1.addArc(i, i + 1, j, j, static_cast<float>(j));
      }
    }

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 3.5);
    g2.addArc(1, 1, 0, 0, 2.5);
    g2.addArc(1, 2, 1, 1, 1.5);
    g2.addArc(2, 2, 1, 1, 4.5);
    std::stringstream in(
        "0\n"
        "6\n"
        "0 1 0 0 3.5\n"
        "1 2 0 0 2.5\n"
        "1 4 1 1 2.5\n"
        "2 3 0 0 2.5\n"
        "2 5 1 1 2.5\n"
        "4 5 1 1 5.5\n"
        "3 6 1 1 2.5\n"
        "5 6 1 1 5.5\n");
    Graph expected = loadTxt(in);
    auto gOutP = compose(g1.cuda(), g2.cuda()).cpu();
    CHECK(isomorphic(gOutP, expected));
  }
}

TEST_CASE("test cuda compose epsilon", "[cuda functions]") {

  auto check = [](const Graph& g1, const Graph& g2) {
    auto gOut = compose(g1, g2);
    auto gOutP = compose(g1.cuda(), g2.cuda()).cpu();
    return isomorphic(gOut, gOutP);
  };

  {
    // Simple test case for output epsilon on first graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 0, epsilon, 1.0);
    g1.addArc(0, 1, 1, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 2, 3);

    CHECK(check(g1, g2));
  }

  {
    // Simple test case for input epsilon on second graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 2, 3);
    g2.addArc(1, 1, epsilon, 0, 2.0);

    CHECK(check(g1, g2));
  }

  // A series of tests making sure we handle redundant epsilon paths correctly
  {
    Graph g1;
    g1.addNode(true, true);
    g1.addArc(0, 0, 0, epsilon);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0, 1.0);

    CHECK(check(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true, true);
    g1.addArc(0, 0, 0, epsilon);

    Graph g2;
    g2.addNode(true, true);
    g2.addArc(0, 0, epsilon, 0);

    CHECK(check(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true, true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(1, 0, 0, epsilon);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0, 1.0);

    CHECK(check(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(0, 0, 0, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0);
    g2.addArc(0, 1, 1, 1);

    CHECK(check(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(0, 1, 0, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0);
    g2.addArc(0, 0, 1, 1);

    CHECK(check(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(0, 1, 0, 1);
    g1.addArc(0, 0, 1, 0);


    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0);
    g2.addArc(0, 0, 1, 0);
    g2.addArc(0, 1, 0, 1);

    CHECK(check(g1, g2));
  }

  {
    // This test case is taken from "Weighted Automata Algorithms", Mehryar
    // Mohri, https://cs.nyu.edu/~mohri/pub/hwa.pdf Section 5.1, Figure 7
    std::unordered_map<std::string, int> symbols = {
        {"a", 0}, {"b", 1}, {"c", 2}, {"d", 3}, {"e", 4}};
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode();
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, symbols["a"], symbols["a"]);
    g1.addArc(1, 2, symbols["b"], epsilon);
    g1.addArc(2, 3, symbols["c"], epsilon);
    g1.addArc(3, 4, symbols["d"], symbols["d"]);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, symbols["a"], symbols["d"]);
    g2.addArc(1, 2, epsilon, symbols["e"]);
    g2.addArc(2, 3, symbols["d"], symbols["a"]);

    CHECK(check(g1, g2));
  }

  {
    // Test multiple input/output epsilon transitions per node
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 1, epsilon, 1.1);
    g1.addArc(0, 1, 2, epsilon, 2.1);
    g1.addArc(0, 1, 3, epsilon, 3.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 3, 2.1);
    g2.addArc(0, 1, 1, 2);

    CHECK(check(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addNode();
    g1.addArc(0, 1, 0);
    g1.addArc(0, 2, epsilon);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon);
    g2.addArc(1, 2, 0);

    CHECK(check(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, epsilon, epsilon, 1);
    g1.addArc(0, 2, 0, 0, 3);
    g1.addArc(1, 2, 0, 0, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, epsilon, 2);
    g2.addArc(1, 2, 0, 0, 2);

    CHECK(check(g1, g2));
  }
}

TEST_CASE("test cuda compose grad", "[cuda functions]") {
  Graph first;
  first.addNode(true);
  first.addNode();
  first.addNode();
  first.addNode();
  first.addNode(false, true);
  first.addArc(0, 1, 0, 0, 0);
  first.addArc(0, 1, 1, 1, 1);
  first.addArc(0, 1, 2, 2, 2);
  first.addArc(1, 2, 0, 0, 0);
  first.addArc(1, 2, 1, 1, 1);
  first.addArc(1, 2, 2, 2, 2);
  first.addArc(2, 3, 0, 0, 0);
  first.addArc(2, 3, 1, 1, 1);
  first.addArc(2, 3, 2, 2, 2);
  first.addArc(3, 4, 0, 0, 0);
  first.addArc(3, 4, 1, 1, 1);
  first.addArc(3, 4, 2, 2, 2);

  Graph second;
  second.addNode(true);
  second.addNode();
  second.addNode(false, true);
  second.addArc(0, 1, 0, 0, 3.5);
  second.addArc(1, 1, 0, 0, 2.5);
  second.addArc(1, 2, 1, 1, 1.5);
  second.addArc(2, 2, 1, 1, 4.5);

  first = first.cuda();
  second = second.cuda();
  auto composed = compose(first, second);
  backward(composed);

  auto firstGrad = first.grad().cpu();
  auto secondGrad = second.grad().cpu();
  std::vector<float> expectedFirst = {1, 0, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0};
  std::vector<float> expectedSecond = {1, 2, 3, 2};
  for (int i = 0; i < firstGrad.numArcs(); i++) {
    CHECK(expectedFirst[i] == firstGrad.weight(i));
  }
  for (int i = 0; i < secondGrad.numArcs(); i++) {
    CHECK(expectedSecond[i] == secondGrad.weight(i));
  }
}

TEST_CASE("test cuda compose epsilon grad", "[cuda functions]") {
  Graph first;
  first.addNode(true);
  first.addNode();
  first.addNode();
  first.addNode();
  first.addNode(false, true);
  first.addArc(0, 1, 0, 0);
  first.addArc(1, 2, 1, epsilon);
  first.addArc(2, 3, 2, epsilon);
  first.addArc(3, 4, 3, 3);

  Graph second;
  second.addNode(true);
  second.addNode();
  second.addNode();
  second.addNode(false, true);
  second.addArc(0, 1, 0, 3);
  second.addArc(1, 2, epsilon, 4);
  second.addArc(2, 3, 3, 0);

  first = first.cuda();
  second = second.cuda();
  auto composed = compose(first, second);
  backward(composed);

  auto firstGrad = first.grad().cpu();
  auto secondGrad = second.grad().cpu();

  std::vector<float> expectedFirst= {1, 1, 1, 1};
  std::vector<float> expectedSecond = {1, 1, 1};
  for (int i = 0; i < firstGrad.numArcs(); i++) {
    CHECK(expectedFirst[i] == firstGrad.weight(i));
  }
  for (int i = 0; i < secondGrad.numArcs(); i++) {
    CHECK(expectedSecond[i] == secondGrad.weight(i));
  }
}

Graph makeChainGraph(const std::vector<int>& input) {
  Graph chain(false);
  chain.addNode(true);
  for (auto i : input) {
    auto n = chain.addNode(false, chain.numNodes() == input.size());
    chain.addArc(n - 1, n, i);
  }
  return chain;
}

TEST_CASE("test edit distance", "[cuda functions]") {
  auto computeEditDistance = [](const int numTokens,
                                const std::vector<int>& x,
                                const std::vector<int>& y,
                                bool useCuda) {
    // Make edits graph
    Graph edits(false);
    edits.addNode(true, true);

    for (int i = 0; i < numTokens; ++i) {
      // Add substitutions
      for (int j = 0; j < numTokens; ++j) {
        edits.addArc(0, 0, i, j, -(i != j));
      }
      // Add insertions and deletions
      edits.addArc(0, 0, i, gtn::epsilon, -1);
      edits.addArc(0, 0, gtn::epsilon, i, -1);
    }

    // Make inputs
    auto xG = makeChainGraph(x);
    auto yG = makeChainGraph(y);

    // Compose and viterbi to get distance
    if (useCuda) {
      xG = xG.cuda();
      yG = yG.cuda();
      edits = edits.cuda();
    }
    auto outG = compose(xG, compose(edits, yG));
    auto score = viterbiScore(outG);
    return -score.item();
  };

  // Small test case
  auto dist = computeEditDistance(5, {0, 1, 0, 1}, {0, 0, 0, 1, 1}, true);
  CHECK(dist == 2);

  // Larger random test cases
  const int minLength = 10;
  const int maxLength = 100;
  for (int numToks = 50; numToks < 70; numToks++) {
    // Random lengths in [minLength, maxLength)
    auto xLen = minLength + rand() % (maxLength - minLength);
    auto yLen = minLength + rand() % (maxLength - minLength);

    // Random vectors x, y with tokens in [0, numToks)
    std::vector<int> x;
    for (int i = 0; i < xLen; i++) {
      x.push_back(rand() % numToks);
    }
    std::vector<int> y;
    for (int i = 0; i < yLen; i++) {
      y.push_back(rand() % numToks);
    }

    auto dist = computeEditDistance(numToks, x, y, true);
    auto expected = computeEditDistance(numToks, x, y, false);
    CHECK(dist == expected);
  }
}


TEST_CASE("test ngrams", "[cuda functions]") {
  auto countNgrams = [](const int numTokens,
                        const std::vector<int>& input,
                        const std::vector<int>& ngram,
                        bool useCuda) {
    // Make n-gram counting graph
    const int n = ngram.size();
    Graph ngramCounter = linearGraph(n, numTokens);
    for (int i = 0; i < numTokens; ++i) {
      ngramCounter.addArc(0, 0, i, gtn::epsilon);
      ngramCounter.addArc(n, n, i, gtn::epsilon);
    }

    // Make inputs
    auto inputG = makeChainGraph(input);
    auto ngramG = makeChainGraph(ngram);

    if (useCuda) {
      ngramCounter = ngramCounter.cuda();
      inputG = inputG.cuda();
      ngramG = ngramG.cuda();
    }
    auto outG = compose(inputG, compose(ngramCounter, ngramG));
    auto score = forwardScore(outG);
    return round(std::exp(score.item()));
  };

  // Small test
  auto counts = countNgrams(2, {0, 1, 0, 1}, {0, 1}, true);
  CHECK(counts == 2);

  // Larger random test cases
  const int minLength = 300;
  const int maxLength = 500;
  const int n = 3;
  const int numToks = 5;
  for (int t = 0; t < 10; t++) {
    // Random length in [minLength, maxLength)
    auto inputLen = minLength + rand() % (maxLength - minLength);

    // Random vectors input, ngram with tokens in [0, numToks)
    std::vector<int> input;
    for (int i = 0; i < inputLen; i++) {
      input.push_back(rand() % numToks);
    }
    std::vector<int> ngram;
    for (int i = 0; i < n; i++) {
      ngram.push_back(rand() % numToks);
    }

    auto count = countNgrams(numToks, input, ngram, true);
    auto expected = countNgrams(numToks, input, ngram, false);
    CHECK(count == expected);
  }
}

TEST_CASE("test cuda project and clone", "[cuda functions]") {
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

  // Test clone
  graph = graph.cuda();
  Graph cloned = clone(graph);
  CHECK(equal(graph.cpu(), cloned.cpu()));

  // Test projecting input
  Graph inputExpected =
      loadTxt(std::stringstream("0 1\n"
                                "3 4\n"
                                "0 1 0 0 2\n"
                                "0 2 1 1 1\n"
                                "1 2 0 0 2\n"
                                "2 3 0 0 1\n"
                                "2 3 1 1 1\n"
                                "1 4 0 0 2\n"
                                "2 4 1 1 3\n"
                                "3 4 0 0 2\n"));
  CHECK(equal(projectInput(graph).cpu(), inputExpected));

  // Test projecting output
  Graph outputExpected =
      loadTxt(std::stringstream("0 1\n"
                                "3 4\n"
                                "0 1 2 2 2\n"
                                "0 2 3 3 1\n"
                                "1 2 1 1 2\n"
                                "2 3 0 0 1\n"
                                "2 3 2 2 1\n"
                                "1 4 1 1 2\n"
                                "2 4 1 1 3\n"
                                "3 4 2 2 2\n"));
  CHECK(equal(projectOutput(graph).cpu(), outputExpected));
}

TEST_CASE("test cuda forward score", "[cuda functions]") {
  {
    // Check score of empty graph
    Graph g;
    CHECK(forwardScore(g.cuda()).item() == -inf);
  }

  {
    // Handles negative infinity
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -inf);
    g.addArc(0, 1, 1, 1, -inf);
    CHECK(forwardScore(g.cuda()).item() == -inf);
  }

  {
    // Handles infinity
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, inf);
    g.addArc(0, 1, 1, 1, 0);
    CHECK(forwardScore(g.cuda()).item() == inf);
  }

  {
    // Handles positive and negative infinity
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, inf);
    g.addArc(0, 1, 1, 1, -inf);
    CHECK(forwardScore(g.cuda()).item() == inf);
  }


  {
    // Single Node
    Graph g;
    g.addNode(true, true);
    CHECK(forwardScore(g.cuda()).item() == 0.0);
  }

  {
    // A simple test case
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 1);
    g.addArc(0, 1, 1, 1, 2);
    g.addArc(0, 1, 2, 2, 3);
    g.addArc(1, 2, 0, 0, 1);
    g.addArc(1, 2, 1, 1, 2);
    g.addArc(1, 2, 2, 2, 3);
    CHECK(forwardScore(g.cuda()).item() == Approx(6.8152));
  }

  {
    // Handle two start nodes
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -5);
    g.addArc(0, 2, 0, 0, 1);
    g.addArc(1, 2, 0, 0, 2);
    float expected = std::log(std::exp(1) + std::exp(-5 + 2) + std::exp(2));
    CHECK(forwardScore(g.cuda()).item() == Approx(expected));
  }

  {
    // Handle two accept nodes
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    g.addArc(1, 2, 0, 0, 2);
    float expected = std::log(2 * std::exp(2) + std::exp(4));
    CHECK(forwardScore(g.cuda()).item() == Approx(expected));
  }

  {
    // Handle case where some arcs don't lead to accepting states
    Graph g;
    g.addNode(true);
    g.addNode(false, false);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    CHECK(forwardScore(g.cuda()).item() == 2.0);
  }

  {
    // A more complex test case
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 1\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = loadTxt(in);
    CHECK(forwardScore(g.cuda()).item() == Approx(8.36931));
  }
}

TEST_CASE("test cuda forward score grad", "[cuda functions]") {
  auto checkGrad = [](const Graph& g) {
    auto gDev = g.cuda();
    backward(forwardScore(g));
    backward(forwardScore(gDev));
    auto gradW = g.grad().weights();
    auto gradDev = gDev.grad().cpu();
    auto gradDevW = gradDev.weights();
    float diff = 0, tot = 0;
    for (int i = 0; i < g.numArcs(); ++i) {
      diff += std::pow(gradW[i] - gradDevW[i], 2);
      tot += std::pow(gradW[i], 2);
    }
    return (diff / tot) < 1e-3;
  };

  {
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 1);
    g.addArc(0, 1, 1, 1, 2);
    g.addArc(0, 1, 2, 2, 3);
    g.addArc(1, 2, 0, 0, 1);
    g.addArc(1, 2, 1, 1, 2);
    g.addArc(1, 2, 2, 2, 3);
    CHECK(checkGrad(g));
  }

  {
    // Handle two start nodes
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -5);
    g.addArc(0, 2, 0, 0, 1);
    g.addArc(1, 2, 0, 0, 2);
    g = g.cuda();
    backward(forwardScore(g));
    auto grad = g.grad().cpu();
    double denom = 1 / (std::exp(-3) + std::exp(1) + std::exp(2));
    CHECK(grad.weight(0) == Approx(denom * std::exp(-3)));
    CHECK(grad.weight(1) == Approx(denom * std::exp(1)));
    CHECK(grad.weight(2) == Approx(denom * (std::exp(-3) + std::exp(2))));
  }

  {
    // Handle two accept nodes
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    g.addArc(1, 2, 0, 0, 2);
    g = g.cuda();
    backward(forwardScore(g));
    auto grad = g.grad().cpu();
    double denom = 1 / (2 * std::exp(2) + std::exp(4));
    CHECK(grad.weight(0) == Approx(denom * (std::exp(2) + std::exp(4))));
    CHECK(grad.weight(1) == Approx(denom * std::exp(2)));
    CHECK(grad.weight(2) == Approx(denom * std::exp(4)));
  }

  {
    // Handle case where some arcs don't lead to accepting states
    Graph g;
    g.addNode(true);
    g.addNode(false, false);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    g = g.cuda();
    backward(forwardScore(g));
    auto grad = g.grad().cpu();
    CHECK(grad.weight(0) == Approx(0.0));
    CHECK(grad.weight(1) == Approx(1.0));
  }

  {
    // Handles negative infinity
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -inf);
    g.addArc(0, 1, 1, 1, -inf);
    g = g.cuda();
    backward(forwardScore(g));
    auto grad = g.grad().cpu();
    CHECK(std::isnan(grad.weight(0)));
    CHECK(std::isnan(grad.weight(1)));

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, -inf);
    g2.addArc(0, 1, 1, 1, 1.0);
    g2 = g2.cuda();
    backward(forwardScore(g2));
    auto grad2 = g2.grad().cpu();
    CHECK(grad2.weight(0) == Approx(0.0));
    CHECK(grad2.weight(1) == Approx(1.0));
  }

  {
    // Handles infinity
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, inf);
    g.addArc(0, 1, 1, 1, inf);
    g = g.cuda();
    backward(forwardScore(g));
    auto grad = g.grad().cpu();
    CHECK(std::isnan(grad.weight(0)));
    CHECK(std::isnan(grad.weight(1)));

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, inf);
    g2.addArc(0, 1, 1, 1, 1.0);
    g2 = g2.cuda();
    backward(forwardScore(g2));
    auto grad2 = g2.grad().cpu();
    CHECK(std::isnan(grad2.weight(0)));
    CHECK(std::isnan(grad2.weight(1)));
  }

  {
    // A more complex test case
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 1\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = loadTxt(in);
    CHECK(checkGrad(g));
  }
}

TEST_CASE("test cuda viterbi score", "[cuda functions]") {
  {
    // Check score of empty graph
    Graph g;
    CHECK(viterbiScore(g.cuda()).item() == -inf);
  }

  {
    // A simple test case
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 1);
    g.addArc(0, 1, 1, 1, 2);
    g.addArc(0, 1, 2, 2, 3);
    g.addArc(1, 2, 0, 0, 1);
    g.addArc(1, 2, 1, 1, 2);
    g.addArc(1, 2, 2, 2, 3);
    CHECK(viterbiScore(g.cuda()).item() == 6.0f);
  }

  {
    // Handle two start nodes
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -5);
    g.addArc(0, 2, 0, 0, 1);
    g.addArc(1, 2, 0, 0, 2);
    CHECK(viterbiScore(g.cuda()).item() == 2.0f);
  }

  {
    // Handle two accept nodes
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    g.addArc(1, 2, 0, 0, 2);
    CHECK(viterbiScore(g.cuda()).item() == 4.0f);
  }

  {
    // A more complex test case
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 1\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = loadTxt(in);
    CHECK(viterbiScore(g.cuda()).item() == 7.0f);
  }
}

TEST_CASE("test cuda viterbi score grad", "[cuda functions]") {
  auto gradsToVec = [](Graph g) {
    g = g.cuda();
    backward(viterbiScore(g));
    auto grad = g.grad().cpu();
    std::vector<float> v(grad.numArcs());
    std::copy(grad.weights(), grad.weights() + grad.numArcs(), v.begin());
    return v;
  };

  {
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 1);
    g.addArc(0, 1, 1, 1, 2);
    g.addArc(0, 1, 2, 2, 3);
    g.addArc(1, 2, 0, 0, 1);
    g.addArc(1, 2, 1, 1, 2);
    g.addArc(1, 2, 2, 2, 3);
    std::vector<float> expected = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    CHECK(gradsToVec(g) == expected);
  }

  {
    // Handle two start nodes
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -5);
    g.addArc(0, 2, 0, 0, 1);
    g.addArc(1, 2, 0, 0, 2);
    std::vector<float> expected = {0.0, 0.0, 1.0};
    CHECK(gradsToVec(g) == expected);
  }

  {
    // Handle two accept nodes
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    g.addArc(1, 2, 0, 0, 2);
    std::vector<float> expected = {1.0, 0.0, 1.0};
    CHECK(gradsToVec(g) == expected);
  }

  {
    // A more complex test case
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 1\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = loadTxt(in);
    // two possible paths with same viterbi score
    std::vector<float> expected1 = {1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    std::vector<float> expected2 = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    CHECK(((gradsToVec(g) == expected1) || (gradsToVec(g) == expected2)));
  }
}
