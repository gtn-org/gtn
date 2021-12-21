#include "catch.hpp"

#include "gtn/graph.h"
#include "gtn/creations.h"
#include "gtn/utils.h"

using namespace gtn;

TEST_CASE("test cuda scalar creation", "[cuda creations]") {
  float weight = static_cast<float>(rand());
  auto g = scalarGraph(weight, Device::CUDA).cpu();
  CHECK(g.numArcs() == 1);
  CHECK(g.label(0) == epsilon);
  CHECK(g.numNodes() == 2);
  CHECK(g.item() == weight);
  CHECK(g.calcGrad() == true);
  CHECK(equal(g, scalarGraph(weight)));
}

TEST_CASE("test cuda linear creation", "[cuda creations]") {
  std::vector<int> nodeSizes = {0, 1, 5};
  std::vector<int> arcSizes = {0, 1, 5, 10};
  for (auto N : nodeSizes) {
    for (auto M : arcSizes) {
      auto gDev = linearGraph(M, N, Device::CUDA).cpu();
      auto gHost = linearGraph(M, N);
      CHECK(equal(gDev, gHost));
    }
  }
}
