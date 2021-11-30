#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <gtn/gtn.h>

using namespace gtn;

TEST_CASE("Test Cuda Utils", "[cuda]") {
    CHECK(cuda::isAvailable());
    int num_devices = cuda::deviceCount();
    CHECK(num_devices > 0);

    int device = cuda::getDevice();
    CHECK(device == 0);

    cuda::setDevice(num_devices - 1);
    device = cuda::getDevice();
    CHECK(device == num_devices - 1);
}

TEST_CASE("Test Graph CUDA", "[Graph.cuda]") {
  {
  Graph g;
  CHECK(!g.isCuda());
  g.addNode(true);
  g.addNode(false, true);
  g.addArc(0, 1, 0, 1, 0.5);
  // cpu to cpu is a no-op
  CHECK(g.id() == g.cpu().id());
  auto gdev = g.cuda();
  CHECK(gdev.numNodes() == g.numNodes());
  CHECK(gdev.numArcs() == g.numArcs());
  CHECK(gdev.isCuda());
  CHECK_THROWS(gdev.item());
  CHECK_THROWS(gdev.arcSort());
  // gpu to gpu is a no-op
  CHECK(gdev.id() == gdev.cuda().id());

  auto ghost = gdev.cpu();
  CHECK(!ghost.isCuda());
  std::cout << ghost << std::endl;
  std::cout << g << std::endl;
  CHECK(equal(ghost, g));
  }


/*  {
  Graph g;
  g.addNode(true);
  g.addNode();
  g.addNode(false, true);
  g.addArc(0, 1, 0, 1, 0.5);
  g.addArc(0, 1, 1, 2, 0.3);
  g.addArc(1, 2, 0, 1, 0.7);
  g.addArc(1, 2, 1, 0, 1.7);
  g = g.cuda();
  CHECK(g.isCuda());
  }*/
}
