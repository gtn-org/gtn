#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "gtn/graph.h"
#include "gtn/utils.h"
#include "gtn/cuda/cuda.h"

using namespace gtn;

TEST_CASE("Test Cuda Utils", "[cuda]") {
    CHECK(!cuda::isAvailable());
    CHECK_THROWS(cuda::deviceCount());
    CHECK_THROWS(cuda::getDevice());
    CHECK_THROWS(cuda::setDevice(0));
}

TEST_CASE("Test Graph CUDA", "[Graph.cuda]") {
    Graph g;
    CHECK(!g.isCuda());
    CHECK(equal(g.cpu(), g));
    CHECK_THROWS(g.cuda());
}
