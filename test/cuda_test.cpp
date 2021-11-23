#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "gtn/cuda.h"
#include "gtn/graph.h"

using namespace gtn;

TEST_CASE("Test Cuda Utils", "[cuda]") {
#if defined(CUDA)
    CHECK(cuda::isAvailable());
    int num_devices = cuda::deviceCount();
    CHECK(num_devices > 0);

    int device = cuda::getDevice();
    CHECK(device == 0);

    cuda::setDevice(num_devices - 1);
    device = cuda::getDevice();
    CHECK(device == num_devices - 1);

#else
    CHECK(!cuda::isAvailable());
#endif
}

TEST_CASE("Test Graph CUDA", "[Graph.cuda]") {
    Graph g;
    CHECK(!g.isCuda());

#if defined(CUDA)
    g.cuda();
    CHECK(g.isCuda());
    g.cpu();
    CHECK(!g.isCuda());
#else
    CHECK(!g.isCuda());
    CHECK_THROWS(g.cuda());
#endif
}
