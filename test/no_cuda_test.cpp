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
    CHECK_THROWS(cuda::detail::allocate(1, 0));
    CHECK_THROWS(cuda::detail::free((void*)0));
    std::vector<float> a = {1, 0, 1};
    std::vector<float> b = {0, 1, 0};
    CHECK_THROWS(cuda::detail::add(a.data(), b.data(), b.data(), 3, true));
    cuda::detail::add(a.data(), b.data(), b.data(), 3, false);
    CHECK(b[0] == 1);
    CHECK(b[1] == 1);
    CHECK(b[2] == 1);
    CHECK_THROWS(cuda::detail::ones(5, 0));
    cuda::detail::copy(a.data(), b.data(), sizeof(float) * 3);
    CHECK(a[0] == 1);
    CHECK(a[1] == 1);
    CHECK(a[2] == 1);
}

TEST_CASE("Test Graph CUDA", "[Graph.cuda]") {
    Graph g;
    CHECK(!g.isCuda());
    CHECK(equal(g.cpu(), g));
    CHECK_THROWS(g.cuda());
}
