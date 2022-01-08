#include "catch.hpp"

#include "gtn/graph.h"
#include "gtn/utils.h"
#include "gtn/cuda/cuda.h"

using namespace gtn;

TEST_CASE("test cuda utils", "[cuda]") {
    CHECK(!cuda::isAvailable());
    CHECK_THROWS(cuda::deviceCount());
    CHECK_THROWS(cuda::getDevice());
    CHECK_THROWS(cuda::setDevice(0));
    CHECK_THROWS(cuda::synchronize());
    CHECK_THROWS(cuda::synchronizeStream());
    CHECK_THROWS(cuda::Event());

    CHECK_THROWS(cuda::detail::allocate(1, 0));
    CHECK_THROWS(cuda::detail::free((void*)0));
    std::vector<float> a = {1, 0, 1};
    std::vector<float> b = {0, 1, 0};
    CHECK_THROWS(cuda::detail::add(a.data(), b.data(), b.data(), 3));
    CHECK_THROWS(cuda::detail::subtract(a.data(), b.data(), b.data(), 3));
    CHECK_THROWS(cuda::detail::negate(a.data(), b.data(), 3));
    CHECK_THROWS(cuda::detail::fill(nullptr, 0, 0));
    cuda::detail::copy(a.data(), b.data(), sizeof(float) * 3);
    CHECK(a == b);
}

TEST_CASE("test graph cuda", "[cuda]") {
    Graph g;
    CHECK(!g.isCuda());
    CHECK(equal(g.cpu(), g));
    CHECK_THROWS(g.cuda());
}
