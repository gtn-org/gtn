
#include "catch.hpp"

#include "common.h"
#include "gtn/hd_span.h"
#include "gtn/cuda/cuda.h"

using namespace gtn::detail;

TEST_CASE("test hd_span", "[hd_span]") {
  {
    HDSpan<int> s;
    CHECK(s.size() == 0);
    s.push_back(3);
    s.push_back(4);
    CHECK(s[0] == 3);
    CHECK(s[1] == 4);
    CHECK(s.size() == 2);
    CHECK(*(s.begin()) == 3);

    // Self assignment
    s = s;
    CHECK(s[0] == 3);
    CHECK(s[1] == 4);

    HDSpan<int> sCopy;
    sCopy = s;
    CHECK(sCopy[0] == 3);
    CHECK(sCopy[1] == 4);
    CHECK(sCopy.size() == 2);
    sCopy.clear();

    s.resize(8);
    CHECK(s.size() == 8);
    s.clear();

    s.resize(3);
    s[0] = 1;
    s[1] = 2;
    s[2] = 3;
    sCopy.resize(3);
    sCopy.copy(s.data());
    CHECK(sCopy[0] == 1);
    CHECK(sCopy[1] == 2);
    CHECK(sCopy[2] == 3);

    s.reserve(8);
    CHECK(s[0] == 1);
    CHECK(s[1] == 2);
    CHECK(s[2] == 3);
    CHECK(s.capacity() >= 8);
    s.reserve(2);
    CHECK(s.capacity() >= s.size());
  }

  // Check resizing works
  {
    HDSpan<int> s(2, 1);
    CHECK(s[0] == 1);
    CHECK(s[1] == 1);

    s.resize(4, 2);
    CHECK(s[0] == 1);
    CHECK(s[1] == 1);
    CHECK(s[2] == 2);
    CHECK(s[3] == 2);

    s.resize(3, 3);
    CHECK(s[0] == 1);
    CHECK(s[1] == 1);
    CHECK(s[2] == 2);
  }

  {
    allocations = 0;
    deallocations = 0;
    HDSpan<int> s;
    for (int i = 0; i < 100; i++) {
      s.push_back(i);
    }
    s.clear();
    CHECK(allocations == deallocations);
  }

  // Test constructors
  {
    HDSpan<int> h(0);
    CHECK(h.size() == 0);
  }
  {
    HDSpan<int> h(5);
    CHECK(h.size() == 5);
  }
  {
    HDSpan<int> h(0, 1);
    CHECK(h.size() == 0);
  }
  {
    HDSpan<int> h(5, 1);
    CHECK(h.size() == 5);
    for (int i = 0; i < 5; ++i) {
      CHECK(h[i] == 1);
    }
  }

  {
    HDSpan<float> h(2, 0.5f);
    CHECK(h[0] == 0.5);
    CHECK(h[1] == 0.5);
  }

  {
    HDSpan<bool> h(2, false, Device::CPU);
    CHECK(h[0] == false);
    CHECK(h[1] == false);
  }

  // Test equality
  {
    HDSpan<int> h1(2);
    h1[0] = 1;
    h1[1] = 2;

    HDSpan<int> h2(2);
    h2[0] = 1;
    h2[1] = 2;
    CHECK(h1 == h2);
  }

  {
    HDSpan<int> h1(2);
    h1[0] = 1;
    h1[1] = 2;

    HDSpan<int> h2(1);
    h2[0] = 1;
    CHECK(h1 != h2);
  }

  {
    HDSpan<int> h1(2);
    h1[0] = 1;
    h1[1] = 2;

    HDSpan<int> h2(2);
    h2[0] = 1;
    h2[1] = 3;
    CHECK(h1 != h2);
  }

  // Check copy and move construction
  {
    HDSpan<int> h1(2, 1);
    HDSpan<int> h2(h1);
    CHECK(h1.data() == h2.data());
    CHECK(h1.size() == h2.size());
    CHECK(h1.capacity() == h2.capacity());

    HDSpan<int> h3(std::move(h1));
    CHECK(h3.data() == h2.data());
    // Should be a no-op
    h1.clear();
    CHECK(h3[0] == 1);
  }

  {
    HDSpan<int> h1(2, 1);
    HDSpan<int> h2(h1);

    HDSpan<int> h3;
    h3 = std::move(h1);
    CHECK(h3.data() == h2.data());
    CHECK(h3.size() == h2.size());
    CHECK(h3.capacity() == h2.capacity());

    // Should be a no-op
    h1.clear();
    CHECK(h3[0] == 1);
  }

  // Test swap
  {
    HDSpan<int> h1(2, 1);
    HDSpan<int> h2(3, 2);
    swap(h1, h2);
    CHECK(h1 == HDSpan<int>(3, 2));
    CHECK(h2 == HDSpan<int>(2, 1));
  }
}

TEST_CASE("test hd_span cuda", "[hd_span]") {
  if (!gtn::cuda::isAvailable()) {
    return;
  }
  HDSpan<int> sHost;
  sHost.push_back(1);
  sHost.push_back(2);

  HDSpan<int> sDevice{true, 0};
  sDevice = sHost;
  CHECK(sDevice.size() == 2);

  HDSpan<int> sHost2;
  sHost2 = sDevice;
  CHECK(sHost2[0] == 1);
  CHECK(sHost2[1] == 2);
  CHECK(sHost.size() == 2);

  // Test equality
  {
    HDSpan<int> h1(2, 1, Device::CUDA);

    HDSpan<int> h2(2, 1, Device::CUDA);
    CHECK(h1 == h2);
  }

  {
    HDSpan<int> h1(2, 1, Device::CUDA);

    HDSpan<int> h2(1, 1, Device::CUDA);
    CHECK(h1 != h2);
  }

  {
    HDSpan<int> h1(2, 2, Device::CUDA);

    HDSpan<int> h2(2, 1, Device::CUDA);
    CHECK(h1 != h2);
  }
}

TEST_CASE("test hd_span operations", "[hd_span]") {
  {
    HDSpan<float> a(2, 1.0);
    HDSpan<float> b(2, 2.0);
    HDSpan<float> c(2);
    add(a, b, c);
    HDSpan<float> expected(2, 3.0);
    CHECK(c == expected);
    subtract(a, b, c);
    expected = HDSpan<float>(2, -1.0);
    CHECK(c == expected);
    negate(a, b);
    CHECK(b == expected);
  }

  if (!cuda::isAvailable()) {
    return;
  }

  {
    HDSpan<float> a(2, 1.0, Device::CUDA);
    HDSpan<float> b(2, 2.0, Device::CPU);
    HDSpan<float> c(2, Device::CPU);
    CHECK_THROWS(add(a, b, c));
    CHECK_THROWS(subtract(a, b, c));
    CHECK_THROWS(negate(a, b));
  }

  {
    // *NB* these may segfault if the test fails since catch will try to access
    // device arrays using []
    HDSpan<float> a(2, 1.0, Device::CUDA);
    HDSpan<float> b(2, 2.0, Device::CUDA);
    HDSpan<float> c(2, Device::CUDA);
    add(a, b, c);
    HDSpan<float> expected(2, 3.0, Device::CUDA);
    CHECK(c == expected);
    subtract(a, b, c);
    HDSpan<float> result;
    expected = HDSpan<float>(2, -1.0, Device::CUDA);
    CHECK(c == expected);
    negate(a, b);
    CHECK(b == expected);
  }
}
