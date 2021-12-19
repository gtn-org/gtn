
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
    HDSpan<bool> h(2, false, false);
    CHECK(h[0] == false);
    CHECK(h[1] == false);
  }
}

TEST_CASE("test hd_span cuda", "[hdspan]") {
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
}
