#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "common.h"

size_t allocations;
size_t deallocations;

// Override globals just for testing
void* operator new(std::size_t size) {
  allocations++;
  return std::malloc(size);
}

void operator delete(void* p) throw() {
  deallocations++;
  free(p);
}

bool checkCuda(
    const Graph& g1,
    const Graph& g2,
    std::function<Graph(const Graph&, const Graph&)> func) {
  if (!gtn::cuda::isAvailable()) {
    return true;
  }
//  auto gOutH = func(g1, g2);
//  auto gOutD = func(g1.cuda(), g2.cuda());
  return false;
}
