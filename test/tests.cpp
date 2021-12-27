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
