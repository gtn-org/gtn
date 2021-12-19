#pragma once

#include "gtn/graph.h"

extern size_t allocations;
extern size_t deallocations;

using namespace gtn;

bool checkCuda(
    const Graph& g1,
    const Graph& g2,
    std::function<Graph(const Graph&, const Graph&)> func);
