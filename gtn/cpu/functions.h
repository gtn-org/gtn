#pragma once

#include "gtn/graph.h"
#include "gtn/functions.h"

namespace gtn {
namespace cpu {

Graph concat(const Graph& g1, const Graph& g2);

Graph concat(const std::vector<Graph>& graphs);

Graph closure(const Graph& g);

Graph union_(const std::vector<Graph>& graphs);

Graph remove(const Graph& g, int label = epsilon);

Graph remove(const Graph& g, int ilabel, int olabel);

Graph compose(const Graph& g1, const Graph& g2);

Graph intersect(const Graph& g1, const Graph& g2);

Graph forwardScore(const Graph& g);

Graph viterbiScore(const Graph& g);

Graph viterbiPath(const Graph& g);

} // namespace cpu
} // namespace gtn
