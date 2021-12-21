#pragma once

#include "gtn/graph.h"

namespace gtn {
namespace cpu {

Graph scalarGraph(float val, bool calcGrad);

Graph linearGraph(int M, int N, bool calcGrad);

} // namespace cpu
} // namespace gtn
