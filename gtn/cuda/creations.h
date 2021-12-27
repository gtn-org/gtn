#pragma once

#include "gtn/graph.h"

namespace gtn {
namespace cuda {

Graph scalarGraph(float val, bool calcGrad, Device device);

Graph linearGraph(int M, int N, bool calcGrad, Device device);

} // namespace cuda
} // namespace gtn
