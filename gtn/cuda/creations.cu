
#include "gtn/cuda/creations.h"

namespace gtn {
namespace cuda {
namespace {

typedef Graph::SharedGraph GraphData;

inline int divUp(int x, int y) {
  return (x + y - 1) / y;
}

__global__
void linearArcsKernel(GraphData g, int M, int N) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < M * N) {
    int m = gTid / N;
    int n = gTid % N;
    g.inArcs[gTid] = gTid;
    g.outArcs[gTid] = gTid;
    g.srcNodes[gTid] = m;
    g.dstNodes[gTid] = m + 1;
    g.ilabels[gTid] = n;
    g.olabels[gTid] = n;
  }
}

__global__
void linearNodesKernel(GraphData g, int M, int N) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < M + 1) {
    g.start[gTid] = gTid == 0;
    g.accept[gTid] = gTid == M;
    g.inArcOffset[gTid + 1] = gTid * N;
    g.outArcOffset[gTid] = gTid * N;
  }
  if (gTid == (M + 1)) {
    g.inArcOffset[0] = 0;
    g.outArcOffset[M + 1] = M * N;
  }
}

} // namespace

Graph scalarGraph(float val, bool calcGrad, Device device) {
  auto g = linearGraph(1, 1, calcGrad, device);
  cuda::detail::fill(g.getData().ilabels.data(), epsilon, 1);
  cuda::detail::fill(g.getData().olabels.data(), epsilon, 1);
  cuda::detail::fill(g.getWeights().data(), val, 1);
  return g;
}

Graph linearGraph(int M, int N, bool calcGrad, Device device) {
  auto g = Graph(calcGrad).cuda();
  auto& gData = g.getData();

  // Set start and accept
  gData.numNodes = M + 1;
  gData.start.resize(M + 1, false);
  gData.accept.resize(M + 1, false);
  gData.startIds.resize(1, 0);
  gData.acceptIds.resize(1, M);

  // Set arcs and offsets
  int numArcs = M * N;
  gData.numArcs = numArcs;;
  gData.inArcOffset.resize(M + 2);
  gData.outArcOffset.resize(M + 2);
  gData.inArcs.resize(numArcs);
  gData.outArcs.resize(numArcs);
  gData.ilabels.resize(numArcs);
  gData.olabels.resize(numArcs);
  gData.srcNodes.resize(numArcs);
  gData.dstNodes.resize(numArcs);

  int NT = 128;
  if (numArcs > 0) {
    int blocks = divUp(numArcs, NT);
    linearArcsKernel<<<blocks, NT>>>(gData, M, N);
  }
  {
    int blocks = divUp(M + 2, NT);
    linearNodesKernel<<<blocks, NT>>>(gData, M, N);
  }

  gData.compiled = true;
  g.getWeights().resize(numArcs, 0);

  g.markArcSorted();
  g.markArcSorted(true);
  return g;
}

} // namespace cuda
} // namespace gtn
