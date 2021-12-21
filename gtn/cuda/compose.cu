/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <tuple>
#include <sstream>

#include <thrust/device_ptr.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "gtn/cuda/cuda.h"
#include "gtn/hd_span.h"
#include "gtn/cuda/functions.h"


using namespace gtn::detail;

namespace gtn {
namespace cuda {
namespace detail {

namespace {

typedef Graph::SharedGraph GraphData;

// A resource manager struct for gradient data.
struct GradInfo {
  int* first;
  int* second;
  ~GradInfo() {
    CUDA_CHECK(cudaFree(first));
    CUDA_CHECK(cudaFree(second));
  };
};

struct ExploreNodeAndArcs {
  int2 arcPair;
  int2 nodePair;
  int nodeIdx;
  bool exploreBoth{false};
  bool exploreFirst{false};
  bool exploreSecond{false};
};

struct ExploreState {
  int first;
  int second;
  bool followFirst;
  bool followSecond;
};

inline int divUp(int x, int y) {
  return (x + y - 1) / y;
}

__device__
inline size_t stateToIndex(
    const int first,
    const int second,
    int numFirst,
    bool followFirst,
    bool followSecond) {
  size_t offset = followFirst ? 1 : (followSecond ? 2 : 0);
  return 3 * (numFirst * second + first) + offset;
}

__device__
inline ExploreState indexToState(size_t n, int numFirst) {
  ExploreState state;
  auto offset = n % 3;
  state.followFirst = (offset == 1);
  state.followSecond = (offset == 2);
  n /= 3;
  state.first = n % numFirst;
  state.second = n / numFirst;
  return state;
}

bool checkAnyTrue(const HDSpan<bool>& flags) {
  thrust::device_ptr<const bool> tPtr(flags.data());
  return thrust::any_of(tPtr, tPtr + flags.size(), thrust::identity<bool>());
}

void setFalse(HDSpan<bool>& span) {
  if (span.size() != 0) {
    cuda::detail::fill(span.data(), false, span.size());
  }
}

std::tuple<int*, int> prefixSumScan(const bool* input, size_t numElts) {
  const size_t scanNumElts = numElts + 1;

  HDSpan<int> output(scanNumElts, 0, Device::CUDA);
  thrust::device_ptr<const bool> iPtr(input);
  thrust::device_ptr<int> oPtr(output.data());
  thrust::exclusive_scan(iPtr, iPtr + numElts, oPtr, (int) 0);

  int sum = 0;
  if (numElts > 0) {
    bool lastVal;
    CUDA_CHECK(cudaMemcpy((void*)(&sum), (void* )(&(output[scanNumElts-2])), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy((void*)(&lastVal), (void* )(&(input[scanNumElts-2])), sizeof(bool), cudaMemcpyDeviceToHost));
    sum += lastVal;
  }
  CUDA_CHECK(cudaMemcpy((void*)(&(output[scanNumElts-1])),(void*)(&sum), sizeof(int), cudaMemcpyHostToDevice));

  return std::make_tuple(output.data(), sum);
}

std::tuple<int*, int> prefixSumScan(const int* input, size_t numElts) {
  const size_t scanNumElts = numElts + 1;

  HDSpan<int> output(scanNumElts, 0, Device::CUDA);
  thrust::device_ptr<const int> iPtr(input);
  thrust::device_ptr<int> oPtr(output.data());
  thrust::inclusive_scan(iPtr, iPtr + numElts, oPtr + 1);

  int sum = 0;
  CUDA_CHECK(cudaMemcpy((void *)(&sum), (void *)(&(output[scanNumElts-1])), sizeof(int), cudaMemcpyDeviceToHost));

  return std::make_tuple(output.data(), sum);
}

__device__ size_t binarySearchBinIndex(const int* bins, int size, int tid) {
  size_t lIdx = 0;
  size_t rIdx = size - 1;

  while (lIdx <= rIdx) {
    size_t intervalIdx = (lIdx + rIdx) / 2;
    const int lVal = bins[intervalIdx];
    const int rVal = bins[intervalIdx + 1];

    if (tid >= rVal) {
      lIdx = intervalIdx + 1;
    } else if (tid < lVal) {
      assert(intervalIdx >= 1);
      rIdx = intervalIdx - 1;
    } else {
      return intervalIdx;
    }
  }
  assert(false);
  return 0;
}

__global__
void calculateArcCrossProductForwardKernel(
      const HDSpan<int> arcOffsets1,
      const HDSpan<int> arcOffsets2,
      const HDSpan<int> exploreIndices,
      int* arcCrossProductOffset,
      int numNodesFirst) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < exploreIndices.size()) {
    auto state = indexToState(exploreIndices[gTid], numNodesFirst);
    const int numArcsFirst = arcOffsets1[state.first + 1] - arcOffsets1[state.first];
    const int numArcsSecond = arcOffsets2[state.second + 1] - arcOffsets2[state.second];

    arcCrossProductOffset[gTid] = numArcsFirst * numArcsSecond;
    if (numArcsSecond == 0 && !state.followSecond) {
      arcCrossProductOffset[gTid] = numArcsFirst;
    }
    if (numArcsFirst == 0 && !state.followFirst) {
      arcCrossProductOffset[gTid] = numArcsSecond;
    }
  }
}

__global__
void calculateArcCrossProductBackwardKernel(
      const HDSpan<int> arcOffsets1,
      const HDSpan<int> arcOffsets2,
      const HDSpan<int> exploreIndices,
      int* arcCrossProductOffset,
      int numNodesFirst) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < exploreIndices.size()) {
    auto state = indexToState(exploreIndices[gTid], numNodesFirst);
    const int numArcsFirst = arcOffsets1[state.first + 1] - arcOffsets1[state.first];
    const int numArcsSecond = arcOffsets2[state.second + 1] - arcOffsets2[state.second];

    if (state.followFirst) {
      arcCrossProductOffset[gTid] = numArcsFirst;
    } else if (state.followSecond) {
      arcCrossProductOffset[gTid] = numArcsSecond;
    } else {
      arcCrossProductOffset[gTid] = numArcsFirst * numArcsSecond;
    }
  }
}

// Takes a pair of nodes, where each member of pair comes from a different
// graph and calculate a vector of number of arcs in the cross product of
// arcs outgoing from each pair.
int* calculateArcCrossProductOffset(
    const HDSpan<int>& exploreIndices,
    const GraphData g1,
    const GraphData g2,
    bool inOrOutArc) {

  int numToExploreNodePair = exploreIndices.size();
  int* arcCrossProductOffset;
  CUDA_CHECK(cudaMalloc((void **)(&(arcCrossProductOffset)), sizeof(int) * numToExploreNodePair));

  const int NT = 128;
  const int gridSize = divUp(numToExploreNodePair, NT);

  if (inOrOutArc) {
    calculateArcCrossProductBackwardKernel<<<gridSize, NT, 0, 0>>>(
        g1.inArcOffset, g2.inArcOffset, exploreIndices,
        arcCrossProductOffset, g1.numNodes);
  } else {
    calculateArcCrossProductForwardKernel<<<gridSize, NT, 0, 0>>>(
        g1.outArcOffset, g2.outArcOffset, exploreIndices,
        arcCrossProductOffset, g1.numNodes);
  }

  return arcCrossProductOffset;
}

// This function needs to be thread safe since multiple threads can
// can call it and they will overlap on curIdx and dstIdx
__device__
void calculateNumArcsAndNodesToExplore(
    int curIdx,
    int dstIdx,
    const HDSpan<bool> reachable,
    HDSpan<bool> newNodes,
    bool* toExplore,
    int* numOutArcs,
    int* numInArcs) {
  if (reachable[dstIdx]) {
    if (!newNodes[dstIdx]) {
      newNodes[dstIdx] = true;
      toExplore[dstIdx] = true;
    }

    // These are atomic increments
    // numOutArcs[curIdx]++;
    // numInArcs[dstIdx]++;
    atomicAdd(&(numOutArcs[curIdx]), 1);
    atomicAdd(&(numInArcs[dstIdx]), 1);
  }
}

// This function needs to be thread safe since multiple threads can
// can call it
__device__
void generateCombinedGraphArcs(
    int dstIdx,
    int curIdx,
    const int2& arcPair,
    const HDSpan<bool> reachable,
    const int* newNodesOffset,
    int* gradInfoFirst,
    int* gradInfoSecond,
    GraphData newGraphDP,
    float* weights,
    int ilabel,
    int olabel,
    float weight) {
  if (reachable[dstIdx]) {
    // Both of these increments are atomic
    // int inArcIdx = newGraphDP.inArcOffset[newNodesOffset[dstIdx]]++;
    // int outArcIdx = newGraphDP.outArcOffset[newNodesOffset[curIdx]]++;

    int inArcIdx = atomicAdd(&(newGraphDP.inArcOffset[newNodesOffset[dstIdx]]), 1);
    int outArcIdx = atomicAdd(&(newGraphDP.outArcOffset[newNodesOffset[curIdx]]), 1);

    // outArcIdx is also the arc identifier
    newGraphDP.outArcs[outArcIdx] = outArcIdx;
    newGraphDP.inArcs[inArcIdx] = outArcIdx;

    // Fill in everything else for this arc
    newGraphDP.ilabels[outArcIdx] = ilabel;
    newGraphDP.olabels[outArcIdx] = olabel;
    newGraphDP.srcNodes[outArcIdx] = newNodesOffset[curIdx];
    newGraphDP.dstNodes[outArcIdx] = newNodesOffset[dstIdx];
    weights[outArcIdx] = weight;

    gradInfoFirst[outArcIdx] = arcPair.x;
    gradInfoSecond[outArcIdx] = arcPair.y;
  }
}

__global__ 
void findReachableKernel(
      const GraphData g1,
      const GraphData g2,
      const int* arcCrossProductOffset,
      const HDSpan<int> exploreIndices,
      int totalArcs,
      HDSpan<bool> toExplore,
      HDSpan<bool> reachable) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    auto idx = binarySearchBinIndex(arcCrossProductOffset, exploreIndices.size(), gTid);
    auto state = indexToState(exploreIndices[idx], g1.numNodes);
    int localIdx = gTid - arcCrossProductOffset[idx];
    assert(localIdx >= 0);

    int firstArcIdx, secondArcIdx;
    if (state.followFirst) {
      firstArcIdx = g1.inArcs[g1.inArcOffset[state.first] + localIdx];
    } else if (state.followSecond) {
      secondArcIdx = g2.inArcs[g2.inArcOffset[state.second] + localIdx];
    } else {
      auto numArcsFirst =
          g1.inArcOffset[state.first + 1] - g1.inArcOffset[state.first];

      firstArcIdx = g1.inArcs[g1.inArcOffset[state.first] + (localIdx % numArcsFirst)];
      secondArcIdx = g2.inArcs[g2.inArcOffset[state.second] + (localIdx / numArcsFirst)];
    }
    if (!(state.followFirst || state.followSecond) &&
        (g1.olabels[firstArcIdx] == g2.ilabels[secondArcIdx])) {
      const int idx = stateToIndex(
          g1.srcNodes[firstArcIdx],
          g2.srcNodes[secondArcIdx],
          g1.numNodes,
          false, false);

      if (!reachable[idx]) {
        reachable[idx] = true;
        toExplore[idx] = true;
      }
      if (g1.olabels[firstArcIdx] != epsilon) {
        if (!reachable[idx + 1]) {
          reachable[idx + 1] = true;
          toExplore[idx + 1] = true;
        }
        if (!reachable[idx + 2]) {
          reachable[idx + 2] = true;
          toExplore[idx + 2] = true;
        }
      }
    } else if (state.followFirst && (g1.olabels[firstArcIdx] == epsilon)) {
      const int idx = stateToIndex(
          g1.srcNodes[firstArcIdx], state.second, g1.numNodes, false, false);
      if (!reachable[idx]) {
        reachable[idx] = true;
        toExplore[idx] = true;
      }
      if (!reachable[idx + 1]) {
        reachable[idx + 1] = true;
        toExplore[idx + 1] = true;
      }
    } else if (state.followSecond && (g2.ilabels[secondArcIdx] == epsilon)) {
      const int idx = stateToIndex(
          state.first, g2.srcNodes[secondArcIdx], g1.numNodes, false, false);
      if (!reachable[idx]) {
        reachable[idx] = true;
        toExplore[idx] = true;
      }
      if (!reachable[idx + 2]) {
        reachable[idx + 2] = true;
        toExplore[idx + 2] = true;
      }
    }
  }
}

__device__
ExploreNodeAndArcs getArcPairAndExploreState(
    const GraphData& g1,
    const GraphData& g2,
    const int* arcCrossProductOffset,
    const HDSpan<int>& exploreIndices,
    int gTid) {
  ExploreNodeAndArcs res;
  auto idx = binarySearchBinIndex(arcCrossProductOffset, exploreIndices.size(), gTid);
  auto state = indexToState(exploreIndices[idx], g1.numNodes);
  res.nodePair = make_int2(state.first, state.second);
  int localIdx = gTid - arcCrossProductOffset[idx];
  assert(localIdx >= 0);

  auto numArcsFirst =
      g1.outArcOffset[state.first + 1] - g1.outArcOffset[state.first];
  auto numArcsSecond =
      g2.outArcOffset[state.second + 1] - g2.outArcOffset[state.second];
  assert(numArcsFirst > 0 || numArcsSecond > 0);

  res.nodeIdx = exploreIndices[idx];
  int firstArcIdx, secondArcIdx;
  if (numArcsFirst > 0 && numArcsSecond > 0 ) {
    // Explore everything
    res.exploreFirst = !state.followSecond && (localIdx / numArcsFirst) == 0;
    res.exploreSecond = !state.followFirst && (localIdx % numArcsFirst) == 0;
    firstArcIdx = g1.outArcs[g1.outArcOffset[state.first] + localIdx % numArcsFirst];
    secondArcIdx = g2.outArcs[g2.outArcOffset[state.second] + localIdx / numArcsFirst];
    res.exploreBoth = g1.olabels[firstArcIdx] == g2.ilabels[secondArcIdx] &&
      (g1.olabels[firstArcIdx] != epsilon || !(state.followFirst || state.followSecond));
  } else if (numArcsSecond == 0 && !state.followSecond) {
    // Explore first
    res.exploreFirst = true;
    firstArcIdx = g1.outArcs[g1.outArcOffset[state.first] + localIdx];
  } else if (numArcsFirst == 0 && !state.followFirst) {
    // Explore second
    res.exploreSecond = true;
    secondArcIdx = g2.outArcs[g2.outArcOffset[state.second] + localIdx];
  }
  if (res.exploreFirst) {
    res.exploreFirst &= (g1.olabels[firstArcIdx] == epsilon);
  }
  if (res.exploreSecond) {
    res.exploreSecond &= (g2.ilabels[secondArcIdx] == epsilon);
  }
  res.arcPair = make_int2(firstArcIdx, secondArcIdx);
  return res;
}

__global__ 
void computeValidNodeAndArcKernel(
      const GraphData g1,
      const GraphData g2,
      const int* arcCrossProductOffset,
      const HDSpan<int> exploreIndices,
      const HDSpan<bool> reachable,
      int totalArcs,
      HDSpan<bool> toExplore,
      HDSpan<bool> newNodes,
      int* numInArcs,
      int* numOutArcs) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    auto res = getArcPairAndExploreState(
        g1, g2, arcCrossProductOffset, exploreIndices, gTid);
    const int firstArcIdx = res.arcPair.x;
    const int secondArcIdx = res.arcPair.y;
    if (res.exploreBoth) {
      const int dstIdx = stateToIndex(
          g1.dstNodes[firstArcIdx],
          g2.dstNodes[secondArcIdx],
          g1.numNodes,
          false,
          false);

      calculateNumArcsAndNodesToExplore(
          res.nodeIdx,
          dstIdx,
          reachable,
          newNodes,
          toExplore.data(),
          numOutArcs,
          numInArcs);
    }

    if (res.exploreFirst) {
      const int dstIdx = stateToIndex(
          g1.dstNodes[firstArcIdx], res.nodePair.y, g1.numNodes, true, false);

      calculateNumArcsAndNodesToExplore(
          res.nodeIdx,
          dstIdx,
          reachable,
          newNodes,
          toExplore.data(),
          numOutArcs,
          numInArcs);
    }

    if (res.exploreSecond) {
      const int dstIdx = stateToIndex(
          res.nodePair.x, g2.dstNodes[secondArcIdx], g1.numNodes, false, true);

      calculateNumArcsAndNodesToExplore(
          res.nodeIdx,
          dstIdx,
          reachable,
          newNodes,
          toExplore.data(),
          numOutArcs,
          numInArcs);
    }
  }
}

__global__ 
void generateNodeAndArcKernel(
      const GraphData g1,
      const GraphData g2,
      const float* weightsFirst,
      const float* weightsSecond,
      const int* arcCrossProductOffset,
      const HDSpan<int> exploreIndices,
      const HDSpan<bool> reachable,
      int totalArcs,
      GraphData newGraph,
      float* weights,
      int* gradInfoFirst,
      int* gradInfoSecond,
      int* newNodesOffset) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    auto res = getArcPairAndExploreState(
        g1, g2, arcCrossProductOffset, exploreIndices, gTid);
    const int firstArcIdx = res.arcPair.x;
    const int secondArcIdx = res.arcPair.y;
    if (res.exploreBoth) {
      const int dstIdx = stateToIndex(
          g1.dstNodes[firstArcIdx],
          g2.dstNodes[secondArcIdx],
          g1.numNodes,
          false,
          false);

      generateCombinedGraphArcs(
          dstIdx,
          res.nodeIdx,
          make_int2(firstArcIdx, secondArcIdx),
          reachable,
          newNodesOffset,
          gradInfoFirst,
          gradInfoSecond,
          newGraph,
          weights,
          g1.ilabels[firstArcIdx],
          g2.olabels[secondArcIdx],
          weightsFirst[firstArcIdx] + weightsSecond[secondArcIdx]);
    }

    if (res.exploreFirst) {
      const int dstIdx = stateToIndex(
          g1.dstNodes[firstArcIdx], res.nodePair.y, g1.numNodes, true, false);

      generateCombinedGraphArcs(
          dstIdx,
          res.nodeIdx,
          make_int2(firstArcIdx, -1),
          reachable,
          newNodesOffset,
          gradInfoFirst,
          gradInfoSecond,
          newGraph,
          weights,
          g1.ilabels[firstArcIdx],
          epsilon,
          weightsFirst[firstArcIdx]);
    }

    if (res.exploreSecond) {
      const int dstIdx = stateToIndex(
          res.nodePair.x, g2.dstNodes[secondArcIdx], g1.numNodes, false, true);

      generateCombinedGraphArcs(
          dstIdx,
          res.nodeIdx,
          make_int2(-1, secondArcIdx),
          reachable,
          newNodesOffset,
          gradInfoFirst,
          gradInfoSecond,
          newGraph,
          weights,
          epsilon,
          g2.olabels[secondArcIdx],
          weightsSecond[secondArcIdx]);
    }
  }
}

__global__
void setStartAndAccept(
    const GraphData g1,
    const GraphData g2,
    const HDSpan<int> exploreIndices,
    GraphData newGraph) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < exploreIndices.size()) {
    auto state = indexToState(exploreIndices[gTid], g1.numNodes);
    newGraph.start[gTid] = g1.start[state.first] && g2.start[state.second]
        && !(state.followFirst || state.followSecond);
    newGraph.accept[gTid] = g1.accept[state.first] && g2.accept[state.second];
  }
}

__global__
void calculateNumArcsKernel(
  const HDSpan<int> nodeIndices,
  const int* inputInArcs,
  const int* inputOutArcs,
  HDSpan<int> outputInArcs,
  HDSpan<int> outputOutArcs) {

  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < nodeIndices.size()) {
    const int index = nodeIndices[gTid];
    outputInArcs[gTid] = inputInArcs[index];
    outputOutArcs[gTid] = inputOutArcs[index];
  }
}

__global__
void findReachableInitKernel(
    const HDSpan<int> idsFirst,
    const HDSpan<int> idsSecond,
    HDSpan<bool> reachable,
    HDSpan<bool> toExplore,
    int numNodesFirst) {

  const int fid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sid = blockIdx.y * blockDim.y + threadIdx.y;

  if (fid < idsFirst.size() && sid < idsSecond.size()) {
    auto idx = stateToIndex(
        idsFirst[fid], idsSecond[sid], numNodesFirst, false, false);
    toExplore[idx] = true;
    reachable[idx] = true;
    toExplore[idx + 1] = true;
    reachable[idx + 1] = true;
    toExplore[idx + 2] = true;
    reachable[idx + 2] = true;
  }
}

void findReachableInit(
    const GraphData& g1,
    const GraphData& g2,
    HDSpan<bool> reachable,
    HDSpan<bool> toExplore) {
  int NT = 16; 
  auto blocks = dim3(
      divUp(g1.acceptIds.size(), NT), divUp(g2.acceptIds.size(), NT));
  auto threads = dim3(NT, NT);
  findReachableInitKernel<<<blocks, threads>>>(g1.acceptIds, g2.acceptIds,
    reachable, toExplore, g1.numNodes);
}

__global__
void secondPassInitKernel(
    const HDSpan<int> idsFirst,
    const HDSpan<int> idsSecond,
    const HDSpan<bool> reachable,
    HDSpan<bool> toExplore,
    HDSpan<bool> newNodes,
    int numNodesFirst) {
  const int fid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sid = blockIdx.y * blockDim.y + threadIdx.y;

  if (fid < idsFirst.size() && sid < idsSecond.size()) {
    auto idx = stateToIndex(
        idsFirst[fid], idsSecond[sid], numNodesFirst, false, false);
    if (reachable[idx]) {
      toExplore[idx] = true;
      newNodes[idx] = true;
    }
  }
}

void secondPassInit(
    const GraphData& g1,
    const GraphData& g2,
    const HDSpan<bool> reachable,
    HDSpan<bool> toExplore,
    HDSpan<bool> newNodes) {
  int NT = 16; 
  auto blocks = dim3(
      divUp(g1.startIds.size(), NT), divUp(g2.startIds.size(), NT));
  auto threads = dim3(NT, NT);
  secondPassInitKernel<<<blocks, threads>>>(g1.startIds, g2.startIds,
    reachable, toExplore, newNodes, g1.numNodes);
}

__global__
void gradKernel(
    int* arcIds,
    const float* deltas,
    float* grad,
    size_t numArcs) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numArcs && arcIds[gTid] >= 0) {
    atomicAdd(grad + arcIds[gTid], deltas[gTid]);
  }
}

void calcGrad(Graph& g, int* arcIds, const Graph& deltas) {
  if (!g.calcGrad()) {
    return;
  }

  HDSpan<float> grad(g.numArcs(), 0.0, Device::CUDA);
  const int NT = 128;
  const int gridSize = divUp(deltas.numArcs(), NT);
  gradKernel<<<gridSize, NT, 0, 0>>>(
      arcIds, deltas.weights(), grad.data(), deltas.numArcs());
  g.addGrad(grad.data());
  grad.clear();
}

__global__
void  boolToIndicesKernel(
    HDSpan<int> ids, const int* counts, const HDSpan<bool> vals, size_t size) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < size && vals[gTid]) {
    ids[counts[gTid]] = gTid;
  }
}

auto boolToIndices(const HDSpan<bool>& vals) {
  int* counts;
  int numTrue;
  std::tie(counts, numTrue) = prefixSumScan(vals.data(), vals.size());

  const int NT = 128;
  const int gridSize = divUp(vals.size(), NT);

  HDSpan<int> ids(numTrue, Device::CUDA);
  boolToIndicesKernel<<<gridSize, NT, 0, 0>>>(ids, counts, vals, vals.size());
  CUDA_CHECK(cudaFree(counts));
  return ids;
}

} // namespace


Graph compose(const Graph& first, const Graph& second) {
  auto nGraph = Graph(nullptr, {first, second});
  auto& nData = nGraph.getData();

  auto g1 = first.getData();
  auto g2 = second.getData();
  
  const int numAllPairNodes = 3 * first.numNodes() * second.numNodes();
  const int numNodesFirst = first.numNodes();

  // Fixed number of CUDA threads and stream for all kernels
  const int NT = 128;

  //////////////////////////////////////////////////////////////////////////
  // Step 1: Data parallel findReachable
  //////////////////////////////////////////////////////////////////////////
  HDSpan<bool> reachable(numAllPairNodes, false, Device::CUDA);
  HDSpan<bool> toExplore(numAllPairNodes, false, Device::CUDA);
  findReachableInit(g1, g2, reachable, toExplore); 

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrue(toExplore)) {

    // Convert bits set in toExplore to indices 
    auto exploreIndices = boolToIndices(toExplore);

    int* arcCrossProductIndex = calculateArcCrossProductOffset(
        exploreIndices, g1, g2, true);

    int* arcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffset, totalArcs) =
        prefixSumScan(arcCrossProductIndex, exploreIndices.size());

    CUDA_CHECK(cudaFree(arcCrossProductIndex));

    // Reset so pristine state for next frontier to explore
    setFalse(toExplore);

    if (totalArcs > 0) {

      const int gridSize = divUp(totalArcs, NT);

      findReachableKernel<<<gridSize, NT, 0, 0>>>(
          g1, g2, arcCrossProductOffset, exploreIndices,
          totalArcs, toExplore, reachable);
    }

    exploreIndices.clear();
    CUDA_CHECK(cudaFree(arcCrossProductOffset));
  } // end while for findReachable

  //////////////////////////////////////////////////////////////////////////
  // Step 2: Compute a) valid nodes in combined graph
  //                 b) Number of in and out arcs in combined graph
  // This information is used to generate offsets for nodes and arcs
  // in the combined graph
  //////////////////////////////////////////////////////////////////////////

  HDSpan<bool> newNodes(numAllPairNodes, 0.0, Device::CUDA);
  int* numOutArcs;
  int* numInArcs;

  CUDA_CHECK(cudaMalloc((void **)(&numOutArcs), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMalloc((void **)(&numInArcs), sizeof(int) * numAllPairNodes));

  CUDA_CHECK(cudaMemset((void*)numOutArcs, 0, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)numInArcs, 0, sizeof(int) * numAllPairNodes));

  setFalse(toExplore);

  secondPassInit(g1, g2, reachable, toExplore, newNodes);

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrue(toExplore)) {

    // Convert bits set in toExplore to node pairs
    auto exploreIndices =  boolToIndices(toExplore);

    int* arcCrossProductIndex = calculateArcCrossProductOffset(
        exploreIndices, g1, g2, false);

    int* arcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffset, totalArcs) =
      prefixSumScan(arcCrossProductIndex, exploreIndices.size());

    CUDA_CHECK(cudaFree(arcCrossProductIndex));

    // Reset so pristine state for next frontier to explore
    setFalse(toExplore);

    if (totalArcs > 0) {

      const int gridSize = divUp(totalArcs, NT);

      computeValidNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(g1, g2,
        arcCrossProductOffset, exploreIndices, reachable, totalArcs,
        toExplore, newNodes, numInArcs, numOutArcs);
    }

    exploreIndices.clear();
    CUDA_CHECK(cudaFree(arcCrossProductOffset));
  }
  toExplore.clear();
  reachable.clear();

  //////////////////////////////////////////////////////////////////////////
  // Step 3: Generate offsets for nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////

  int totalNodes;
  int* newNodesOffset;
  std::tie(newNodesOffset, totalNodes) = prefixSumScan(newNodes.data(), numAllPairNodes);

  nData.numNodes = totalNodes;
  nData.start.resize(totalNodes);
  nData.accept.resize(totalNodes);
  nData.inArcOffset.resize(totalNodes + 1);
  nData.outArcOffset.resize(totalNodes + 1);

  // Convert bits to indices
  auto exploreIndices =  boolToIndices(newNodes);

  // Generate offsets for nodes and arcs
  if (exploreIndices.size() > 0) {
    const int NT = 128;
    const int gridSize = divUp(exploreIndices.size(), NT);

    calculateNumArcsKernel<<<gridSize, NT, 0, 0>>>(exploreIndices,
      numInArcs, numOutArcs, nData.inArcOffset, nData.outArcOffset);
  }
  CUDA_CHECK(cudaFree(numOutArcs));
  CUDA_CHECK(cudaFree(numInArcs));

  int totalInArcs;
  int totalOutArcs;

  int* inArcOffsetGPU;
  int* outArcOffsetGPU;

  std::tie(inArcOffsetGPU, totalInArcs) = prefixSumScan(nData.inArcOffset.data(), totalNodes);

  std::tie(outArcOffsetGPU, totalOutArcs) = prefixSumScan(nData.outArcOffset.data(), totalNodes);
  assert(totalInArcs == totalOutArcs);
  nData.numArcs = totalOutArcs;
  nData.inArcs.resize(totalOutArcs);
  nData.outArcs.resize(totalOutArcs);
  nData.ilabels.resize(totalOutArcs);
  nData.olabels.resize(totalOutArcs);
  nData.srcNodes.resize(totalOutArcs);
  nData.dstNodes.resize(totalOutArcs);
  nGraph.getWeights().resize(totalOutArcs);

  nData.inArcOffset.copy(inArcOffsetGPU);
  nData.outArcOffset.copy(outArcOffsetGPU);

  auto gradInfo = std::make_shared<GradInfo>();
  CUDA_CHECK(cudaMalloc((void **)(&gradInfo->first), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&gradInfo->second), sizeof(int) * totalOutArcs));

  //////////////////////////////////////////////////////////////////////////
  // Step 4: Generate nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////
  int* arcCrossProductIndex = calculateArcCrossProductOffset(
      exploreIndices, g1, g2, false);

  int* arcCrossProductOffset;
  int totalArcs;

  std::tie(arcCrossProductOffset, totalArcs) =
      prefixSumScan(arcCrossProductIndex, exploreIndices.size());
  CUDA_CHECK(cudaFree(arcCrossProductIndex));

  if (exploreIndices.size() > 0) {
    setFalse(nData.start);
    setFalse(nData.accept);

    const int gridSize = divUp(exploreIndices.size(), NT);
    setStartAndAccept<<<gridSize, NT, 0, 0>>>(g1, g2, exploreIndices, nData);
    nData.startIds = boolToIndices(nData.start);
    nData.acceptIds = boolToIndices(nData.accept);
  }
  if (totalArcs > 0) {
    const int gridSize = divUp(totalArcs, NT);

    generateNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(g1, g2,
        first.weights(), second.weights(), arcCrossProductOffset, exploreIndices, newNodes,
        totalArcs, nData, nGraph.weights(), gradInfo->first, gradInfo->second, newNodesOffset);
  }

  exploreIndices.clear();
  CUDA_CHECK(cudaFree(arcCrossProductOffset));

  // Reset incremented offsets to original value
  nData.inArcOffset.copy(inArcOffsetGPU);
  nData.outArcOffset.copy(outArcOffsetGPU);

  newNodes.clear();
  CUDA_CHECK(cudaFree(newNodesOffset));
  CUDA_CHECK(cudaFree(inArcOffsetGPU));
  CUDA_CHECK(cudaFree(outArcOffsetGPU));

  auto gradFunc = [gradInfo](std::vector<Graph>& inputs, Graph deltas) {
    calcGrad(inputs[0], gradInfo->first, deltas);
    calcGrad(inputs[1], gradInfo->second, deltas);
  };
  nGraph.setGradFunc(std::move(gradFunc));
  return nGraph;
}

} // namespace detail
} // namespace cuda 
} // namespace gtn
