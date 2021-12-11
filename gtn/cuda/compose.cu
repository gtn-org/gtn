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

struct nodeAndArcPairGPU {
  int2 nodePair;
  int2 arcPair;
  int2 checkEpsilonArcPair;
  bool checkArcPair;
  bool isValid;
};

inline int div_up(int x, int y) {
  return (x + y - 1) / y;
}

__device__ __host__
inline int TwoDToOneDIndex(int n1, int n2, int n1Extent) {
  assert(n1 < n1Extent);
  return n1 + n2 * n1Extent;
}

__device__
inline int2 oneDToTwoDIndex(int n, int n1Extent) {
  assert(n1Extent > 0);
  const int n2 = n / n1Extent;
  const int n1 = n % n1Extent;
  return make_int2(n1, n2);
}


bool checkAnyTrue(const HDSpan<int>& flags) {
  thrust::device_ptr<const int> tPtr(flags.data());
  return thrust::any_of(tPtr, tPtr + flags.size(), thrust::identity<int>());
}

void setZero(HDSpan<int>& span) {
  cuda::detail::fill(span.data(), 0, span.size());
}

std::tuple<int*, int> prefixSumScan(const int* input, size_t numElts) {
  const size_t scanNumElts = numElts + 1;

  int *output;
  CUDA_CHECK(cudaMalloc((void **)(&(output)), sizeof(int) * scanNumElts));
  CUDA_CHECK(cudaMemcpy((void *)(output), (void *)(input), sizeof(int) * numElts, cudaMemcpyDeviceToDevice));
  thrust::device_ptr<int> tPtr(output);
  thrust::exclusive_scan(tPtr, tPtr + scanNumElts, tPtr);

  int sum = 0;
  CUDA_CHECK(cudaMemcpy((void *)(&sum), (void *)(&(output[scanNumElts-1])), sizeof(int), cudaMemcpyDeviceToHost));

  return std::make_tuple(output, sum);
}

__device__
nodeAndArcPairGPU computeNodeAndArcPair(
    int tid,
    const int* arcCrossProductOffset,
    const HDSpan<int> arcOffsets1,
    const HDSpan<int> arcOffsets2,
    const HDSpan<int> exploreIndices) {

  nodeAndArcPairGPU result;
  result.checkArcPair = false;
  result.checkEpsilonArcPair = make_int2(false, false);
  result.isValid = false;

  int localIdx, numArcs;
  size_t intervalIdx;

  // There should be at least two values to form a range
  size_t numArcCrossProductOffset = exploreIndices.size() + 1;
  assert(numArcCrossProductOffset >= 2);
  const size_t numIntervals = numArcCrossProductOffset - 1;

  // Binary search
  {
    size_t lIdx = 0;
    size_t rIdx = numIntervals - 1;

    while (lIdx <= rIdx) {
      intervalIdx = (lIdx + rIdx) / 2;
      const int lVal = arcCrossProductOffset[intervalIdx];
      const int rVal = arcCrossProductOffset[intervalIdx + 1];

      if (tid >= rVal) {
        lIdx = intervalIdx + 1;
      } else if (tid < lVal) {
        assert(intervalIdx >= 1);
        rIdx = intervalIdx - 1;
      } else {
        assert((lVal <= tid) && (tid < rVal));

        result.isValid = true;
        result.nodePair = oneDToTwoDIndex(
            exploreIndices[intervalIdx], arcOffsets1.size() - 1);

        // The range of idx is from
        // [0, toExploreNumArcsFirst[intervalIdx] * toExploreNumArcsSecond[intervalIdx])
        localIdx = tid - lVal;
        numArcs = rVal - lVal;

        break;
      }
    }
  }

  if (result.isValid == true) {
    auto toExploreNumArcsFirst =
        arcOffsets1[result.nodePair.x + 1] - arcOffsets1[result.nodePair.x];
    auto toExploreNumArcsSecond =
        arcOffsets2[result.nodePair.y + 1] - arcOffsets2[result.nodePair.y];
    assert(localIdx >= 0);
    assert(localIdx < numArcs);
    assert(numArcs > 0);

    const int arcProd = toExploreNumArcsFirst * toExploreNumArcsSecond;

    if (numArcs == arcProd) {
      result.checkArcPair = true;

      // We map the tids to 2D grid where the
      // x-axis is toExploreNumArcsFirst[i] (row)
      // y-axis is toExploreNumArcsSecond[i] (column)
      assert(toExploreNumArcsFirst > 0);
      result.arcPair = make_int2(
        localIdx % toExploreNumArcsFirst,
        localIdx / toExploreNumArcsFirst);

      // Pick the tids from the first row since we need only one
      // tid per arc of the node from the first graph to check for
      // epsilon
      if (localIdx < toExploreNumArcsFirst) {
        result.checkEpsilonArcPair.x = true;
      }

      // Pick the tids from the first column since we need only one
      // tid per arc of the node from the first graph to check for
      // epsilon
      if ((localIdx % toExploreNumArcsFirst) == 0) {
        result.checkEpsilonArcPair.y = true;
      }
    } else if ((arcProd == 0) && (numArcs == toExploreNumArcsFirst)) {
      // TODO: Likely not the brightest idea to use -1 as sentinel
      result.arcPair = make_int2(localIdx, -1);
      result.checkEpsilonArcPair.x = true;
    } else if ((arcProd == 0) && (numArcs == toExploreNumArcsSecond)) {
      // TODO: Likely not the brightest idea to use -1 as sentinel
      result.arcPair = make_int2(-1, localIdx);
      result.checkEpsilonArcPair.y = true;
    }
  }

  return result;
}

__global__
void calculateArcCrossProductOffsetKernel(
      const GraphData g1,
      const GraphData g2,
      const HDSpan<int> exploreIndices,
      int* arcCrossProductOffset,
      bool inOrOutArc) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < exploreIndices.size()) {
    int2 idx = oneDToTwoDIndex(exploreIndices[gTid], g1.numNodes);
    auto arcOffsets = inOrOutArc ? g1.inArcOffset.data() : g1.outArcOffset.data();
    const int numArcsFirst = arcOffsets[idx.x + 1] - arcOffsets[idx.x];

    arcOffsets = inOrOutArc ? g2.inArcOffset.data() : g2.outArcOffset.data();
    const int numArcsSecond = arcOffsets[idx.y + 1] - arcOffsets[idx.y];

    // Even when numArcsFirst or numArcsSecond is 0 we have to consider
    // the case when the other graph has arcs with epsilon label
    if (numArcsFirst != 0 && numArcsSecond == 0) {
      arcCrossProductOffset[gTid] = numArcsFirst;
    } else if (numArcsFirst == 0 && numArcsSecond != 0) {
      arcCrossProductOffset[gTid] = numArcsSecond;
    } else {
      arcCrossProductOffset[gTid] = numArcsFirst * numArcsSecond;
    }
  }
}

// Takes a pair of nodes, where each member of pair comes from a different
// graph and calculate a vector of number of arcs in the cross product of
// arcs outgoing from each pair.
// This should be a kernel call
int* calculateArcCrossProductOffsetGPU(
    const HDSpan<int>& exploreIndices,
    const GraphData g1,
    const GraphData g2,
    bool inOrOutArc) {

  int numToExploreNodePair = exploreIndices.size();
  int* arcCrossProductOffset;
  CUDA_CHECK(cudaMalloc((void **)(&(arcCrossProductOffset)), sizeof(int) * numToExploreNodePair));

  const int NT = 128;
  const int gridSize = div_up(numToExploreNodePair, NT);

  calculateArcCrossProductOffsetKernel<<<gridSize, NT, 0, 0>>>(
      g1, g2, exploreIndices, arcCrossProductOffset, inOrOutArc);

  return arcCrossProductOffset;
}

// This function needs to be thread safe since multiple threads can
// can call it and they will overlap on curIdx and dstIdx
__device__
void calculateNumArcsAndNodesToExplore(
    int curIdx,
    int dstIdx,
    const HDSpan<int> reachable,
    int* newNodes,
    int* toExplore,
    int* numOutArcs,
    int* numInArcs) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodes
    int oldVal = atomicCAS(&(newNodes[dstIdx]), false, true);
    if (!oldVal) {
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
    const HDSpan<int> reachable,
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
      const GraphData graphDP1GPU,
      const GraphData graphDP2GPU,
      const int* arcCrossProductOffsetGPU,
      const HDSpan<int> exploreIndices,
      int totalArcs,
      HDSpan<int> toExploreGPU,
      HDSpan<int> reachable,
      int* epsilonMatchedGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    int numNodesFirst = graphDP1GPU.numNodes;
    nodeAndArcPairGPU result = computeNodeAndArcPair(
        gTid, arcCrossProductOffsetGPU, graphDP1GPU.inArcOffset,
        graphDP2GPU.inArcOffset, exploreIndices);

    // printf("tid = %d, valid = %d\n", gTid, result.isValid);
    // Does this node pair match?
    if (result.isValid) {
      int inArcOffset = graphDP1GPU.inArcOffset[result.nodePair.x];
      const int firstArcIdx = graphDP1GPU.inArcs[inArcOffset + result.arcPair.x];

      inArcOffset = graphDP2GPU.inArcOffset[result.nodePair.y];
      const int secondArcIdx = graphDP2GPU.inArcs[inArcOffset + result.arcPair.y];

      // printf("tid = %d, cp = %d\n", gTid, result.checkArcPair);

      if (result.checkArcPair &&
          (graphDP1GPU.olabels[firstArcIdx] == graphDP2GPU.ilabels[secondArcIdx])) {
        const int idx = TwoDToOneDIndex(
            graphDP1GPU.srcNodes[firstArcIdx],
            graphDP2GPU.srcNodes[secondArcIdx],
            numNodesFirst);

	// printf("tid = %d, idx = %d\n", gTid, idx);

        if (graphDP1GPU.olabels[firstArcIdx] == epsilon) {
          epsilonMatchedGPU[idx] = true;
        }

        // idx may not be unique amongst all threads.
        int oldVal = atomicCAS(&(reachable[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
      }

      // Only valid for arcs incoming to node from first graph
      if (result.checkEpsilonArcPair.x &&
          (graphDP1GPU.olabels[firstArcIdx] == epsilon)) {
        const int idx = TwoDToOneDIndex(
            graphDP1GPU.srcNodes[firstArcIdx], result.nodePair.y, numNodesFirst);
        int oldVal = atomicCAS(&(reachable[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
      }

      // Only valid for arcs incoming to node from second graph
      if (result.checkEpsilonArcPair.y &&
          (graphDP2GPU.ilabels[secondArcIdx] == epsilon)) {
        const int idx = TwoDToOneDIndex(
            result.nodePair.x, graphDP2GPU.srcNodes[secondArcIdx], numNodesFirst);
        int oldVal = atomicCAS(&(reachable[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
      }
    }
  }
}

__global__ 
void computeValidNodeAndArcKernel(
      const GraphData graphDP1GPU,
      const GraphData graphDP2GPU,
      const int* arcCrossProductOffsetGPU,
      const HDSpan<int> exploreIndices,
      const HDSpan<int> reachable,
      const int* epsilonMatchedGPU,
      int totalArcs,
      HDSpan<int> toExplore,
      int* newNodes,
      int* numInArcsGPU,
      int* numOutArcsGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    int numNodesFirst = graphDP1GPU.numNodes;
    // Map tid to corresponding node and arc pair
    // Search to find which node pair this tid will fall into
    nodeAndArcPairGPU result = computeNodeAndArcPair(
        gTid, arcCrossProductOffsetGPU, graphDP1GPU.outArcOffset,
        graphDP2GPU.outArcOffset, exploreIndices);

    if (result.isValid) {
      int outArcOffset = graphDP1GPU.outArcOffset[result.nodePair.x];
      const int firstArcIdx = graphDP1GPU.outArcs[outArcOffset + result.arcPair.x];

      outArcOffset = graphDP2GPU.outArcOffset[result.nodePair.y];
      const int secondArcIdx =
          graphDP2GPU.outArcs[outArcOffset + result.arcPair.y];

      const bool epsilonMatch = epsilonMatchedGPU[TwoDToOneDIndex(
          result.nodePair.x, result.nodePair.y, numNodesFirst)];

      // Does this node pair match?
      // Skip epsilon matches
      if (result.checkArcPair &&
          (graphDP1GPU.olabels[firstArcIdx] == graphDP2GPU.ilabels[secondArcIdx])) {
        const int dstIdx = TwoDToOneDIndex(
            graphDP1GPU.dstNodes[firstArcIdx],
            graphDP2GPU.dstNodes[secondArcIdx],
            numNodesFirst);
        const int curIdx =
            TwoDToOneDIndex(result.nodePair.x, result.nodePair.y, numNodesFirst);

        // printf("krnl 1a dst %d cur %d\n", dstIdx, curIdx);

        // We track if any two arcs outgoing from this node pair match
        // on epsilon. We record if they do.
        if (graphDP1GPU.olabels[firstArcIdx] != epsilon) {
          calculateNumArcsAndNodesToExplore(
              curIdx,
              dstIdx,
              reachable,
              newNodes,
              toExplore.data(),
              numOutArcsGPU,
              numInArcsGPU);
        }
      }

      if (result.checkEpsilonArcPair.x &&
          (!epsilonMatch || graphDP2GPU.accept[result.nodePair.y] ||
           !graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP1GPU.olabels[firstArcIdx] == epsilon)) {
        const int dstIdx = TwoDToOneDIndex(
            graphDP1GPU.dstNodes[firstArcIdx], result.nodePair.y, numNodesFirst);
        const int curIdx =
            TwoDToOneDIndex(result.nodePair.x, result.nodePair.y, numNodesFirst);

        // printf("krnl 1b dst %d cur %d\n", dstIdx, curIdx);

        calculateNumArcsAndNodesToExplore(
            curIdx,
            dstIdx,
            reachable,
            newNodes,
            toExplore.data(),
            numOutArcsGPU,
            numInArcsGPU);
      }

      if (result.checkEpsilonArcPair.y &&
          (!epsilonMatch || graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP2GPU.ilabels[secondArcIdx] == epsilon)) {
        const int dstIdx = TwoDToOneDIndex(
            result.nodePair.x, graphDP2GPU.dstNodes[secondArcIdx], numNodesFirst);
        const int curIdx =
            TwoDToOneDIndex(result.nodePair.x, result.nodePair.y, numNodesFirst);

        // printf("krnl 1c dst %d cur %d\n", dstIdx, curIdx);

        calculateNumArcsAndNodesToExplore(
            curIdx,
            dstIdx,
            reachable,
            newNodes,
            toExplore.data(),
            numOutArcsGPU,
            numInArcsGPU);
      }
    }
  }
}

__global__ 
void generateNodeAndArcKernel(
      const GraphData graphDP1GPU,
      const GraphData graphDP2GPU,
      const float* weightsFirst,
      const float* weightsSecond,
      const int* arcCrossProductOffsetGPU,
      const HDSpan<int> exploreIndices,
      const HDSpan<int> reachable,
      const int* epsilonMatchedGPU,
      int totalArcs,
      GraphData newGraphDPGPU,
      float* weights,
      int* gradInfoFirstGPU,
      int* gradInfoSecondGPU,
      int* newNodesOffset
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    int numNodesFirst = graphDP1GPU.numNodes;
    // Map tid to corresponding node and arc pair
    // Search to find which node pair this tid will fall into
    nodeAndArcPairGPU result = computeNodeAndArcPair(
        gTid, arcCrossProductOffsetGPU, graphDP1GPU.outArcOffset,
        graphDP2GPU.outArcOffset, exploreIndices);

    if (result.isValid) {
      int outArcOffset = graphDP1GPU.outArcOffset[result.nodePair.x];
      const int firstArcIdx = graphDP1GPU.outArcs[outArcOffset + result.arcPair.x];

      outArcOffset = graphDP2GPU.outArcOffset[result.nodePair.y];
      const int secondArcIdx =
          graphDP2GPU.outArcs[outArcOffset + result.arcPair.y];

      const int curIdx = TwoDToOneDIndex(
          result.nodePair.x, result.nodePair.y, numNodesFirst);

      const bool epsilonMatch = epsilonMatchedGPU[TwoDToOneDIndex(
          result.nodePair.x, result.nodePair.y, numNodesFirst)];

      // Does this node pair match?
      if (result.checkArcPair &&
          (graphDP1GPU.olabels[firstArcIdx] == graphDP2GPU.ilabels[secondArcIdx])) {
        int2 dstNodePair = make_int2(
            graphDP1GPU.dstNodes[firstArcIdx], graphDP2GPU.dstNodes[secondArcIdx]);

        const int dstIdx = TwoDToOneDIndex(
            dstNodePair.x, dstNodePair.y, numNodesFirst);

        // We track if any two arcs outgoing from this node pair match
        // on epsilon. We record if they do.
        if (graphDP1GPU.olabels[firstArcIdx] != epsilon) {
          generateCombinedGraphArcs(
              dstIdx,
              curIdx,
              make_int2(firstArcIdx, secondArcIdx),
              reachable,
              newNodesOffset,
              gradInfoFirstGPU,
              gradInfoSecondGPU,
              newGraphDPGPU,
              weights,
              graphDP1GPU.ilabels[firstArcIdx],
              graphDP2GPU.olabels[secondArcIdx],
              weightsFirst[firstArcIdx] + weightsSecond[secondArcIdx]);
        }
      }

      // The epsilon matches
      if (result.checkEpsilonArcPair.x &&
          (!epsilonMatch || graphDP2GPU.accept[result.nodePair.y] ||
           !graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP1GPU.olabels[firstArcIdx] == epsilon)) {
        // When arc from first node has epsilon label then we consider
        // second node
        int2 dstNodePair = make_int2(
            graphDP1GPU.dstNodes[firstArcIdx], result.nodePair.y);
        const int dstIdx = TwoDToOneDIndex(
            dstNodePair.x, dstNodePair.y, numNodesFirst);

        generateCombinedGraphArcs(
            dstIdx,
            curIdx,
            make_int2(firstArcIdx, -1),
            reachable,
            newNodesOffset,
            gradInfoFirstGPU,
            gradInfoSecondGPU,
            newGraphDPGPU,
            weights,
            graphDP1GPU.ilabels[firstArcIdx],
            epsilon,
            weightsFirst[firstArcIdx]);
      }

      // The epsilon matches
      if (result.checkEpsilonArcPair.y &&
          (!epsilonMatch || graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP2GPU.ilabels[secondArcIdx] == epsilon)) {
        // When arc from second node has epsilon label then we consider
        // first node
        int2 dstNodePair = make_int2(
            result.nodePair.x, graphDP2GPU.dstNodes[secondArcIdx]);
        const int dstIdx = TwoDToOneDIndex(
            dstNodePair.x, dstNodePair.y, numNodesFirst);

        generateCombinedGraphArcs(
            dstIdx,
            curIdx,
            make_int2(-1, secondArcIdx),
            reachable,
            newNodesOffset,
            gradInfoFirstGPU,
            gradInfoSecondGPU,
            newGraphDPGPU,
            weights,
            epsilon,
            graphDP2GPU.olabels[secondArcIdx],
            weightsSecond[secondArcIdx]);
      }
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
    auto idx = oneDToTwoDIndex(exploreIndices[gTid], g1.numNodes);
    newGraph.start[gTid] = g1.start[idx.x] && g2.start[idx.y];
    newGraph.accept[gTid] = g1.accept[idx.x] && g2.accept[idx.y];
  }
}

__global__
void calculateNumArcsKernel(
  const HDSpan<int> nodeIndices,
  const int* inputInArcsGPU,
  const int* inputOutArcsGPU,
  HDSpan<int> outputInArcsGPU,
  HDSpan<int> outputOutArcsGPU) {

  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < nodeIndices.size()) {
    const int index = nodeIndices[gTid];
    outputInArcsGPU[gTid] = inputInArcsGPU[index];
    outputOutArcsGPU[gTid] = inputOutArcsGPU[index];
  }
}

__global__
void findReachableInitKernel(
    const HDSpan<int> idsFirst,
    const HDSpan<int> idsSecond,
    HDSpan<int> reachable,
    HDSpan<int> toExplore,
    int numNodesFirst) {

  const int fid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sid = blockIdx.y * blockDim.y + threadIdx.y;

  if (fid < idsFirst.size() && sid < idsSecond.size()) {
    auto idx = TwoDToOneDIndex(idsFirst[fid], idsSecond[sid], numNodesFirst);
    toExplore[idx] = true;
    reachable[idx] = true;
  }
}

void findReachableInit(
    const GraphData& g1,
    const GraphData& g2,
    HDSpan<int> reachable,
    HDSpan<int> toExplore) {
  int NT = 16; 
  auto blocks = dim3(
      div_up(g1.acceptIds.size(), NT), div_up(g2.acceptIds.size(), NT));
  auto threads = dim3(NT, NT);
  findReachableInitKernel<<<blocks, threads>>>(g1.acceptIds, g2.acceptIds,
    reachable, toExplore, g1.numNodes);
}

__global__
void secondPassInitKernel(
    const HDSpan<int> idsFirst,
    const HDSpan<int> idsSecond,
    const HDSpan<int> reachable,
    HDSpan<int> toExplore,
    HDSpan<int> newNodes,
    int numNodesFirst) {
  const int fid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sid = blockIdx.y * blockDim.y + threadIdx.y;

  if (fid < idsFirst.size() && sid < idsSecond.size()) {
    auto idx = TwoDToOneDIndex(idsFirst[fid], idsSecond[sid], numNodesFirst);
    if (reachable[idx]) {
      toExplore[idx] = true;
      newNodes[idx] = true;
    }
  }
}

void secondPassInit(
    const GraphData& g1,
    const GraphData& g2,
    const HDSpan<int> reachable,
    HDSpan<int> toExplore,
    HDSpan<int> newNodes) {
  int NT = 16; 
  auto blocks = dim3(
      div_up(g1.startIds.size(), NT), div_up(g2.startIds.size(), NT));
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

  HDSpan<float> grad(g.numArcs(), 0, true);
  const int NT = 128;
  const int gridSize = div_up(deltas.numArcs(), NT);
  gradKernel<<<gridSize, NT, 0, 0>>>(
      arcIds, deltas.weights(), grad.data(), deltas.numArcs());
  g.addGrad(grad.data());
  grad.clear();
}

__global__
void  boolToIndicesKernel(
    HDSpan<int> ids, const int* counts, const HDSpan<int> vals, size_t size) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < size && vals[gTid]) {
    ids[counts[gTid]] = gTid;
  }
}

auto boolToIndices(const HDSpan<int> vals) {
  int* counts;
  int numTrue;
  std::tie(counts, numTrue) = prefixSumScan(vals.data(), vals.size());

  const int NT = 128;
  const int gridSize = div_up(vals.size(), NT);

  HDSpan<int> ids{true, vals.device()};
  ids.resize(numTrue);
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
  
  const int numAllPairNodes = first.numNodes() * second.numNodes();
  const int numNodesFirst = first.numNodes();

  // Fixed number of CUDA threads and stream for all kernels
  const int NT = 128;

  //////////////////////////////////////////////////////////////////////////
  // Step 1: Data parallel findReachable
  //////////////////////////////////////////////////////////////////////////
  HDSpan<int> reachable(numAllPairNodes, 0, true);
  HDSpan<int> toExplore(numAllPairNodes, 0, true);

  int* epsilonMatchedGPU;
  CUDA_CHECK(cudaMalloc((void **)(&epsilonMatchedGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)epsilonMatchedGPU, false, sizeof(int) * numAllPairNodes));

  findReachableInit(g1, g2, reachable, toExplore); 

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrue(toExplore)) {

    // Convert bits set in toExplore to indices 
    auto exploreIndices = boolToIndices(toExplore);

    int* arcCrossProductIndexGPU = calculateArcCrossProductOffsetGPU(
        exploreIndices, g1, g2, true);

    int* arcCrossProductOffsetGPU;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, totalArcs) =
      prefixSumScan(arcCrossProductIndexGPU, exploreIndices.size());

    CUDA_CHECK(cudaFree(arcCrossProductIndexGPU));

    // Reset so pristine state for next frontier to explore
    setZero(toExplore);

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      findReachableKernel<<<gridSize, NT, 0, 0>>>(
          g1, g2, arcCrossProductOffsetGPU, exploreIndices,
          totalArcs, toExplore, reachable, epsilonMatchedGPU);
    }

    exploreIndices.clear();
    CUDA_CHECK(cudaFree(arcCrossProductOffsetGPU));
  } // end while for findReachable

  //////////////////////////////////////////////////////////////////////////
  // Step 2: Compute a) valid nodes in combined graph
  //                 b) Number of in and out arcs in combined graph
  // This information is used to generate offsets for nodes and arcs
  // in the combined graph
  //////////////////////////////////////////////////////////////////////////

  HDSpan<int> newNodes(numAllPairNodes, 0, true);
  int* numOutArcsGPU;
  int* numInArcsGPU;

  CUDA_CHECK(cudaMalloc((void **)(&numOutArcsGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMalloc((void **)(&numInArcsGPU), sizeof(int) * numAllPairNodes));

  CUDA_CHECK(cudaMemset((void*)numOutArcsGPU, 0, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)numInArcsGPU, 0, sizeof(int) * numAllPairNodes));

  setZero(toExplore);

  secondPassInit(g1, g2, reachable, toExplore, newNodes);

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrue(toExplore)) {

    // Convert bits set in toExplore to node pairs
    auto exploreIndices =  boolToIndices(toExplore);

    int* arcCrossProductIndexGPU = calculateArcCrossProductOffsetGPU(
        exploreIndices, g1, g2, false);

    int* arcCrossProductOffsetGPU;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, totalArcs) =
      prefixSumScan(arcCrossProductIndexGPU, exploreIndices.size());

    CUDA_CHECK(cudaFree(arcCrossProductIndexGPU));

    // Reset so pristine state for next frontier to explore
    setZero(toExplore);

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      computeValidNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(g1, g2,
        arcCrossProductOffsetGPU, exploreIndices, reachable, epsilonMatchedGPU,
        totalArcs, toExplore, newNodes.data(), numInArcsGPU, numOutArcsGPU);
    }

    exploreIndices.clear();
    CUDA_CHECK(cudaFree(arcCrossProductOffsetGPU));
  }
  toExplore.clear();
  reachable.clear();

  //////////////////////////////////////////////////////////////////////////
  // Step 3: Generate offsets for nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////

  int totalNodes;
  int* newNodesOffsetGPU;
  std::tie(newNodesOffsetGPU, totalNodes) = prefixSumScan(newNodes.data(), numAllPairNodes);

  nData.numNodes = totalNodes;
  nData.start.resize(totalNodes);
  nData.accept.resize(totalNodes);
  nData.inArcOffset.resize(totalNodes + 1);
  nData.outArcOffset.resize(totalNodes + 1);

  // Convert bits to indices
  auto exploreIndices =  boolToIndices(newNodes);

  // Generate offsets for nodes and arcs
  {
    const int NT = 128;
    const int gridSize = div_up(exploreIndices.size(), NT);

    calculateNumArcsKernel<<<gridSize, NT, 0, 0>>>(exploreIndices,
      numInArcsGPU, numOutArcsGPU, nData.inArcOffset, nData.outArcOffset);
  }

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

  {
    float* weights;
    CUDA_CHECK(cudaMalloc((void **)(&weights), sizeof(float) * totalOutArcs));
    nGraph.setWeights(weights);
  }

  nData.inArcOffset.copy(inArcOffsetGPU);
  nData.outArcOffset.copy(outArcOffsetGPU);

  auto gradInfo = std::make_shared<GradInfo>();
  CUDA_CHECK(cudaMalloc((void **)(&gradInfo->first), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&gradInfo->second), sizeof(int) * totalOutArcs));

  //////////////////////////////////////////////////////////////////////////
  // Step 4: Generate nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////

  int* arcCrossProductIndexGPU = calculateArcCrossProductOffsetGPU(
      exploreIndices, g1, g2, false);

  int* arcCrossProductOffsetGPU;
  int totalArcs;

  std::tie(arcCrossProductOffsetGPU, totalArcs) =
      prefixSumScan(arcCrossProductIndexGPU, exploreIndices.size());
  CUDA_CHECK(cudaFree(arcCrossProductIndexGPU));

  if (exploreIndices.size() > 0) {
    setZero(nData.start);
    setZero(nData.accept);

    const int gridSize = div_up(exploreIndices.size(), NT);
    setStartAndAccept<<<gridSize, NT, 0, 0>>>(g1, g2, exploreIndices, nData);
    nData.startIds = boolToIndices(nData.start);
    nData.acceptIds = boolToIndices(nData.accept);
  }

  if (totalArcs > 0) {

    const int gridSize = div_up(totalArcs, NT);

    generateNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(g1, g2, first.weights(), second.weights(),
      arcCrossProductOffsetGPU, exploreIndices, newNodes, epsilonMatchedGPU, totalArcs,
      nData, nGraph.weights(), gradInfo->first, gradInfo->second, newNodesOffsetGPU);
  }

  exploreIndices.clear();
  CUDA_CHECK(cudaFree(arcCrossProductOffsetGPU));

  // Reset incremented offsets to original value
  nData.inArcOffset.copy(inArcOffsetGPU);
  nData.outArcOffset.copy(outArcOffsetGPU);

  CUDA_CHECK(cudaFree(epsilonMatchedGPU));
  newNodes.clear();
  CUDA_CHECK(cudaFree(numOutArcsGPU));
  CUDA_CHECK(cudaFree(numInArcsGPU));
  CUDA_CHECK(cudaFree(newNodesOffsetGPU));
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
