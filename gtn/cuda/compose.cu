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
#include <iostream>
#include <sstream>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "gtn/cuda/cuda.h"
#include "gtn/cuda/functions.h"

namespace gtn {
namespace cuda {
namespace detail {

namespace {

typedef Graph::GraphGPU GraphGPU;

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
inline int2 OneDToTwoDIndexGPU(int n, int n1Extent) {
  assert(n1Extent > 0);
  const int n2 = n / n1Extent;
  const int n1 = n % n1Extent;
  return make_int2(n1, n2);
}


bool checkAnyTrueGPU(const int* flags, int numFlags) {
  thrust::device_ptr<const int> tPtr(flags);
  const int sum = thrust::reduce(tPtr, tPtr + numFlags, int(0));

  return (sum > 0);
}

std::tuple<int*, size_t, int> prefixSumScanGPU(const int* input, size_t numElts, bool appendSum) {
  const size_t scanNumElts = appendSum ? numElts + 1 : numElts;

  int *output;
  CUDA_CHECK(cudaMalloc((void **)(&(output)), sizeof(int) * scanNumElts));
  CUDA_CHECK(cudaMemcpy((void *)(output), (void *)(input), sizeof(int) * numElts, cudaMemcpyDeviceToDevice));

  int sum = 0;
  if (numElts > 0) {
    thrust::device_ptr<int> tPtr(output);
    thrust::exclusive_scan(tPtr, tPtr + numElts, tPtr);

    int lastElementInput;
    int lastElementOutput;
    CUDA_CHECK(cudaMemcpy((void *)(&lastElementInput), (void *)(&(input[numElts-1])), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy((void *)(&lastElementOutput), (void *)(&(output[numElts-1])), sizeof(int), cudaMemcpyDeviceToHost));
    sum = lastElementInput + lastElementOutput;
  }

  if (appendSum) {
    assert(scanNumElts > 0);
    CUDA_CHECK(cudaMemcpy((void *)(&(output[scanNumElts-1])), (void *)(&sum), sizeof(int), cudaMemcpyHostToDevice));
  }

  return std::make_tuple(output, scanNumElts, sum);
}

// Map thread id to corresponding node and arc pair
// Also map thread id to two flags checkEpsilonArcPair.first,
// checkEpsilonArcPair.second When checkEpsilonArcPair.first is set,
// corresponding tid will check for arcs with epsilon arcs in the node from
// first graph Same logic happens for checkEpsilonArcPair.second Search to find
// which node pair this tid will fall into Linear search for now
// (arcCrossProductOffset is sorted by definition)
__device__
nodeAndArcPairGPU computeNodeAndArcPair(
    int tid,
    size_t numArcCrossProductOffset,
    const int* arcCrossProductOffset,
    const int* toExploreNumArcsFirst,
    const int* toExploreNumArcsSecond,
    const int* toExploreNodePairFirst,
    const int* toExploreNodePairSecond) {

  nodeAndArcPairGPU result;
  result.checkArcPair = false;
  result.checkEpsilonArcPair = make_int2(false, false);
  result.isValid = false;

  int localIdx, numArcs;
  size_t intervalIdx;

  // There should be at least two values to form a range
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
        result.nodePair = make_int2(
            toExploreNodePairFirst[intervalIdx], toExploreNodePairSecond[intervalIdx]);

        // The range of idx is from
        // [0, toExploreNumArcsFirst[intervalIdx] * toExploreNumArcsSecond[intervalIdx])
        localIdx = tid - lVal;
        numArcs = rVal - lVal;

        break;
      }
    }
  }

  if (result.isValid == true) {

    assert(localIdx >= 0);
    assert(localIdx < numArcs);
    assert(numArcs > 0);

    const int arcProd =
        toExploreNumArcsFirst[intervalIdx] * toExploreNumArcsSecond[intervalIdx];

    if (numArcs == arcProd) {
      result.checkArcPair = true;

      // We map the tids to 2D grid where the
      // x-axis is toExploreNumArcsFirst[i] (row)
      // y-axis is toExploreNumArcsSecond[i] (column)
      assert(toExploreNumArcsFirst[intervalIdx] > 0);
      result.arcPair = make_int2(
        localIdx % toExploreNumArcsFirst[intervalIdx],
        localIdx / toExploreNumArcsFirst[intervalIdx]);

      // Pick the tids from the first row since we need only one
      // tid per arc of the node from the first graph to check for
      // epsilon
      if (localIdx < toExploreNumArcsFirst[intervalIdx]) {
        result.checkEpsilonArcPair.x = true;
      }

      // Pick the tids from the first column since we need only one
      // tid per arc of the node from the first graph to check for
      // epsilon
      if ((localIdx % toExploreNumArcsFirst[intervalIdx]) == 0) {
        result.checkEpsilonArcPair.y = true;
      }
    } else if ((arcProd == 0) && (numArcs == toExploreNumArcsFirst[intervalIdx])) {
      // TODO: Likely not the brightest idea to use -1 as sentinel
      result.arcPair = make_int2(localIdx, -1);
      result.checkEpsilonArcPair.x = true;
    } else if ((arcProd == 0) && (numArcs == toExploreNumArcsSecond[intervalIdx])) {
      // TODO: Likely not the brightest idea to use -1 as sentinel
      result.arcPair = make_int2(-1, localIdx);
      result.checkEpsilonArcPair.y = true;
    }
  }

  return result;
}

__global__
void calculateArcCrossProductOffsetKernel(
      const GraphGPU graphDP1GPU,
      const GraphGPU graphDP2GPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      int* toExploreNumArcsFirstGPU,
      int* toExploreNumArcsSecondGPU,
      int* arcCrossProductOffsetGPU,
      size_t numToExploreNodePair,
      bool inOrOutArc) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numToExploreNodePair) {
    int node = toExploreNodePairFirstGPU[gTid];
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph1 = ((node + 1) == graphDP1GPU.numNodes)
        ? graphDP1GPU.numArcs
        : graphDP1GPU.inArcOffset[node + 1];
    const int outArcOffsetGraph1 = ((node + 1) == graphDP1GPU.numNodes)
        ? graphDP1GPU.numArcs
        : graphDP1GPU.outArcOffset[node + 1];

    const int numArcsFirst = inOrOutArc
        ? inArcOffsetGraph1 - graphDP1GPU.inArcOffset[node]
        : outArcOffsetGraph1 - graphDP1GPU.outArcOffset[node];

    node = toExploreNodePairSecondGPU[gTid];
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph2 = ((node + 1) == graphDP2GPU.numNodes)
        ? graphDP2GPU.numArcs
        : graphDP2GPU.inArcOffset[node + 1];
    const int outArcOffsetGraph2 = ((node + 1) == graphDP2GPU.numNodes)
        ? graphDP2GPU.numArcs
        : graphDP2GPU.outArcOffset[node + 1];

    const int numArcsSecond = inOrOutArc
        ? inArcOffsetGraph2 - graphDP2GPU.inArcOffset[node]
        : outArcOffsetGraph2 - graphDP2GPU.outArcOffset[node];

    toExploreNumArcsFirstGPU[gTid] = numArcsFirst;
    toExploreNumArcsSecondGPU[gTid] = numArcsSecond;

    // Even when numArcsFirst or numArcsSecond is 0 we have to consider
    // the case when the other graph has arcs with epsilon label
    if (numArcsFirst != 0 && numArcsSecond != 0) {
      arcCrossProductOffsetGPU[gTid] = numArcsFirst * numArcsSecond;
    } else if (numArcsFirst != 0 && numArcsSecond == 0) {
      arcCrossProductOffsetGPU[gTid] = numArcsFirst;
    } else if (numArcsFirst == 0 && numArcsSecond != 0) {
      arcCrossProductOffsetGPU[gTid] = numArcsSecond;
    } else {
      arcCrossProductOffsetGPU[gTid] = 0;
    }
  }
}

// Takes a pair of nodes, where each member of pair comes from a different
// graph and calculate a vector of number of arcs in the cross product of
// arcs outgoing from each pair.
// This should be a kernel call
std::tuple<int*, int*, int*>
calculateArcCrossProductOffsetGPU(
    const int* toExploreNodePairFirstGPU,
    const int* toExploreNodePairSecondGPU,
    size_t numToExploreNodePair,
    const GraphGPU graphDP1GPU,
    const GraphGPU graphDP2GPU,
    bool inOrOutArc) {

  int* toExploreNumArcsFirstGPU;
  int* toExploreNumArcsSecondGPU;
  int* arcCrossProductOffsetGPU;
  CUDA_CHECK(cudaMalloc((void **)(&(toExploreNumArcsFirstGPU)), sizeof(int) * numToExploreNodePair));
  CUDA_CHECK(cudaMalloc((void **)(&(toExploreNumArcsSecondGPU)), sizeof(int) * numToExploreNodePair));
  CUDA_CHECK(cudaMalloc((void **)(&(arcCrossProductOffsetGPU)), sizeof(int) * numToExploreNodePair));

  const int NT = 128;
  const int gridSize = div_up(numToExploreNodePair, NT);

  calculateArcCrossProductOffsetKernel<<<gridSize, NT, 0, 0>>>(
      graphDP1GPU, graphDP2GPU, toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
      toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU, arcCrossProductOffsetGPU,
      numToExploreNodePair, inOrOutArc);

  return std::make_tuple(arcCrossProductOffsetGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU);
}

// This function needs to be thread safe since multiple threads can
// can call it and they will overlap on curIdx and dstIdx
__device__
void calculateNumArcsAndNodesToExplore(
    int curIdx,
    int dstIdx,
    const int* reachable,
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

    // printf("cidx %d didx %d\n", curIdx, dstIdx);
    // printf("no %d ni %d\n", numOutArcs[curIdx], numInArcs[dstIdx]);
  }
}

// This function needs to be thread safe since multiple threads can
// can call it
__device__
void generateCombinedGraphNodesAndArcs(
    int dstIdx,
    int curIdx,
    const int2& arcPair,
    const int2& dstNodeStartAndAccept,
    const int* reachable,
    const int* newNodesOffset,
    int* newNodesVisited,
    int* toExplore,
    int* gradInfoFirst,
    int* gradInfoSecond,
    GraphGPU newGraphDP,
    float* weights,
    int ilabel,
    int olabel,
    float weight) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodesVisited
    int oldVal = atomicCAS(&(newNodesVisited[dstIdx]), false, true);
    if (!oldVal) {
      toExplore[dstIdx] = true;
    }

    // Set accept and start nodes
    // I think I only need it for dst nodes and src nodes
    // Note: Multiple threads can have the same dstIdx and write to the same
    //       location and collide. This _should_ be fine since they are going
    //       to write the same value
    newGraphDP.start[newNodesOffset[dstIdx]] = dstNodeStartAndAccept.x;
    newGraphDP.accept[newNodesOffset[dstIdx]] = dstNodeStartAndAccept.y;

    // Both of these increments are atomic
    // int inArcIdx = newGraphDP.inArcOffset[newNodesOffset[dstIdx]]++;
    // int outArcIdx = newGraphDP.outArcOffset[newNodesOffset[curIdx]]++;

    int inArcIdx = atomicAdd(&(newGraphDP.inArcOffset[newNodesOffset[dstIdx]]), 1);
    int outArcIdx = atomicAdd(&(newGraphDP.outArcOffset[newNodesOffset[curIdx]]), 1);

    // printf("dstIdx %d curIdx %d\n", dstIdx, curIdx);
    // printf("inArcIdx %d outArcIdx %d\n", inArcIdx, outArcIdx);

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
void convertToNodePairKernel(
  const int* flagsGPU,
  const int* indicesGPU,
  int* toExploreNodePairFirstGPU,
  int* toExploreNodePairSecondGPU,
  int extent,
  size_t numFlags,
  size_t numValidNodes) {

  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numFlags) {
    if (flagsGPU[gTid] == true) {
      const int index = indicesGPU[gTid];
      assert(index >= 0);
      assert(index < numValidNodes);

      int2 node = OneDToTwoDIndexGPU(gTid, extent);
      toExploreNodePairFirstGPU[index] = node.x;
      toExploreNodePairSecondGPU[index] = node.y;
    }
  }
}

// Convert bool array two pairs for true flags
std::tuple<int*, int*, size_t> convertToNodePairGPU(
    const int* flagsGPU,
    size_t numFlags,
    int extent) {
  int* indicesGPU;
  size_t numIndices;
  size_t numValidNodes;

  std::tie(indicesGPU, numIndices, numValidNodes) = prefixSumScanGPU(flagsGPU, numFlags, false);
  assert(numFlags == numIndices);

  int* toExploreNodePairFirstGPU;
  int* toExploreNodePairSecondGPU;
  CUDA_CHECK(cudaMalloc((void **)(&(toExploreNodePairFirstGPU)), sizeof(int) * numValidNodes));
  CUDA_CHECK(cudaMalloc((void **)(&(toExploreNodePairSecondGPU)), sizeof(int) * numValidNodes));

  const int NT = 128;
  const int gridSize = div_up(numFlags, NT);

  convertToNodePairKernel<<<gridSize, NT, 0, 0>>>(flagsGPU, indicesGPU,
    toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
    extent, numFlags, numValidNodes);

  CUDA_CHECK(cudaFree(indicesGPU));
  return std::make_tuple(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numValidNodes);
}

__device__
int2 getStartAndAccept(
    const GraphGPU graphDP1,
    const GraphGPU graphDP2,
    const int2& dstNodePair) {

  int2 dstNodeStartAndAccept = make_int2(
      graphDP1.start[dstNodePair.x] && graphDP2.start[dstNodePair.y],
      graphDP1.accept[dstNodePair.x] &&
          graphDP2.accept[dstNodePair.y]);

  return dstNodeStartAndAccept;
}


__global__ 
void findReachableKernel(
      const GraphGPU graphDP1GPU,
      const GraphGPU graphDP2GPU,
      const int* arcCrossProductOffsetGPU,
      const int* toExploreNumArcsFirstGPU,
      const int* toExploreNumArcsSecondGPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      int numNodesFirst,
      int totalArcs,
      size_t numArcCrossProductOffset,
      int* toExploreGPU,
      int* reachableGPU,
      int* epsilonMatchedGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    nodeAndArcPairGPU result = 
      computeNodeAndArcPair(
        gTid, numArcCrossProductOffset, arcCrossProductOffsetGPU,
        toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU);

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
        int oldVal = atomicCAS(&(reachableGPU[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
	// printf("r %d t %d \n", reachableGPU[idx], toExploreGPU[idx]);
      }

      // Only valid for arcs incoming to node from first graph
      if (result.checkEpsilonArcPair.x &&
          (graphDP1GPU.olabels[firstArcIdx] == epsilon)) {
        const int idx = TwoDToOneDIndex(
            graphDP1GPU.srcNodes[firstArcIdx], result.nodePair.y, numNodesFirst);
        int oldVal = atomicCAS(&(reachableGPU[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
      }

      // Only valid for arcs incoming to node from second graph
      if (result.checkEpsilonArcPair.y &&
          (graphDP2GPU.ilabels[secondArcIdx] == epsilon)) {
        const int idx = TwoDToOneDIndex(
            result.nodePair.x, graphDP2GPU.srcNodes[secondArcIdx], numNodesFirst);
        int oldVal = atomicCAS(&(reachableGPU[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
      }
    }
  }
}

__global__ 
void computeValidNodeAndArcKernel(
      const GraphGPU graphDP1GPU,
      const GraphGPU graphDP2GPU,
      const int* arcCrossProductOffsetGPU,
      const int* toExploreNumArcsFirstGPU,
      const int* toExploreNumArcsSecondGPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      const int* reachableGPU,
      const int* epsilonMatchedGPU,
      int numNodesFirst,
      int totalArcs,
      size_t numArcCrossProductOffset,
      int* toExploreGPU,
      int* newNodesGPU,
      int* numInArcsGPU,
      int* numOutArcsGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    // Map tid to corresponding node and arc pair
    // Search to find which node pair this tid will fall into
    nodeAndArcPairGPU result = 
      computeNodeAndArcPair(
        gTid, numArcCrossProductOffset, arcCrossProductOffsetGPU,
        toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU);

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
              reachableGPU,
              newNodesGPU,
              toExploreGPU,
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
            reachableGPU,
            newNodesGPU,
            toExploreGPU,
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
            reachableGPU,
            newNodesGPU,
            toExploreGPU,
            numOutArcsGPU,
            numInArcsGPU);
      }
    }
  }
}

__global__ 
void generateNodeAndArcKernel(
      const GraphGPU graphDP1GPU,
      const GraphGPU graphDP2GPU,
      const float* weightsFirst,
      const float* weightsSecond,
      const int* arcCrossProductOffsetGPU,
      const int* toExploreNumArcsFirstGPU,
      const int* toExploreNumArcsSecondGPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      const int* reachableGPU,
      const int* epsilonMatchedGPU,
      int numNodesFirst,
      int totalArcs,
      size_t numArcCrossProductOffset,
      GraphGPU newGraphDPGPU,
      float* weights,
      int* toExploreGPU,
      int* gradInfoFirstGPU,
      int* gradInfoSecondGPU,
      int* newNodesOffsetGPU,
      int* newNodesVisitedGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    // Map tid to corresponding node and arc pair
    // Search to find which node pair this tid will fall into
    nodeAndArcPairGPU result = 
      computeNodeAndArcPair(
        gTid, numArcCrossProductOffset, arcCrossProductOffsetGPU,
        toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU);

    if (result.isValid) {
      int outArcOffset = graphDP1GPU.outArcOffset[result.nodePair.x];
      const int firstArcIdx = graphDP1GPU.outArcs[outArcOffset + result.arcPair.x];

      outArcOffset = graphDP2GPU.outArcOffset[result.nodePair.y];
      const int secondArcIdx =
          graphDP2GPU.outArcs[outArcOffset + result.arcPair.y];

      const bool epsilonMatch = epsilonMatchedGPU[TwoDToOneDIndex(
          result.nodePair.x, result.nodePair.y, numNodesFirst)];

      // Does this node pair match?
      if (result.checkArcPair &&
          (graphDP1GPU.olabels[firstArcIdx] == graphDP2GPU.ilabels[secondArcIdx])) {
        int2 dstNodePair = make_int2(
            graphDP1GPU.dstNodes[firstArcIdx], graphDP2GPU.dstNodes[secondArcIdx]);

        const int dstIdx = TwoDToOneDIndex(
            dstNodePair.x, dstNodePair.y, numNodesFirst);
        const int curIdx = TwoDToOneDIndex(
            result.nodePair.x, result.nodePair.y, numNodesFirst);

	// printf("krn2a dstIdx=%d curIdx=%d\n", dstIdx, curIdx);

        const int2 dstNodeStartAccept =
            getStartAndAccept(graphDP1GPU, graphDP2GPU, dstNodePair);

        // We track if any two arcs outgoing from this node pair match
        // on epsilon. We record if they do.
        if (graphDP1GPU.olabels[firstArcIdx] != epsilon) {
          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              make_int2(firstArcIdx, secondArcIdx),
              dstNodeStartAccept,
              reachableGPU,
              newNodesOffsetGPU,
              newNodesVisitedGPU,
              toExploreGPU,
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
        const int curIdx = TwoDToOneDIndex(
            result.nodePair.x, result.nodePair.y, numNodesFirst);

	// printf("krn2b dstIdx=%d curIdx=%d\n", dstIdx, curIdx);

        const int2 dstNodeStartAccept =
            getStartAndAccept(graphDP1GPU, graphDP2GPU, dstNodePair);

        generateCombinedGraphNodesAndArcs(
            dstIdx,
            curIdx,
            make_int2(firstArcIdx, -1),
            dstNodeStartAccept,
            reachableGPU,
            newNodesOffsetGPU,
            newNodesVisitedGPU,
            toExploreGPU,
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
        const int curIdx = TwoDToOneDIndex(
            result.nodePair.x, result.nodePair.y, numNodesFirst);

	// printf("krn2c dstIdx=%d curIdx=%d\n", dstIdx, curIdx);
	
        const int2 dstNodeStartAndAccept =
            getStartAndAccept(graphDP1GPU, graphDP2GPU, dstNodePair);

        generateCombinedGraphNodesAndArcs(
            dstIdx,
            curIdx,
            make_int2(-1, secondArcIdx),
            dstNodeStartAndAccept,
            reachableGPU,
            newNodesOffsetGPU,
            newNodesVisitedGPU,
            toExploreGPU,
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
void calculateNumArcsKernel(
  const int* flagsGPU,
  const int* indicesGPU,
  const int* inputInArcsGPU,
  const int* inputOutArcsGPU,
  int* outputInArcsGPU,
  int* outputOutArcsGPU,
  size_t numFlags,
  size_t numValidNodes) {

  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numFlags) {
    if (flagsGPU[gTid] == true) {
      const int index = indicesGPU[gTid];
      assert(index >= 0);
      assert(index < numValidNodes);

      outputInArcsGPU[index] = inputInArcsGPU[gTid];
      outputOutArcsGPU[index] = inputOutArcsGPU[gTid];
    }
  }
}

__global__
void fourthPassInitKernel(
    const GraphGPU graphDP1GPU,
    const GraphGPU graphDP2GPU,
    const int* reachableGPU,
    const int* newNodesOffsetGPU,
    GraphGPU newGraphDPGPU,
    int* toExploreGPU,
    int* newNodesVisitedGPU,
    int numNodesFirst,
    int numNodes) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numNodes) {
    int2 indices = OneDToTwoDIndexGPU(gTid, numNodesFirst);

    if (graphDP1GPU.start[indices.x] && graphDP2GPU.start[indices.y]) {
      if (reachableGPU[gTid]) {
        toExploreGPU[gTid] = true;
        newNodesVisitedGPU[gTid] = true;
        newGraphDPGPU.start[newNodesOffsetGPU[gTid]] = true;
        newGraphDPGPU.accept[newNodesOffsetGPU[gTid]] =
           graphDP1GPU.accept[indices.x] && graphDP2GPU.accept[indices.y];
      }
    }
  }
}

__global__
void secondPassInitKernel(
    const GraphGPU graphDP1GPU,
    const GraphGPU graphDP2GPU,
    const int* reachableGPU,
    int* toExploreGPU,
    int* newNodesGPU,
    int numNodesFirst,
    int numNodes) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numNodes) {
    int2 indices = OneDToTwoDIndexGPU(gTid, numNodesFirst);

    if (graphDP1GPU.start[indices.x] && graphDP2GPU.start[indices.y]) {
      if (reachableGPU[gTid]) {
        toExploreGPU[gTid] = true;
        newNodesGPU[gTid] = true;
      }
    }
  }
}

__global__
void findReachableInitInitKernel(
    int* acceptFirst,
    int* acceptSecond,
    int* reachableGPU,
    int* toExploreGPU,
    int numNodesFirst,
    int numNodes) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numNodes) {
    int2 indices = OneDToTwoDIndexGPU(gTid, numNodesFirst);

    if (acceptFirst[indices.x] && acceptSecond[indices.y]) {
      toExploreGPU[gTid] = true;
      reachableGPU[gTid] = true;
    }
  }
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
  const int NT = 128;
  const int gridSize = div_up(deltas.numArcs(), NT);

  float* grad;
  CUDA_CHECK(cudaMalloc((void**)(&grad), sizeof(float) * g.numArcs()));
  CUDA_CHECK(cudaMemset(static_cast<void*>(grad), 0, sizeof(float) * g.numArcs()));
  gradKernel<<<gridSize, NT, 0, 0>>>(arcIds, deltas.weights(), grad, deltas.numArcs());
  g.addGrad(grad);
  CUDA_CHECK(cudaFree(grad));
}

} // namespace


Graph compose(const Graph& first, const Graph& second) {
  auto nGraph = Graph(nullptr, {first, second});
  auto& nGraphGPU = nGraph.deviceData();

  auto& g1 = first.deviceData();
  auto& g2 = second.deviceData();
  
  const int numAllPairNodes = first.numNodes() * second.numNodes();
  const int numNodesFirst = first.numNodes();

  // Fixed number of CUDA threads and stream for all kernels
  const int NT = 128;

  //////////////////////////////////////////////////////////////////////////
  // Step 1: Data parallel findReachable
  //////////////////////////////////////////////////////////////////////////
  int* reachableGPU;
  int* epsilonMatchedGPU;
  int* toExploreGPU;

  CUDA_CHECK(cudaMalloc((void **)(&reachableGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMalloc((void **)(&epsilonMatchedGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMalloc((void **)(&toExploreGPU), sizeof(int) * numAllPairNodes));

  CUDA_CHECK(cudaMemset((void*)reachableGPU, false, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)epsilonMatchedGPU, false, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes));
  {
    const int gridSize = div_up(numAllPairNodes, NT);

    findReachableInitInitKernel<<<gridSize, NT, 0, 0>>>(g1.accept, g2.accept,
      reachableGPU, toExploreGPU, numNodesFirst, numAllPairNodes);
  }

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrueGPU(toExploreGPU, numAllPairNodes)) {

    int* toExploreNodePairFirstGPU;
    int* toExploreNodePairSecondGPU;
    size_t numToExploreNodePair;

    // Convert bits set in toExplore to node pairs
    std::tie(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numToExploreNodePair) =
      convertToNodePairGPU(toExploreGPU, numAllPairNodes, numNodesFirst);

    int* arcCrossProductIndexGPU;
    int* toExploreNumArcsFirstGPU;
    int* toExploreNumArcsSecondGPU;

    std::tie(arcCrossProductIndexGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU) =
      calculateArcCrossProductOffsetGPU(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
        numToExploreNodePair, g1, g2, true);

    int* arcCrossProductOffsetGPU;
    size_t numArcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, numArcCrossProductOffset, totalArcs) =
      prefixSumScanGPU(arcCrossProductIndexGPU, numToExploreNodePair, true);
    assert(numArcCrossProductOffset == (numToExploreNodePair + 1));

    CUDA_CHECK(cudaFree(arcCrossProductIndexGPU));

    // Reset so pristine state for next frontier to explore
    CUDA_CHECK(cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes));

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      findReachableKernel<<<gridSize, NT, 0, 0>>>(g1, g2, arcCrossProductOffsetGPU,
		      toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU, toExploreNodePairFirstGPU,
		      toExploreNodePairSecondGPU, numNodesFirst, totalArcs, numArcCrossProductOffset,
		      toExploreGPU, reachableGPU, epsilonMatchedGPU);
    }

    CUDA_CHECK(cudaFree(toExploreNodePairFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNodePairSecondGPU));
    CUDA_CHECK(cudaFree(arcCrossProductOffsetGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsSecondGPU));
  } // end while for findReachable

  //////////////////////////////////////////////////////////////////////////
  // Step 2: Compute a) valid nodes in combined graph
  //                 b) Number of in and out arcs in combined graph
  // This information is used to generate offsets for nodes and arcs
  // in the combined graph
  //////////////////////////////////////////////////////////////////////////

  int* newNodesGPU;
  int* numOutArcsGPU;
  int* numInArcsGPU;

  CUDA_CHECK(cudaMalloc((void **)(&newNodesGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMalloc((void **)(&numOutArcsGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMalloc((void **)(&numInArcsGPU), sizeof(int) * numAllPairNodes));

  CUDA_CHECK(cudaMemset((void*)newNodesGPU, false, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)numOutArcsGPU, 0, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)numInArcsGPU, 0, sizeof(int) * numAllPairNodes));

  CUDA_CHECK(cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes));

  {
    const int gridSize = div_up(numAllPairNodes, NT);

    secondPassInitKernel<<<gridSize, NT, 0, 0>>>(g1, g2, reachableGPU,
      toExploreGPU, newNodesGPU, numNodesFirst, numAllPairNodes);
  }

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrueGPU(toExploreGPU, numAllPairNodes)) {

    int* toExploreNodePairFirstGPU;
    int* toExploreNodePairSecondGPU;
    size_t numToExploreNodePair;

    // Convert bits set in toExplore to node pairs
    std::tie(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numToExploreNodePair) =
      convertToNodePairGPU(toExploreGPU, numAllPairNodes, numNodesFirst);

    int* arcCrossProductIndexGPU;
    int* toExploreNumArcsFirstGPU;
    int* toExploreNumArcsSecondGPU;

    std::tie(arcCrossProductIndexGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU) =
      calculateArcCrossProductOffsetGPU(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
        numToExploreNodePair, g1, g2, false);

    int* arcCrossProductOffsetGPU;
    size_t numArcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, numArcCrossProductOffset, totalArcs) =
      prefixSumScanGPU(arcCrossProductIndexGPU, numToExploreNodePair, true);
    assert(numArcCrossProductOffset == (numToExploreNodePair + 1));

    CUDA_CHECK(cudaFree(arcCrossProductIndexGPU));

    // Reset so pristine state for next frontier to explore
    CUDA_CHECK(cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes));

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      computeValidNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(g1, g2,
        arcCrossProductOffsetGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, reachableGPU,
        epsilonMatchedGPU, numNodesFirst, totalArcs, numArcCrossProductOffset,
        toExploreGPU, newNodesGPU, numInArcsGPU, numOutArcsGPU);
    }

    CUDA_CHECK(cudaFree(toExploreNodePairFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNodePairSecondGPU));
    CUDA_CHECK(cudaFree(arcCrossProductOffsetGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsSecondGPU));
  }

  //////////////////////////////////////////////////////////////////////////
  // Step 3: Generate offsets for nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////

  int totalNodes;
  int* newNodesOffsetGPU;
  size_t numElements;
  std::tie(newNodesOffsetGPU, numElements, totalNodes) = prefixSumScanGPU(newNodesGPU, numAllPairNodes, false);
  assert(numElements == numAllPairNodes);

  nGraphGPU.numNodes = totalNodes;
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.start)), sizeof(int) * totalNodes));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.accept)), sizeof(int) * totalNodes));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.inArcOffset)), sizeof(int) * (totalNodes + 1)));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.outArcOffset)), sizeof(int) * (totalNodes + 1)));

  // Generate offsets for nodes and arcs
  {
    const int NT = 128;
    const int gridSize = div_up(numAllPairNodes, NT);

    calculateNumArcsKernel<<<gridSize, NT, 0, 0>>>(newNodesGPU, newNodesOffsetGPU,
      numInArcsGPU, numOutArcsGPU, nGraphGPU.inArcOffset, nGraphGPU.outArcOffset,
      numAllPairNodes, totalNodes);
  }

  int totalInArcs;
  int totalOutArcs;

  int* inArcOffsetGPU;
  int* outArcOffsetGPU;

  std::tie(inArcOffsetGPU, numElements, totalInArcs) = prefixSumScanGPU(nGraphGPU.inArcOffset, totalNodes, true);
  assert(numElements == totalNodes + 1);

  std::tie(outArcOffsetGPU, numElements, totalOutArcs) = prefixSumScanGPU(nGraphGPU.outArcOffset, totalNodes, true);
  assert(numElements == totalNodes + 1);

  assert(totalInArcs == totalOutArcs);
  nGraphGPU.numArcs = totalOutArcs;
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.inArcs)), sizeof(int) * totalInArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.outArcs)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.ilabels)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.olabels)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.srcNodes)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(nGraphGPU.dstNodes)), sizeof(int) * totalOutArcs));

  {
    float* weights;
    CUDA_CHECK(cudaMalloc((void **)(&weights), sizeof(float) * totalOutArcs));
    nGraph.setWeights(weights);
  }

  CUDA_CHECK(cudaMemcpy((void *)(nGraphGPU.inArcOffset), (void *)(inArcOffsetGPU), sizeof(int) * (totalNodes + 1), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(nGraphGPU.outArcOffset), (void *)(outArcOffsetGPU), sizeof(int) * (totalNodes + 1), cudaMemcpyDeviceToDevice));

  auto gradInfo = std::make_shared<GradInfo>();
  CUDA_CHECK(cudaMalloc((void **)(&gradInfo->first), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&gradInfo->second), sizeof(int) * totalOutArcs));

  //////////////////////////////////////////////////////////////////////////
  // Step 4: Generate nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////

  int* newNodesVisitedGPU;
  CUDA_CHECK(cudaMalloc((void **)(&newNodesVisitedGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)newNodesVisitedGPU, false, sizeof(int) * numAllPairNodes));

  // Reset so pristine state for next frontier to explore
  CUDA_CHECK(cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void *)(nGraphGPU.start), false, sizeof(int) * totalNodes));
  CUDA_CHECK(cudaMemset((void *)(nGraphGPU.accept), false, sizeof(int) * totalNodes));

  {
    const int gridSize = div_up(numAllPairNodes, NT);

    fourthPassInitKernel<<<gridSize, NT, 0, 0>>>(g1, g2, reachableGPU,
      newNodesOffsetGPU, nGraphGPU, toExploreGPU, newNodesVisitedGPU,
      numNodesFirst, numAllPairNodes);
  }

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrueGPU(toExploreGPU, numAllPairNodes)) {

    int* toExploreNodePairFirstGPU;
    int* toExploreNodePairSecondGPU;
    size_t numToExploreNodePair;

    // Convert bits set in toExplore to node pairs
    std::tie(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numToExploreNodePair) =
      convertToNodePairGPU(toExploreGPU, numAllPairNodes, numNodesFirst);

    int* arcCrossProductIndexGPU;
    int* toExploreNumArcsFirstGPU;
    int* toExploreNumArcsSecondGPU;

    std::tie(arcCrossProductIndexGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU) =
      calculateArcCrossProductOffsetGPU(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
        numToExploreNodePair, g1, g2, false);

    int* arcCrossProductOffsetGPU;
    size_t numArcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, numArcCrossProductOffset, totalArcs) =
      prefixSumScanGPU(arcCrossProductIndexGPU, numToExploreNodePair, true);
    assert(numArcCrossProductOffset == (numToExploreNodePair + 1));

    CUDA_CHECK(cudaFree(arcCrossProductIndexGPU));

    // Reset so pristine state for next frontier to explore
    CUDA_CHECK(cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes));

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      generateNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(g1, g2, first.weights(), second.weights(),
        arcCrossProductOffsetGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, reachableGPU,
        epsilonMatchedGPU, numNodesFirst, totalArcs, numArcCrossProductOffset,
        nGraphGPU, nGraph.weights(), toExploreGPU, gradInfo->first, gradInfo->second,
        newNodesOffsetGPU, newNodesVisitedGPU);
    }

    CUDA_CHECK(cudaFree(toExploreNodePairFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNodePairSecondGPU));
    CUDA_CHECK(cudaFree(arcCrossProductOffsetGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsSecondGPU));
  }

  // Reset incremented offsets to original value
  CUDA_CHECK(cudaMemcpy((void *)(nGraphGPU.inArcOffset), (void *)(inArcOffsetGPU), sizeof(int) * (nGraphGPU.numNodes + 1), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(nGraphGPU.outArcOffset), (void *)(outArcOffsetGPU), sizeof(int) * (nGraphGPU.numNodes+ 1), cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaFree(reachableGPU));
  CUDA_CHECK(cudaFree(epsilonMatchedGPU));
  CUDA_CHECK(cudaFree(toExploreGPU));
  CUDA_CHECK(cudaFree(newNodesGPU));
  CUDA_CHECK(cudaFree(numOutArcsGPU));
  CUDA_CHECK(cudaFree(numInArcsGPU));
  CUDA_CHECK(cudaFree(newNodesOffsetGPU));
  CUDA_CHECK(cudaFree(inArcOffsetGPU));
  CUDA_CHECK(cudaFree(outArcOffsetGPU));
  CUDA_CHECK(cudaFree(newNodesVisitedGPU));

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
