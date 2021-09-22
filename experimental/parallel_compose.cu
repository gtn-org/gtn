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

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "parallel_compose.h"
#include "prefix_scan.h"

/// usage: `CUDA_CHECK(cudaError_t err[, const char* prefix])`
#define CUDA_CHECK(err) \
  cudaCheck(err, __FILE__, __LINE__)

namespace gtn {
namespace detail {
namespace dataparallel {

namespace {

void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream ess;
    ess << '[' << file << ':' << line
        << "] CUDA error: " << cudaGetErrorString(err);
    throw std::runtime_error(ess.str());
  }
}

struct GraphDataParallelGPU {
  size_t numNodes;
  size_t numArcs;

  // True if a node is accept or start, false otherwise
  int* accept;
  int* start;

  // One value per node - i-th value corresponds to i-th node
  // Last element is the total number of arcs, so that
  // each element and its neighbor forms a range
  int* inArcOffset;
  int* outArcOffset;

  // One value per arc
  int* inArcs;
  int* outArcs;

  // One value per arc
  // i-th value corresponds to i-th arc
  int* ilabels;
  int* olabels;
  int* srcNodes;
  int* dstNodes;
  float* weights;
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

  // Linear search
  /*
  {
    nodeAndArcPairGPU result2;
    result2.checkArcPair = false;
    result2.checkEpsilonArcPair = make_int2(false, false);
    result2.isValid = false;
    int localIdx2, numArcs2;
    size_t intervalIdx2;

    for (size_t i = 0; i < numIntervals; ++i) {
      const int lVal = arcCrossProductOffset[i];
      const int rVal = arcCrossProductOffset[i + 1];

      if ((lVal <= tid) && (tid < rVal)) {
        intervalIdx2 = i;
        result2.isValid = true;
        result2.nodePair = make_int2(
            toExploreNodePairFirst[intervalIdx2], toExploreNodePairSecond[intervalIdx2]);

        // The range of idx is from
        // [0, toExploreNumArcsFirst[intervalIdx2] * toExploreNumArcsSecond[intervalIdx2])
        localIdx2 = tid - lVal;
        numArcs2 = rVal - lVal;

        break;
      }
    }

    assert(result.isValid == result2.isValid);
    if (result2.isValid) {
      assert(result.checkArcPair == result2.checkArcPair);
      assert(result.checkEpsilonArcPair.x == result2.checkEpsilonArcPair.x);
      assert(result.checkEpsilonArcPair.y == result2.checkEpsilonArcPair.y);
      assert(localIdx == localIdx2);
      assert(numArcs == numArcs2);
      assert(intervalIdx == intervalIdx2);
    }
  }*/

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
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
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
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
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
    /*
    int oldVal = newNodes[dstIdx];
    if (!newNodes[dstIdx]) {
      newNodes[dstIdx] = true;
    }*/
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
    GraphDataParallelGPU& newGraphDP,
    int ilabel,
    int olabel,
    float weight) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodesVisited
    /*
    int oldVal = newNodesVisited[dstIdx];
    if (!newNodesVisited[dstIdx]) {
      newNodesVisited[dstIdx] = true;
    }*/

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
    newGraphDP.weights[outArcIdx] = weight;

    // printf("ilabels %d olabels %d srcNodes %d dstNodes %d weights %f\n",
           // newGraphDP.ilabels[outArcIdx], newGraphDP.olabels[outArcIdx],
	   // newGraphDP.srcNodes[outArcIdx], newGraphDP.dstNodes[outArcIdx],
	   // newGraphDP.weights[outArcIdx]);

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
    const GraphDataParallelGPU& graphDP1,
    const GraphDataParallelGPU& graphDP2,
    const int2& dstNodePair) {

  int2 dstNodeStartAndAccept = make_int2(
      graphDP1.start[dstNodePair.x] && graphDP2.start[dstNodePair.y],
      graphDP1.accept[dstNodePair.x] &&
          graphDP2.accept[dstNodePair.y]);

  return dstNodeStartAndAccept;
}

GraphDataParallelGPU copyToGPU(const GraphDataParallel& graphDP) {
  GraphDataParallelGPU graphDPGPU;

  graphDPGPU.numNodes = graphDP.inArcOffset.size();
  graphDPGPU.numArcs = graphDP.inArcs.size();

  assert(graphDP.accept.size() == graphDPGPU.numNodes);
  assert(graphDP.start.size() == graphDPGPU.numNodes);
  assert(graphDP.inArcOffset.size() == graphDPGPU.numNodes);
  assert(graphDP.outArcOffset.size() == graphDPGPU.numNodes);

  assert(graphDP.inArcs.size() == graphDPGPU.numArcs);
  assert(graphDP.outArcs.size() == graphDPGPU.numArcs);
  assert(graphDP.ilabels.size() == graphDPGPU.numArcs);
  assert(graphDP.olabels.size() == graphDPGPU.numArcs);
  assert(graphDP.srcNodes.size() == graphDPGPU.numArcs);
  assert(graphDP.dstNodes.size() == graphDPGPU.numArcs);
  assert(graphDP.weights.size() == graphDPGPU.numArcs);

  // Allocate memory
  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.accept)), sizeof(int) * graphDPGPU.numNodes));

  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.start)), sizeof(int) * graphDPGPU.numNodes));

  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.inArcOffset)), sizeof(int) * graphDPGPU.numNodes));
  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.outArcOffset)), sizeof(int) * graphDPGPU.numNodes));

  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.inArcs)), sizeof(int) * graphDPGPU.numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.outArcs)), sizeof(int) * graphDPGPU.numArcs));

  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.ilabels)), sizeof(int) * graphDPGPU.numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.olabels)), sizeof(int) * graphDPGPU.numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.srcNodes)), sizeof(int) * graphDPGPU.numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.dstNodes)), sizeof(int) * graphDPGPU.numArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(graphDPGPU.weights)), sizeof(float) * graphDPGPU.numArcs));

  // Copy
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.accept), (void *)(graphDP.accept.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.start), (void *)(graphDP.start.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.inArcOffset), (void *)(graphDP.inArcOffset.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.outArcOffset), (void *)(graphDP.outArcOffset.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.inArcs), (void *)(graphDP.inArcs.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.outArcs), (void *)(graphDP.outArcs.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.ilabels), (void *)(graphDP.ilabels.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.olabels), (void *)(graphDP.olabels.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.srcNodes), (void *)(graphDP.srcNodes.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.dstNodes), (void *)(graphDP.dstNodes.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(graphDPGPU.weights), (void *)(graphDP.weights.data()), sizeof(float) * graphDPGPU.numArcs, cudaMemcpyHostToDevice));

  return graphDPGPU;
}

__global__ 
void findReachableKernel(
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
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
        /*
        int oldVal = reachableGPU[idx];
        if (!reachableGPU[idx]) {
          reachableGPU[idx] = true;
        }*/
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
        /*
        int oldVal = reachableGPU[idx];
        if (!reachableGPU[idx]) {
          reachableGPU[idx] = true;
        }*/
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
        /*
        int oldVal = reachableGPU[idx];
        if (!reachableGPU[idx]) {
          reachableGPU[idx] = true;
        }*/
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
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
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
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
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
      GraphDataParallelGPU newGraphDPGPU,
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
              graphDP1GPU.ilabels[firstArcIdx],
              graphDP2GPU.olabels[secondArcIdx],
              graphDP1GPU.weights[firstArcIdx] + graphDP2GPU.weights[secondArcIdx]);
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
            graphDP1GPU.ilabels[firstArcIdx],
            epsilon,
            graphDP1GPU.weights[firstArcIdx]);
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
            epsilon,
            graphDP2GPU.olabels[secondArcIdx],
            graphDP2GPU.weights[secondArcIdx]);
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
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
    const int* reachableGPU,
    const int* newNodesOffsetGPU,
    GraphDataParallelGPU newGraphDPGPU,
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
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
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
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
    int* reachableGPU,
    int* toExploreGPU,
    int numNodesFirst,
    int numNodes) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numNodes) {
    int2 indices = OneDToTwoDIndexGPU(gTid, numNodesFirst);

    if (graphDP1GPU.accept[indices.x] && graphDP2GPU.accept[indices.y]) {
      toExploreGPU[gTid] = true;
      reachableGPU[gTid] = true;
    }
  }
}

} // namespace

Graph compose(const Graph& first, const Graph& second) {
  GraphDataParallel graphDP1, graphDP2;

  // Convert from AOS to SOA
  graphDP1 = convertToDataParallel(first);
  graphDP2 = convertToDataParallel(second);

  // Copy to GPU
  GraphDataParallelGPU graphDP1GPU, graphDP2GPU;
  graphDP1GPU = copyToGPU(graphDP1);
  graphDP2GPU = copyToGPU(graphDP2);
  
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

    findReachableInitInitKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU,
      reachableGPU, toExploreGPU, numNodesFirst, numAllPairNodes);
  }

  // std::cout << "num all pair nodes " << numAllPairNodes << std::endl;

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
        numToExploreNodePair, graphDP1GPU, graphDP2GPU, true);

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

      findReachableKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU, arcCrossProductOffsetGPU,
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

    secondPassInitKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU, reachableGPU,
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
        numToExploreNodePair, graphDP1GPU, graphDP2GPU, false);

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

      computeValidNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU,
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
  GraphDataParallelGPU newGraphDPGPU;

  int totalNodes;
  int* newNodesOffsetGPU;
  size_t numElements;
  std::tie(newNodesOffsetGPU, numElements, totalNodes) = prefixSumScanGPU(newNodesGPU, numAllPairNodes, false);
  assert(numElements == numAllPairNodes);

  newGraphDPGPU.numNodes = totalNodes;
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.start)), sizeof(int) * totalNodes));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.accept)), sizeof(int) * totalNodes));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.inArcOffset)), sizeof(int) * totalNodes));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.outArcOffset)), sizeof(int) * totalNodes));

  // Generate offsets for nodes and arcs
  {
    const int NT = 128;
    const int gridSize = div_up(numAllPairNodes, NT);

    calculateNumArcsKernel<<<gridSize, NT, 0, 0>>>(newNodesGPU, newNodesOffsetGPU,
      numInArcsGPU, numOutArcsGPU, newGraphDPGPU.inArcOffset, newGraphDPGPU.outArcOffset,
      numAllPairNodes, totalNodes);
  }

  int totalInArcs;
  int totalOutArcs;

  int* inArcOffsetGPU;
  int* outArcOffsetGPU;

  std::tie(inArcOffsetGPU, numElements, totalInArcs) = prefixSumScanGPU(newGraphDPGPU.inArcOffset, totalNodes, false);
  assert(numElements == totalNodes);

  std::tie(outArcOffsetGPU, numElements, totalOutArcs) = prefixSumScanGPU(newGraphDPGPU.outArcOffset, totalNodes, false);
  assert(numElements == totalNodes);

  assert(totalInArcs == totalOutArcs);
  newGraphDPGPU.numArcs = totalOutArcs;
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.inArcs)), sizeof(int) * totalInArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.outArcs)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.ilabels)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.olabels)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.srcNodes)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.dstNodes)), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&(newGraphDPGPU.weights)), sizeof(float) * totalOutArcs));

  CUDA_CHECK(cudaMemcpy((void *)(newGraphDPGPU.inArcOffset), (void *)(inArcOffsetGPU), sizeof(int) * totalNodes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDPGPU.outArcOffset), (void *)(outArcOffsetGPU), sizeof(int) * totalNodes, cudaMemcpyDeviceToDevice));

  // std::cout << "totalInArcs " << totalInArcs << " totalOutArcs " << totalOutArcs << std::endl;

  // SOA for gradInfo
  std::pair<std::vector<int>, std::vector<int>> gradInfo;
  gradInfo.first.resize(totalOutArcs);
  gradInfo.second.resize(totalOutArcs);

  int *gradInfoFirstGPU;
  int *gradInfoSecondGPU;

  CUDA_CHECK(cudaMalloc((void **)(&gradInfoFirstGPU), sizeof(int) * totalOutArcs));
  CUDA_CHECK(cudaMalloc((void **)(&gradInfoSecondGPU), sizeof(int) * totalOutArcs));

  //////////////////////////////////////////////////////////////////////////
  // Step 4: Generate nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////

  int* newNodesVisitedGPU;
  CUDA_CHECK(cudaMalloc((void **)(&newNodesVisitedGPU), sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void*)newNodesVisitedGPU, false, sizeof(int) * numAllPairNodes));

  // Reset so pristine state for next frontier to explore
  CUDA_CHECK(cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes));
  CUDA_CHECK(cudaMemset((void *)(newGraphDPGPU.start), false, sizeof(int) * totalNodes));
  CUDA_CHECK(cudaMemset((void *)(newGraphDPGPU.accept), false, sizeof(int) * totalNodes));

  {
    const int gridSize = div_up(numAllPairNodes, NT);

    fourthPassInitKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU, reachableGPU,
      newNodesOffsetGPU, newGraphDPGPU, toExploreGPU, newNodesVisitedGPU,
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
        numToExploreNodePair, graphDP1GPU, graphDP2GPU, false);

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

      generateNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU,
        arcCrossProductOffsetGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, reachableGPU,
        epsilonMatchedGPU, numNodesFirst, totalArcs, numArcCrossProductOffset,
        newGraphDPGPU, toExploreGPU, gradInfoFirstGPU, gradInfoSecondGPU,
        newNodesOffsetGPU, newNodesVisitedGPU);
    }

    CUDA_CHECK(cudaFree(toExploreNodePairFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNodePairSecondGPU));
    CUDA_CHECK(cudaFree(arcCrossProductOffsetGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsFirstGPU));
    CUDA_CHECK(cudaFree(toExploreNumArcsSecondGPU));
  }

  // Reset incremented offsets to original value
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDPGPU.inArcOffset), (void *)(inArcOffsetGPU), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDPGPU.outArcOffset), (void *)(outArcOffsetGPU), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToDevice));

  // Copy graph on GPU to CPU
  GraphDataParallel newGraphDP;
  newGraphDP.start.resize(totalNodes);
  newGraphDP.accept.resize(totalNodes);
  newGraphDP.inArcOffset.resize(totalNodes);
  newGraphDP.outArcOffset.resize(totalNodes);
  newGraphDP.inArcs.resize(totalInArcs);
  newGraphDP.outArcs.resize(totalOutArcs);
  newGraphDP.ilabels.resize(totalOutArcs);
  newGraphDP.olabels.resize(totalOutArcs);
  newGraphDP.srcNodes.resize(totalOutArcs);
  newGraphDP.dstNodes.resize(totalOutArcs);
  newGraphDP.weights.resize(totalOutArcs);
 
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.accept.data()), (void *)(newGraphDPGPU.accept), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.start.data()), (void *)(newGraphDPGPU.start), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.inArcOffset.data()), (void *)(newGraphDPGPU.inArcOffset), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.outArcOffset.data()), (void *)(newGraphDPGPU.outArcOffset), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.inArcs.data()), (void *)(newGraphDPGPU.inArcs), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.outArcs.data()), (void *)(newGraphDPGPU.outArcs), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.ilabels.data()), (void *)(newGraphDPGPU.ilabels), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.olabels.data()), (void *)(newGraphDPGPU.olabels), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.srcNodes.data()), (void *)(newGraphDPGPU.srcNodes), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.dstNodes.data()), (void *)(newGraphDPGPU.dstNodes), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(newGraphDP.weights.data()), (void *)(newGraphDPGPU.weights), sizeof(float) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost));

  assert(newGraphDPGPU.numArcs == totalOutArcs);
  CUDA_CHECK(cudaMemcpy((void *)(gradInfo.first.data()), (void *)(gradInfoFirstGPU), sizeof(int) * totalOutArcs, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void *)(gradInfo.second.data()), (void *)(gradInfoSecondGPU), sizeof(int) * totalOutArcs, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(reachableGPU));
  CUDA_CHECK(cudaFree(epsilonMatchedGPU));
  CUDA_CHECK(cudaFree(toExploreGPU));
  CUDA_CHECK(cudaFree(newNodesGPU));
  CUDA_CHECK(cudaFree(numOutArcsGPU));
  CUDA_CHECK(cudaFree(numInArcsGPU));
  CUDA_CHECK(cudaFree(newNodesOffsetGPU));
  CUDA_CHECK(cudaFree(inArcOffsetGPU));
  CUDA_CHECK(cudaFree(outArcOffsetGPU));
  CUDA_CHECK(cudaFree(gradInfoFirstGPU));
  CUDA_CHECK(cudaFree(gradInfoSecondGPU));
  CUDA_CHECK(cudaFree(newNodesVisitedGPU));

  CUDA_CHECK(cudaFree(newGraphDPGPU.start));
  CUDA_CHECK(cudaFree(newGraphDPGPU.accept));
  CUDA_CHECK(cudaFree(newGraphDPGPU.inArcOffset));
  CUDA_CHECK(cudaFree(newGraphDPGPU.outArcOffset));
  CUDA_CHECK(cudaFree(newGraphDPGPU.inArcs));
  CUDA_CHECK(cudaFree(newGraphDPGPU.outArcs));
  CUDA_CHECK(cudaFree(newGraphDPGPU.ilabels));
  CUDA_CHECK(cudaFree(newGraphDPGPU.olabels));
  CUDA_CHECK(cudaFree(newGraphDPGPU.srcNodes));
  CUDA_CHECK(cudaFree(newGraphDPGPU.dstNodes));
  CUDA_CHECK(cudaFree(newGraphDPGPU.weights));
  newGraphDPGPU.numNodes = 0;
  newGraphDPGPU.numArcs = 0;

  if (0)
  {
    std::cout << "nodes " << newGraphDP.inArcOffset.size() << std::endl;
    std::cout << "nodes " << newGraphDP.outArcOffset.size() << std::endl;

    std::cout << "start" << std::endl;
    for (auto i : newGraphDP.start) {
      std::cout << i << std::endl;
    }

    std::cout << "accept" << std::endl;
    for (auto i : newGraphDP.accept) {
      std::cout << i << std::endl;
    }

    std::cout << "inArcOffset" << std::endl;
    for (auto i : newGraphDP.inArcOffset) {
      std::cout << i << std::endl;
    }

    std::cout << "outArcOffset" << std::endl;
    for (auto i : newGraphDP.outArcOffset) {
      std::cout << i << std::endl;
    }

    std::cout << "inArcs" << std::endl;
    for (auto i : newGraphDP.inArcs) {
      std::cout << i << std::endl;
    }

    std::cout << "outArcs" << std::endl;
    for (auto i : newGraphDP.outArcs) {
      std::cout << i << std::endl;
    }

    std::cout << "ilabels" << std::endl;
    for (auto i : newGraphDP.ilabels) {
      std::cout << i << std::endl;
    }

    std::cout << "olabels" << std::endl;
    for (auto i : newGraphDP.olabels) {
      std::cout << i << std::endl;
    }

    std::cout << "srcNodes" << std::endl;
    for (auto i : newGraphDP.srcNodes) {
      std::cout << i << std::endl;
    }

    std::cout << "dstNodes" << std::endl;
    for (auto i : newGraphDP.dstNodes) {
      std::cout << i << std::endl;
    }

    std::cout << "weights" << std::endl;
    for (auto i : newGraphDP.weights) {
      std::cout << i << std::endl;
    }
  }
  // Not needed since the CPU data is never incremented
  // Shift offset values back down after adding arcs to newGraphDP
  // The offset values got converted from exclusive prefix sum to inclusive
  // Need to convert them back to exclusive prefix sum  by starting with 0
  // and shifting to right by 1
  // for (int i = newGraphDP.outArcOffset.size() - 1; i >= 0; --i) {
    // newGraphDP.outArcOffset[i] = i == 0 ? 0 : newGraphDP.outArcOffset[i - 1];
    // newGraphDP.inArcOffset[i] = i == 0 ? 0 : newGraphDP.inArcOffset[i - 1];
  // }

  // Convert back and add in autograd metadata
  auto nGraph = convertFromDataParallel(newGraphDP);
  nGraph.setInputs({first, second});

  if (0)
  {
    std::cout << "numNodes " << nGraph.numNodes() << std::endl;

    std::cout << "accept" << std::endl;
    for (auto i : nGraph.accept()) {
      std::cout << i << std::endl;
    }

    std::cout << "start" << std::endl;
    for (auto i : nGraph.start()) {
      std::cout << i << std::endl;
    }

    std::cout << "numIn" << std::endl;
    for (int i = 0; i < nGraph.numNodes(); ++i) {
      std::cout << nGraph.numIn(i) << std::endl;
    }

    std::cout << "numOut" << std::endl;
    for (int i = 0; i < nGraph.numNodes(); ++i) {
      std::cout << nGraph.numOut(i) << std::endl;
    }
  }

  // Convert gradInfo SOA to AOS
  std::vector<std::pair<int, int>> gradInfoAOS;
  for (int i = 0; i < gradInfo.first.size(); ++i) {
    gradInfoAOS.emplace_back(gradInfo.first[i], gradInfo.second[i]);
  }

  // TODO eliminate this copy pasta.
  auto gradFunc = [gradInfo = std::move(gradInfoAOS)](
                      std::vector<Graph>& inputs, Graph deltas) {
    // In this case the arc's parents are always from the
    // first and second input graphs respectively.
    bool calcGrad1 = inputs[0].calcGrad();
    bool calcGrad2 = inputs[1].calcGrad();
    auto grad1 = calcGrad1 ? std::vector<float>(inputs[0].numArcs(), 0.0)
                           : std::vector<float>{};
    auto grad2 = calcGrad2 ? std::vector<float>(inputs[1].numArcs(), 0.0)
                           : std::vector<float>{};
    for (int i = 0; i < gradInfo.size(); i++) {
      auto arcGrad = deltas.weight(i);
      auto& arcs = gradInfo[i];
      if (calcGrad1 && arcs.first >= 0) {
        grad1[arcs.first] += arcGrad;
      }
      if (calcGrad2 && arcs.second >= 0) {
        grad2[arcs.second] += arcGrad;
      }
    }
    inputs[0].addGrad(std::move(grad1));
    inputs[1].addGrad(std::move(grad2));
  };
  nGraph.setGradFunc(std::move(gradFunc));
  return nGraph;
}

} // namespace dataparallel
} // namespace detail
} // namespace gtn
