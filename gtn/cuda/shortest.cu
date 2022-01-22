#include <thrust/device_ptr.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "gtn/creations.h"
#include "gtn/cuda/cuda.h"
#include "gtn/cuda/functions.h"
#include "gtn/hd_span.h"

using namespace gtn::detail;

namespace gtn {
namespace cuda {
namespace detail {

namespace{

typedef Graph::SharedGraph GraphData;

__constant__ constexpr float kInf = std::numeric_limits<float>::infinity();
__constant__ constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr int kNT = 128;

typedef union {
  float2 val; // val.x stores max
  int2 idx; // val.y stores argmax
  unsigned long long int data; // used for atomics
} ArgMaxT;

__device__ unsigned long long int atomicArgMax(
    ArgMaxT* address, float val, int idx) {
  auto laddr = (unsigned long long int*) address;
  ArgMaxT old = *address;
  ArgMaxT curr;
  curr.val.x = val;
  curr.idx.y = idx;
  while (curr.val.x > (*address).val.x) {
    old.data = atomicCAS(laddr, old.data, curr.data);
  }
  return old.data;
}

struct SubMaxAndExp{
  SubMaxAndExp(float maxVal) : maxVal(maxVal) {};
  float maxVal;

  __device__ float operator()(const float& in) {
    return expf(in - maxVal);
};
};

struct GradInfo {

  HDSpan<float> scores;
  HDSpan<ArgMaxT> maxes;

  GradInfo(
      const HDSpan<float>& scores,
      const HDSpan<ArgMaxT>& maxes) :
    scores(scores), maxes(maxes) { };

  ~GradInfo() {
    scores.clear();
    maxes.clear();
  };
};

__host__ __device__
inline int divUp(int x, int y) {
  return (x + y - 1) / y;
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
void initExploreKernel(
    const HDSpan<bool> start,
    const HDSpan<int> arcOffsets,
    HDSpan<int> degrees,
    HDSpan<int> toExplore,
    HDSpan<ArgMaxT> maxes,
    HDSpan<int> count) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < degrees.size()) {
    degrees[gTid] = arcOffsets[gTid + 1] - arcOffsets[gTid];
    if (start[gTid]) {
      maxes[gTid] = ArgMaxT{make_float2(0, -1)};
      if (degrees[gTid] == 0) {
        int oldCount = atomicAdd(count.data(), 1);
        toExplore[oldCount] = gTid;
      }
    }
  } 
}

void initExplore(
    const HDSpan<bool>& start,
    const HDSpan<int>& arcOffsets,
    HDSpan<int>& degrees,
    HDSpan<int>& toExplore,
    HDSpan<int>& numToExplore,
    HDSpan<ArgMaxT>& maxes) {
  int countHost = 0;
  if (start.size() > 0) {
    int blocks = divUp(start.size(), kNT);
    initExploreKernel<<<blocks, kNT>>>(
        start, arcOffsets, degrees, toExplore, maxes, numToExplore);
    CUDA_CHECK(
        cudaMemcpy((void*)&countHost,
        (void*)numToExplore.begin(),
        sizeof(int),
        cudaMemcpyDeviceToHost));
  }
  toExplore.resize(countHost);
} 

__global__
void arcOffsetsKernel(
    const HDSpan<int> indices,
    const HDSpan<int> allInOffsets,
    const HDSpan<int> allOutOffsets,
    HDSpan<int> inOffsets,
    HDSpan<int> outOffsets,
    HDSpan<int2> sums) {
  int n = inOffsets.size();
  extern __shared__ int temp[];
  int gTid = threadIdx.x;
  int isOutArcs = blockIdx.x;
  int pout = 0, pin = 1;

  auto& allOffsets = isOutArcs ? allOutOffsets : allInOffsets;
  auto& out = isOutArcs ? outOffsets : inOffsets;
  if (gTid > 0 && gTid < n) {
    int nodeIdx = indices[gTid - 1];
    int numArcs = allOffsets[nodeIdx + 1] - allOffsets[nodeIdx];
    temp[pout * n + gTid] = numArcs;
  } else {
    temp[pout * n + gTid] = 0;
  }
  temp[pin * n + gTid] = 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    pout = 1 - pout;
    pin = 1 - pout;
    temp[pout * n + gTid] = temp[pin * n + gTid];
    if (gTid >= offset) {
      temp[pout * n + gTid] += temp[pin * n + gTid - offset];
    }
    __syncthreads();
  }
  if (gTid < n) {
    out[gTid] = temp[pout * n + gTid];
  }
  if (gTid == (n - 1)) {
    if (isOutArcs) {
      sums[0].y = out[gTid];
    } else {
      sums[0].x = out[gTid];
    }
  }
}

int2 getArcOffsets(
    const HDSpan<int>& allInOffsets,
    const HDSpan<int>& allOutOffsets,
    HDSpan<int>& inOffsets,
    HDSpan<int>& outOffsets,
    const HDSpan<int>& indices) {
  int n = indices.size() + 1;
  inOffsets.resize(n);
  outOffsets.resize(n);
  HDSpan<int2> sums(1, Device::CUDA);
  arcOffsetsKernel<<<2, n, 2 * sizeof(int) * n>>>(
      indices, allInOffsets, allOutOffsets,
      inOffsets, outOffsets, sums);
  int2 sumsHost;
  CUDA_CHECK(
    cudaMemcpy((void*)(&sumsHost),
    (void*)sums.begin(),
    sizeof(int2),
    cudaMemcpyDeviceToHost));
  sums.clear();
  return sumsHost;
}

std::tuple<float, float, int> reduceAccept(
    const HDSpan<float>& scores,
    const HDSpan<int>& acceptIds,
    bool tropical) {
  if (acceptIds.size() == 0) {
    return std::make_tuple(kNegInf, kNegInf, 0);
  }
  HDSpan<float> acceptScores(acceptIds.size(), Device::CUDA);
  thrust::device_ptr<const float> sPtr(scores.data());
  thrust::device_ptr<const int> aPtr(acceptIds.data());
  thrust::device_ptr<float> asPtr(acceptScores.data());
  thrust::gather(aPtr, aPtr + acceptIds.size(), sPtr, asPtr);

  auto maxIt = thrust::max_element(asPtr, asPtr + acceptIds.size());
  int maxIdx = maxIt - asPtr;
  float m;
  CUDA_CHECK(cudaMemcpy(
      (void*)(&m), (void* )(acceptScores.data() + maxIdx),
      sizeof(float), cudaMemcpyDeviceToHost));
  float logsum;
  if (tropical || m == kNegInf || m == kInf) {
    logsum = 0;
  } else {
    logsum = std::log(thrust::transform_reduce(
      asPtr, asPtr + acceptIds.size(), SubMaxAndExp(m), (float) 0, thrust::plus<float>()));
  }
  acceptScores.clear();
  return std::make_tuple(m + logsum, m, maxIdx);
}

__global__
void maxArcScoreKernel(
    const GraphData g,
    const HDSpan<int> currExplore,
    const HDSpan<float> scores,
    HDSpan<ArgMaxT> maxes,
    const float* weights,
    const HDSpan<int> arcOffsets,
    int numArcs) {
  int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < numArcs) {
    auto idx = binarySearchBinIndex(arcOffsets.data(), currExplore.size(), gTid);
    auto localIdx = gTid - arcOffsets[idx];
    auto nodeIdx = currExplore[idx];
    auto arcIdx = g.inArcs[g.inArcOffset[nodeIdx] + localIdx];
    int srcNodeIdx = g.srcNodes[arcIdx];
    float score = weights[arcIdx] + scores[srcNodeIdx];
    atomicArgMax(maxes.data() + nodeIdx, score, localIdx);
  }
}

__global__
void reduceArcScoresKernel(
    const GraphData g,
    const HDSpan<int> currExplore,
    HDSpan<float> scores,
    HDSpan<ArgMaxT> maxes,
    const float* weights,
    const HDSpan<int> arcOffsets,
    int numArcs,
    bool tropical) {
  int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < numArcs) {
    auto idx = binarySearchBinIndex(arcOffsets.data(), currExplore.size(), gTid);
    auto nodeIdx = currExplore[idx];
    float maxScore = maxes[nodeIdx].val.x;
    if (tropical || maxScore == kInf || maxScore == kNegInf) {
      scores[nodeIdx] = maxScore;
    } else {
      auto localIdx = gTid - arcOffsets[idx];
      auto arcIdx = g.inArcs[g.inArcOffset[nodeIdx] + localIdx];
      int srcNodeIdx = g.srcNodes[arcIdx];
      float score = weights[arcIdx] + scores[srcNodeIdx];
      auto expScore = expf(score - maxScore);
      if (localIdx == 0 && g.start[nodeIdx]) {
        expScore += expf(-maxScore);
      }
      atomicAdd(scores.data() + nodeIdx, expScore);
    }
  }
}

__global__
void logScoresKernel(
    const HDSpan<int> currExplore,
    const HDSpan<ArgMaxT> maxes, 
    HDSpan<float> scores) {
  int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < currExplore.size()) {
    auto nodeIdx = currExplore[gTid];
    scores[nodeIdx] = logf(scores[nodeIdx]) + maxes[nodeIdx].val.x;
  }
}

__global__
void decrementDegreesKernel(
    const GraphData g,
    const HDSpan<int> currExplore,
    HDSpan<int> nextExplore,
    HDSpan<int> count,
    HDSpan<int> degrees,
    const HDSpan<int> arcOffsets,
    int numArcs) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < numArcs) {
    auto idx = binarySearchBinIndex(arcOffsets.data(), currExplore.size(), gTid);
    auto localIdx = gTid - arcOffsets[idx];
    auto nodeIdx = currExplore[idx];
    auto arcIdx = g.outArcs[g.outArcOffset[nodeIdx] + localIdx];
    auto dstNodeIdx = g.dstNodes[arcIdx];
    auto oldDegrees = atomicSub(degrees.data() + dstNodeIdx, 1);
    if (oldDegrees == 1) {
      int oldCount = atomicAdd(count.data(), 1);
      nextExplore[oldCount] = dstNodeIdx;
    }
  }
}

/*__global__
void setAcceptGradsTropicalKernel(
    const HDSpan<int> acceptIds,
    const int maxId,
    HDSpan<float> nodeGrads) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid >= acceptIds.size()) {
    return;
  }
  int nodeIdx = acceptIds[gTid];
  nodeGrads[nodeIdx] = gTid == maxId ? 1.0 : 0.0;
}

__global__
void setAcceptGradsKernel(
    const HDSpan<int> acceptIds,
    const float output,
    const float maxScore,
    const HDSpan<float> nodeScores,
    HDSpan<float> nodeGrads) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid >= acceptIds.size()) {
    return;
  }
  float denom = expf(output - maxScore);
  int nodeIdx = acceptIds[gTid];
  nodeGrads[nodeIdx] = expf(nodeScores[nodeIdx] - maxScore) / denom;
}

__global__
void updateGradientsKernel(
    GraphData g,
    const float* weights,
    const float* deltas,
    const HDSpan<int> arcOffsets,
    int numArcs,
    const HDSpan<float> maxScores,
    const HDSpan<int> maxIds,
    const HDSpan<float> nodeScores,
    HDSpan<float> nodeGrads,
    HDSpan<float> arcGrads,
    HDSpan<int> degrees,
    const HDSpan<int> nodeIndices,
    HDSpan<bool> toExplore,
    bool tropical) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid >= numArcs) {
    return;
  }
  auto idx = binarySearchBinIndex(arcOffsets.data(), nodeIndices.size(), gTid);
  auto nodeIdx = nodeIndices[idx];
  auto localIdx =  (gTid - arcOffsets[idx]);
  auto arcIdx = g.inArcs[g.inArcOffset[nodeIdx] + localIdx];

  auto srcNode = g.srcNodes[arcIdx];
  float nodeGradUpdate;
  if (tropical) {
    nodeGradUpdate = localIdx == maxIds[nodeIdx] ? 1.0 : 0.0;
  } else {
    auto denom = expf(nodeScores[nodeIdx] - maxScores[nodeIdx]);
    nodeGradUpdate = 
      expf(nodeScores[srcNode] + weights[arcIdx] - maxScores[nodeIdx]) / denom;
  }
  nodeGradUpdate *= nodeGrads[nodeIdx];
  atomicAdd(nodeGrads.data() + srcNode, nodeGradUpdate);
  arcGrads[arcIdx] = nodeGradUpdate * deltas[0];
  auto oldDegrees = atomicSub(degrees.data() + srcNode, 1);
  if (oldDegrees == 1) {
    toExplore[srcNode] = true;
  }
}*/

/*void shortestDistanceGrad(
    Graph& g,
    const float output,
    const float maxScore,
    const int maxId,
    const Graph& deltas,
    const HDSpan<float>& nodeScores,
    const HDSpan<float>& maxScores,
    const HDSpan<int>& maxIds,
    bool tropical) {

  const int NT = 128;
  auto gData = g.getData();
  auto degrees = initDegrees(gData.outArcOffset);
  HDSpan<float> nodeGrads(g.numNodes(), 0.0, Device::CUDA);
  HDSpan<float> arcGrads(g.numArcs(), 0.0, Device::CUDA);

  // Initialize the accept node gradients
  {
    int blocks = divUp(g.numAccept(), NT);
    if (tropical) {
      setAcceptGradsTropicalKernel<<<blocks, NT>>>(
          gData.acceptIds, maxId, nodeGrads);
    } else {
      setAcceptGradsKernel<<<blocks, NT>>>(
          gData.acceptIds, output, maxScore, nodeScores, nodeGrads);
    }
  }

  auto toExplore = initExplore(gData.accept, degrees);
  HDSpan<int> exploreIndices(g.numNodes(), Device::CUDA);
  return;
  while (checkAnyTrue(toExplore)) {
    boolToIndices(toExplore, exploreIndices);
    auto inOffsets = getArcOffsets(gData.inArcOffset, exploreIndices, nullptr);
  
    if (inOffsets.second > 0) {
      int blocks = divUp(inOffsets.second, NT);
      updateGradientsKernel<<<blocks, NT>>>(
          gData,
          g.weights(),
          deltas.weights(),
          inOffsets.first,
          inOffsets.second,
          maxScores,
          maxIds,
          nodeScores,
          nodeGrads,
          arcGrads,
          degrees,
          exploreIndices,
          toExplore,
          tropical);
    }
    inOffsets.first.clear();
  }
  g.addGrad(arcGrads.data());
  exploreIndices.clear();
  toExplore.clear();
  degrees.clear();
  nodeGrads.clear();
  arcGrads.clear();
}*/



} // namespace

Graph shortestDistance(const Graph& g, bool tropical) {
  auto gData = g.getData();
  HDSpan<float> scores(g.numNodes(), 0.0f, Device::CUDA);
  HDSpan<ArgMaxT> maxes(g.numNodes(), Device::CUDA);
  HDSpan<int> degrees(g.numNodes(), Device::CUDA);
  HDSpan<int> currExplore(g.numNodes(), Device::CUDA); 
  HDSpan<int> nextExplore(g.numNodes(), Device::CUDA); 
  HDSpan<int> numToExplore(1, 0, Device::CUDA);
  initExplore(
      gData.start, gData.inArcOffset, degrees, currExplore, numToExplore, maxes);

  HDSpan<int> inOffsets(g.numNodes(), Device::CUDA);
  HDSpan<int> outOffsets(g.numNodes(), Device::CUDA);
  while (currExplore.size() > 0) {
    auto numArcs = getArcOffsets(
        gData.inArcOffset, gData.outArcOffset, inOffsets, outOffsets, currExplore);
    if (numArcs.x > 0) {
      int blocks = divUp(numArcs.x, kNT);
      maxArcScoreKernel<<<blocks, kNT, 0>>>(
          gData, currExplore, scores, maxes, g.weights(), inOffsets, numArcs.x);
      reduceArcScoresKernel<<<blocks, kNT, 0>>>(
          gData, currExplore, scores, maxes, g.weights(), inOffsets, numArcs.x, tropical);
      blocks = divUp(currExplore.size(), kNT);
      logScoresKernel<<<blocks, kNT, 0>>>(currExplore, maxes, scores);
    }
    CUDA_CHECK(cudaMemsetAsync(numToExplore.begin(), 0, sizeof(int)));
    if (numArcs.y > 0) {
      int blocks = divUp(numArcs.y, kNT);
      decrementDegreesKernel<<<blocks, kNT>>>(
          gData, currExplore, nextExplore, numToExplore,
          degrees, outOffsets, numArcs.y);
      swap(currExplore, nextExplore);
    }
    int countHost;
    CUDA_CHECK(
        cudaMemcpy((void*)&countHost,
        (void*)numToExplore.begin(),
        sizeof(int),
        cudaMemcpyDeviceToHost));
    currExplore.resize(countHost);
  }
  currExplore.clear();
  nextExplore.clear();
  numToExplore.clear();
  degrees.clear();
  inOffsets.clear();
  outOffsets.clear();
  
  // Gather and log add the accept scores
  auto scoreAndMax = reduceAccept(scores, gData.acceptIds, tropical);

  auto gradInfo = std::make_shared<GradInfo>(scores, maxes);
  auto gradFunc = [scoreAndMax, gradInfo, tropical](
      std::vector<Graph>& inputs, Graph deltas) {
/*    shortestDistanceGrad(
        inputs[0],
        std::get<0>(scoreAndMax),
        std::get<1>(scoreAndMax),
        std::get<2>(scoreAndMax),
        deltas,
        gradInfo->scores,
        gradInfo->maxScores,
        gradInfo->maxIds,
        tropical);*/
  };

  // Make the result graph
  auto ngraph = scalarGraph(
      std::get<0>(scoreAndMax), Device::CUDA, g.calcGrad());
  ngraph.setInputs({g});
  ngraph.setGradFunc(gradFunc);
  return ngraph;
}
} // namespace detail
} // namespace cuda 
} // namespace gtn
