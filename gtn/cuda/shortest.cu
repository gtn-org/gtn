#include <thrust/device_vector.h>
#include <thrust/fill.h>

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

__constant__ float kInf = std::numeric_limits<float>::infinity();
__constant__ float kNegInf = -std::numeric_limits<float>::infinity();
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

struct GradInfo {

  HDSpan<float> scores;
  HDSpan<ArgMaxT> maxes;
  HDSpan<float> score;
  HDSpan<ArgMaxT> maxAndIdx;

  GradInfo(
      const HDSpan<float>& scores,
      const HDSpan<ArgMaxT>& maxes,
      const HDSpan<float>& score,
      const HDSpan<ArgMaxT>& maxAndIdx) :
    scores(scores), maxes(maxes), score(score), maxAndIdx(maxAndIdx) { };

  ~GradInfo() {
    scores.clear();
    maxes.clear();
    score.clear();
    maxAndIdx.clear();
  };
};

__host__ __device__
inline int divUp(int x, int y) {
  return (x + y - 1) / y;
}

__global__
void initExploreKernel(
    const HDSpan<bool> flags,
    const HDSpan<int> arcOffsets,
    HDSpan<int> degrees,
    HDSpan<int> toExplore,
    HDSpan<int> count) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < degrees.size()) {
    degrees[gTid] = arcOffsets[gTid + 1] - arcOffsets[gTid];
    if (flags[gTid] && degrees[gTid] == 0) {
      int oldCount = atomicAdd(count.data(), 1);
      toExplore[oldCount] = gTid;
    }
  } 
}

void initExplore(
    const HDSpan<bool>& flags,
    const HDSpan<int>& arcOffsets,
    HDSpan<int>& degrees,
    HDSpan<int>& toExplore,
    HDSpan<int>& numNodes) {
  int numNodesH = 0;
  if (flags.size() > 0) {
    int blocks = divUp(flags.size(), kNT);
    initExploreKernel<<<blocks, kNT>>>(
        flags, arcOffsets, degrees, toExplore, numNodes);
    CUDA_CHECK(
      cudaMemcpy((void*)&numNodesH,
      (void*)numNodes.begin(),
      sizeof(int),
      cudaMemcpyDeviceToHost));
  }
  toExplore.resize(numNodesH);
} 

__global__
void reduceAccept(
    const HDSpan<float> scores,
    const HDSpan<int> acceptIds,
    HDSpan<float> score,
    HDSpan<ArgMaxT> maxAndIdx,
    bool tropical) {
  score[0] = 0.0;
  maxAndIdx[0] = {kNegInf, -1};
  __syncthreads();

  for (int gTid = threadIdx.x; gTid < acceptIds.size(); gTid += blockDim.x) {
    int nodeIdx = acceptIds[gTid];
    atomicArgMax(maxAndIdx.data(), scores[nodeIdx], gTid);
  }
  __syncthreads();
  float maxScore = maxAndIdx[0].val.x;
  if (tropical || maxScore == kInf || maxScore == kNegInf) {
    score[0] = maxScore;
  } else {
    for (int gTid = threadIdx.x; gTid < acceptIds.size(); gTid += blockDim.x) {
      int nodeIdx = acceptIds[gTid];
      atomicAdd(score.data(), expf(scores[nodeIdx] - maxScore));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      score[0] = logf(score[0]) + maxScore;
    }
  }
}

__global__
void reduceArcScoresKernel(
    const GraphData g,
    const HDSpan<int> currExplore,
    HDSpan<float> scores,
    HDSpan<ArgMaxT> maxes,
    const float* weights,
    bool tropical) {
  // max and score should be in shared memory
  int nodeIdx = currExplore[blockIdx.x];
  int numArcs = g.inArcOffset[nodeIdx + 1] - g.inArcOffset[nodeIdx]; 
  numArcs += g.start[nodeIdx];
  // Compute the max score
  ArgMaxT maxAndIdx = {kNegInf, -1};
  for (int gTid = threadIdx.x; gTid < numArcs; gTid += blockDim.x) {
    float score;
    if (gTid == numArcs - 1 && g.start[nodeIdx]) {
      score = 0;
    } else {
      auto arcIdx = g.inArcs[g.inArcOffset[nodeIdx] + gTid];
      int srcNodeIdx = g.srcNodes[arcIdx];
      score = weights[arcIdx] + scores[srcNodeIdx];
    }
    if (score > maxAndIdx.val.x) {
      maxAndIdx.val.x = score;
      maxAndIdx.idx.y = gTid;
    }
  }
  if (threadIdx.x < numArcs) {
    atomicArgMax(maxes.data() + nodeIdx, maxAndIdx.val.x, maxAndIdx.idx.y);
  }
  __syncthreads();
  float maxScore = maxes[nodeIdx].val.x;
  if (tropical || maxScore == kInf || maxScore == kNegInf) {
    scores[nodeIdx] = maxScore;
  } else {
    float expScore = 0.0;
    for (int gTid = threadIdx.x; gTid < numArcs; gTid += blockDim.x) {
      float score;
      if (gTid == numArcs - 1 && g.start[nodeIdx]) {
        score = 0.0;
      } else {
        auto arcIdx = g.inArcs[g.inArcOffset[nodeIdx] + gTid];
        int srcNodeIdx = g.srcNodes[arcIdx];
        score = weights[arcIdx] + scores[srcNodeIdx];
      }
      expScore += expf(score - maxScore);
    }
    if (threadIdx.x < numArcs) {
      atomicAdd(scores.data() + nodeIdx, expScore);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      scores[nodeIdx] = logf(scores[nodeIdx]) + maxScore;
    }
  }
}

__global__
void decrementDegreesKernel(
    const GraphData g,
    const HDSpan<int> currExplore,
    HDSpan<int> nextExplore,
    HDSpan<int> count,
    HDSpan<int> degrees) {
  int nodeIdx = currExplore[blockIdx.x];
  int numArcs = g.outArcOffset[nodeIdx + 1] - g.outArcOffset[nodeIdx];
  for (int gTid = threadIdx.x; gTid < numArcs; gTid += blockDim.x) {
    auto arcIdx = g.outArcs[g.outArcOffset[nodeIdx] + gTid];
    auto dstNodeIdx = g.dstNodes[arcIdx];
    auto oldDegrees = atomicSub(degrees.data() + dstNodeIdx, 1);
    if (oldDegrees == 1) {
      int oldCount = atomicAdd(count.data(), 1);
      nextExplore[oldCount] = dstNodeIdx;
    }
  }
}

__global__
void setAcceptGradsKernel(
    const HDSpan<int> acceptIds,
    const HDSpan<float> score,
    const HDSpan<ArgMaxT> maxAndIdx,
    const HDSpan<float> nodeScores,
    HDSpan<float> nodeGrads,
    bool tropical) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid >= acceptIds.size()) {
    return;
  }
  int nodeIdx = acceptIds[gTid];
  if (tropical) {
    nodeGrads[nodeIdx] = gTid == maxAndIdx[0].idx.y ? 1.0 : 0.0;
  } else {
    float maxScore = maxAndIdx[0].val.x;
    float denom = expf(score[0] - maxScore);
    nodeGrads[nodeIdx] = expf(nodeScores[nodeIdx] - maxScore) / denom;
  }
}

__global__
void updateGradientsKernel(
    GraphData g,
    const float* weights,
    const float* deltas,
    const HDSpan<ArgMaxT> maxes,
    const HDSpan<float> nodeScores,
    HDSpan<float> nodeGrads,
    HDSpan<float> arcGrads,
    HDSpan<int> degrees,
    const HDSpan<int> currExplore,
    HDSpan<int> nextExplore,
    HDSpan<int> count,
    bool tropical) {
  int nodeIdx = currExplore[blockIdx.x];
  int numArcs = g.inArcOffset[nodeIdx + 1] - g.inArcOffset[nodeIdx]; 
  for (int gTid = threadIdx.x; gTid < numArcs; gTid += blockDim.x) {
    auto arcIdx = g.inArcs[g.inArcOffset[nodeIdx] + gTid];
    auto srcNode = g.srcNodes[arcIdx];
    float nodeGradUpdate;
    if (tropical) {
      nodeGradUpdate = (gTid == maxes[nodeIdx].idx.y) ? 1.0 : 0.0;
    } else {
      float maxScore = maxes[nodeIdx].val.x;
      auto denom = expf(nodeScores[nodeIdx] - maxScore);
      nodeGradUpdate = 
        expf(nodeScores[srcNode] + weights[arcIdx] - maxScore) / denom;
    }
    nodeGradUpdate *= nodeGrads[nodeIdx];
    atomicAdd(nodeGrads.data() + srcNode, nodeGradUpdate);
    arcGrads[arcIdx] = nodeGradUpdate * deltas[0];
    auto oldDegrees = atomicSub(degrees.data() + srcNode, 1);
    if (oldDegrees == 1) {
      int oldCount = atomicAdd(count.data(), 1);
      nextExplore[oldCount] = srcNode;
    }
  }
}

void shortestDistanceGrad(
    Graph& g,
    const Graph& deltas,
    const HDSpan<float>& nodeScores,
    const HDSpan<ArgMaxT>& maxScores,
    const HDSpan<float>& score,
    const HDSpan<ArgMaxT>& maxAndIdx,
    bool tropical) {
  auto gData = g.getData();
  HDSpan<int> degrees(g.numNodes(), Device::CUDA);
  HDSpan<int> currExplore(g.numNodes(), Device::CUDA); 
  HDSpan<int> nextExplore(g.numNodes(), Device::CUDA); 
  HDSpan<int> numToExplore(1, 0, Device::CUDA);
  HDSpan<float> nodeGrads(g.numNodes(), 0.0, Device::CUDA);
  HDSpan<float> arcGrads(g.numArcs(), 0.0, Device::CUDA);

  initExplore(
      gData.accept,
      gData.outArcOffset,
      degrees,
      currExplore,
      numToExplore);

  // Initialize the accept node gradients
  if (g.numAccept() > 0) {
    int blocks = divUp(g.numAccept(), kNT);
    setAcceptGradsKernel<<<blocks, kNT>>>(
        gData.acceptIds, score, maxAndIdx, nodeScores, nodeGrads, tropical);
  }

  while (currExplore.size() > 0) {
    int arcsPerNode = divUp(g.numArcs(), g.numNodes());
    int NT = std::min(1024, divUp(arcsPerNode, 32) * 32);
    CUDA_CHECK(cudaMemsetAsync(numToExplore.begin(), 0, sizeof(int)));
    updateGradientsKernel<<<currExplore.size(), NT>>>(
        gData,
        g.weights(),
        deltas.weights(),
        maxScores,
        nodeScores,
        nodeGrads,
        arcGrads,
        degrees,
        currExplore,
        nextExplore,
        numToExplore,
        tropical);
    swap(currExplore, nextExplore);
    int countHost;
    CUDA_CHECK(cudaMemcpy(
        (void*)&countHost,
        (void*)numToExplore.begin(),
        sizeof(int),
        cudaMemcpyDeviceToHost));
    currExplore.resize(countHost);
  }

  g.addGrad(arcGrads.data());
  nodeGrads.clear();
  arcGrads.clear();
  currExplore.clear();
  nextExplore.clear();
  numToExplore.clear();
}

} // namespace

Graph shortestDistance(const Graph& g, bool tropical) {
  auto gData = g.getData();
  HDSpan<float> scores(g.numNodes(), 0.0f, Device::CUDA);
  HDSpan<ArgMaxT> maxes(g.numNodes(), Device::CUDA);
  HDSpan<int> degrees(g.numNodes(), Device::CUDA);
  HDSpan<int> currExplore(g.numNodes(), Device::CUDA); 
  HDSpan<int> nextExplore(g.numNodes(), Device::CUDA); 
  HDSpan<int> numToExplore(1, 0, Device::CUDA);

  thrust::fill(
      thrust::device,
      maxes.begin(),
      maxes.end(),
      ArgMaxT{-std::numeric_limits<float>::infinity(), -1});

  initExplore(
      gData.start, gData.inArcOffset, degrees, currExplore, numToExplore);

  cudaStream_t str;
  cudaStreamCreate(&str);
  while (currExplore.size() > 0) {
    int arcsPerNode = divUp(g.numArcs(), g.numNodes());
    int NT = std::min(1024, divUp(arcsPerNode, 32) * 32);
    reduceArcScoresKernel<<<currExplore.size(), NT, 0, str>>>(
        gData, currExplore, scores, maxes, g.weights(), tropical);
    CUDA_CHECK(cudaMemsetAsync(numToExplore.begin(), 0, sizeof(int)));
    decrementDegreesKernel<<<currExplore.size(), NT>>>(
        gData, currExplore, nextExplore, numToExplore, degrees);
    swap(currExplore, nextExplore);
    int countHost;
    CUDA_CHECK(cudaMemcpyAsync(
        (void*)&countHost,
        (void*)numToExplore.begin(),
        sizeof(int),
        cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(0);
    currExplore.resize(countHost);
  }
  cudaStreamSynchronize(str);
  cudaStreamDestroy(str);
  currExplore.clear();
  nextExplore.clear();
  numToExplore.clear();
  degrees.clear();
  
  // Gather and log add the accept scores
  HDSpan<float> score(1, Device::CUDA);;
  HDSpan<ArgMaxT> maxAndIdx(1, Device::CUDA);
  int NT = std::min(divUp(g.numAccept() + 1, 32) * 32, 1024);
  reduceAccept<<<1, NT>>>(
      scores, gData.acceptIds, score, maxAndIdx, tropical);

  auto gradInfo = std::make_shared<GradInfo>(
      scores, maxes, score, maxAndIdx);
  auto gradFunc = [gradInfo, tropical](
      std::vector<Graph>& inputs, Graph deltas) {
    shortestDistanceGrad(
        inputs[0],
        deltas,
        gradInfo->scores,
        gradInfo->maxes,
        gradInfo->score,
        gradInfo->maxAndIdx,
        tropical);
  };

  // Make the result graph
  auto ngraph = scalarGraph(0, Device::CUDA, g.calcGrad());
  ngraph.getWeights() = score;
  ngraph.setInputs({g});
  ngraph.setGradFunc(gradFunc);
  return ngraph;
}
} // namespace detail
} // namespace cuda 
} // namespace gtn
