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

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct MaxFunc {
  __device__
  thrust::tuple<float, int> operator()(
      const thrust::tuple<float, int>& lhs,
      const thrust::tuple<float, int>& rhs){
    return thrust::get<0>(lhs) > thrust::get<0>(rhs) ? lhs : rhs;
  };
};

struct SubMaxAndExp{
  SubMaxAndExp(float maxVal) : maxVal(maxVal) {};
  float maxVal;

  __device__ float operator()(const float& in) {
      return expf(in - maxVal);
  };
};

struct LogAndAddMax {
  __device__ float operator()(const float& val, const float& max) {
    if (max == kNegInf || max == kInf) {
      return max;
    }
    return logf(val) + max;
  };
};

struct GradInfo {

  HDSpan<float> scores;
  HDSpan<float> maxScores;
  HDSpan<int> maxIds;

  GradInfo(
      const HDSpan<float>& scores,
      const HDSpan<float>& maxScores,
      const HDSpan<int>& maxIds) :
    scores(scores), maxScores(maxScores), maxIds(maxIds) { };

  ~GradInfo() {
    scores.clear();
    maxScores.clear();
    maxIds.clear();
  };
};

inline int divUp(int x, int y) {
  return (x + y - 1) / y;
}

bool checkAnyTrue(const HDSpan<bool>& flags) {
  thrust::device_ptr<const bool> tPtr(flags.data());
  return thrust::any_of(tPtr, tPtr + flags.size(), thrust::identity<bool>());
}

void setFalse(HDSpan<bool>& flags) {
  cuda::detail::fill(flags.data(), false, flags.size());
}

std::tuple<int*, int> prefixSumScan(const bool* input, size_t numElts) {
  const size_t scanNumElts = numElts + 1;

  HDSpan<int> output(scanNumElts, Device::CUDA);
  thrust::device_ptr<const bool> iPtr(input);
  thrust::device_ptr<int> oPtr(output.data());
  thrust::exclusive_scan(iPtr, iPtr + numElts, oPtr, (int) 0);

  int sum;
  bool lastVal;
  CUDA_CHECK(cudaMemcpy((void*)(&sum), (void* )(&(output[scanNumElts-2])), sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(&lastVal), (void* )(&(input[scanNumElts-2])), sizeof(bool), cudaMemcpyDeviceToHost));
  sum += lastVal;
  CUDA_CHECK(cudaMemcpy((void*)(&(output[scanNumElts-1])),(void*)(&sum), sizeof(int), cudaMemcpyHostToDevice));

  return std::make_tuple(output.data(), sum);
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

HDSpan<int> initDegrees(const HDSpan<int>& arcOffsets) {
  HDSpan<int> degrees(arcOffsets.size() - 1, Device::CUDA);
  thrust::device_ptr<const int> oPtr(arcOffsets.data());
  thrust::device_ptr<int> dPtr(degrees.data());
  thrust::transform(
    oPtr + 1, oPtr + arcOffsets.size(), oPtr, dPtr, thrust::minus<int>());
  return degrees;
}


__global__
void initExploreKernel(
    const HDSpan<bool> flags, const HDSpan<int> degrees, HDSpan<bool> toExplore) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < flags.size()) {
    toExplore[gTid] = flags[gTid] && degrees[gTid] == 0;
  } 
}

HDSpan<bool> initExplore(const HDSpan<bool>& flags, const HDSpan<int>& degrees) {
  HDSpan<bool> toExplore(flags.size(), false, Device::CUDA);
  if (flags.size() > 0) {
    int NT = 128;
    int blocks = divUp(flags.size(), NT);
    initExploreKernel<<<blocks, NT>>>(flags, degrees, toExplore);
  }
  return toExplore;
}

__global__
void arcCountsKernel(
    const HDSpan<int> allOffsets,
    const HDSpan<int> indices,
    HDSpan<int> counts,
    const bool* flags) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < indices.size()) {
    counts[gTid] = allOffsets[indices[gTid] + 1] - allOffsets[indices[gTid]];
    if (flags != nullptr) {
      counts[gTid] += flags[indices[gTid]];
    }
  }
}

std::pair<HDSpan<int>, int> getArcOffsets(
    const HDSpan<int>& allOffsets,
    const HDSpan<int>& indices,
    const bool* flags) {
  HDSpan<int> offsets(indices.size() + 1, Device::CUDA);
  int NT = 128;
  int blocks = divUp(indices.size(), NT);
  arcCountsKernel<<<blocks, NT>>>(allOffsets, indices, offsets, flags); 
  thrust::device_ptr<int> oPtr(offsets.data());
  thrust::exclusive_scan(oPtr, oPtr + offsets.size(), oPtr);
  int sum;
  CUDA_CHECK(
    cudaMemcpy((void*)(&sum), (void*)(&(offsets[offsets.size() - 1])), sizeof(int), cudaMemcpyDeviceToHost));
  return std::make_pair(offsets, sum);
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
void gatherScoresKernel(
    const GraphData g,
    const HDSpan<int> arcOffsets,
    HDSpan<float> arcScores,
    HDSpan<int> arcIds,
    HDSpan<int> nodeKeys,
    const HDSpan<float> nodeScores,
    const float* weights,
    const HDSpan<int> nodeIndices) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < arcScores.size()) {
    auto idx = binarySearchBinIndex(arcOffsets.data(), nodeIndices.size(), gTid);
    auto localIdx = gTid - arcOffsets[idx];
    auto nodeIdx = nodeIndices[idx];
    // Special case to add an score of 0 for start nodes
    if (g.start[nodeIdx] && gTid == arcOffsets[idx + 1] - 1) {
      arcScores[gTid] = 0;
    } else {
      auto arcIdx = g.inArcs[g.inArcOffset[nodeIdx] + localIdx];
      auto srcNodeIdx = g.srcNodes[arcIdx];
      auto weight = weights[arcIdx];
      arcScores[gTid] = nodeScores[srcNodeIdx] + weight;
    }
    arcIds[gTid] = localIdx;
    nodeKeys[gTid] = idx;
  }
}

__global__
void removeMaxExponentialKernel(
    HDSpan<float> arcScores,
    const HDSpan<int> nodeIds,
    const HDSpan<float> maxVals) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < arcScores.size()) {
    arcScores[gTid] = expf(arcScores[gTid] - maxVals[nodeIds[gTid]]);
  }
}

__global__
void scatterScoresAndMaxKernel(
    const HDSpan<int> exploreIndices,
    const HDSpan<float> localScores,
    HDSpan<float> scores,
    const HDSpan<float> localMaxScores,
    HDSpan<float> maxScores,
    const HDSpan<int> localMaxIds,
    HDSpan<int> maxIds) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < exploreIndices.size()) {
    auto nodeIdx = exploreIndices[gTid];
    scores[nodeIdx] = localScores[gTid];
    maxScores[nodeIdx] = localMaxScores[gTid];
    maxIds[nodeIdx] = localMaxIds[gTid];
  }
}

void reduceByKey(
    HDSpan<float>& arcScores,
    HDSpan<int>& arcIds,
    const HDSpan<int>& nodeKeys,
    const HDSpan<int>& exploreIndices,
    HDSpan<float>& nodeScores,
    HDSpan<float>& maxScores,
    HDSpan<int>& maxIds,
    bool tropical) {
  int numNodes = exploreIndices.size();

  // Compute the maximum value by key
  thrust::device_ptr<const int> kInPtr(nodeKeys.data());
  thrust::device_ptr<float> vInPtr(arcScores.data());
  thrust::device_ptr<int> arcIdsPtr(arcIds.data());
  HDSpan<int> outKeys(numNodes, Device::CUDA);
  HDSpan<float> localMaxScores(numNodes, Device::CUDA);
  HDSpan<int> localMaxIds(numNodes, Device::CUDA);
  thrust::device_ptr<int> kOutPtr(outKeys.data());
  thrust::device_ptr<float> maxScoresPtr(localMaxScores.data());
  thrust::device_ptr<int> maxIdsPtr(localMaxIds.data());
  thrust::reduce_by_key(
      kInPtr,
      kInPtr + nodeKeys.size(),
      thrust::make_zip_iterator(thrust::make_tuple(vInPtr, arcIdsPtr)),
      kOutPtr,
      thrust::make_zip_iterator(thrust::make_tuple(maxScoresPtr, maxIdsPtr)),
      thrust::equal_to<int>(),
      MaxFunc());
  outKeys.clear();

  HDSpan<float> scores(Device::CUDA);
  if (!tropical) {
    // Remove the max and exponentiate each arc score
    int NT = 128;
    int blocks = divUp(arcScores.size(), NT);
    removeMaxExponentialKernel<<<blocks, NT>>>(arcScores, nodeKeys, localMaxScores); 
    // Sum by key to get exponentiated scores
    scores.resize(numNodes);
    thrust::device_ptr<float> scorePtr(scores.data());
    thrust::reduce_by_key(
        kInPtr, kInPtr + nodeKeys.size(), vInPtr, kOutPtr, scorePtr);
    // Take log and add back max 
    thrust::transform(
        scorePtr, scorePtr + numNodes, maxScoresPtr, scorePtr, LogAndAddMax());
  }

  // Scatter the scores, max score, and argmax
  {
    int NT = 128;
    int blocks = divUp(numNodes, NT);
    scatterScoresAndMaxKernel<<<blocks, NT>>>(
        exploreIndices,
        tropical ? localMaxScores : scores,
        nodeScores,
        localMaxScores,
        maxScores,
        localMaxIds,
        maxIds);
  }

  scores.clear();
  localMaxScores.clear();
  localMaxIds.clear();
}

__global__
void decrementDegreesKernel(
    const GraphData g,
    HDSpan<int> degrees,
    HDSpan<bool> toExplore,
    const HDSpan<int> arcOffsets,
    int numArcs,
    const HDSpan<int> nodeIndices) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gTid < numArcs) {
    auto idx = binarySearchBinIndex(arcOffsets.data(), nodeIndices.size(), gTid);
    auto localIdx = gTid - arcOffsets[idx];
    auto nodeIdx = nodeIndices[idx];
    auto arcIdx = g.outArcs[g.outArcOffset[nodeIdx] + localIdx];
    auto dstNodeIdx = g.dstNodes[arcIdx];
    auto oldDegrees = atomicSub(degrees.data() + dstNodeIdx, 1);
    assert(oldDegrees >= 0);
    if (oldDegrees == 1) {
      toExplore[dstNodeIdx] = true;
    }
  }
}

__global__
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
}

void shortestDistanceGrad(
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
  while (checkAnyTrue(toExplore)) {
    auto exploreIndices = boolToIndices(toExplore);
    setFalse(toExplore);
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
    exploreIndices.clear();
    inOffsets.first.clear();
  }

  g.addGrad(arcGrads.data());
  toExplore.clear();
  degrees.clear();
  nodeGrads.clear();
  arcGrads.clear();
}

} // namespace

Graph shortestDistance(const Graph& g, bool tropical) {
  auto gData = g.getData();
  HDSpan<float> scores(g.numNodes(), Device::CUDA);
  HDSpan<float> maxScores(g.numNodes(), Device::CUDA);
  HDSpan<int> maxIds(g.numNodes(), Device::CUDA);
  auto degrees = initDegrees(gData.inArcOffset);
  auto toExplore = initExplore(gData.start, degrees);
  while(checkAnyTrue(toExplore)) {

    auto exploreIndices = boolToIndices(toExplore);
    auto inOffsets = getArcOffsets(
        gData.inArcOffset, exploreIndices, gData.start.data());
    if (inOffsets.second > 0) {
      int NT = 128;
      int blocks = divUp(inOffsets.second, NT);
      HDSpan<float> arcScores(inOffsets.second, Device::CUDA);
      HDSpan<int> arcIds(inOffsets.second, Device::CUDA);
      HDSpan<int> nodeKeys(inOffsets.second, Device::CUDA);
      gatherScoresKernel<<<blocks, NT>>>(
          gData, inOffsets.first, arcScores, arcIds, nodeKeys, scores, g.weights(), exploreIndices);

      reduceByKey(
          arcScores, arcIds, nodeKeys, exploreIndices, scores, maxScores, maxIds, tropical);

      arcIds.clear();
      arcScores.clear();
      nodeKeys.clear();
    }
    inOffsets.first.clear();

    setFalse(toExplore);
    auto outOffsets = getArcOffsets(gData.outArcOffset, exploreIndices, nullptr);
    if (outOffsets.second > 0) {
      int NT = 128;
      int blocks = divUp(outOffsets.second, NT);
      decrementDegreesKernel<<<blocks, NT>>>(
          gData, degrees, toExplore, outOffsets.first, outOffsets.second, exploreIndices);
    }
    exploreIndices.clear();
    outOffsets.first.clear();
  }
  degrees.clear();
  toExplore.clear();

  // Gather and log add the accept scores
  auto scoreAndMax = reduceAccept(scores, gData.acceptIds, tropical);

  if (tropical) {
    maxScores.clear();
  } else {
    maxIds.clear();
  }
  auto gradInfo = std::make_shared<GradInfo>(scores, maxScores, maxIds);
  auto gradFunc = [scoreAndMax, gradInfo, tropical](
      std::vector<Graph>& inputs, Graph deltas) {
    shortestDistanceGrad(
        inputs[0],
        std::get<0>(scoreAndMax),
        std::get<1>(scoreAndMax),
        std::get<2>(scoreAndMax),
        deltas,
        gradInfo->scores,
        gradInfo->maxScores,
        gradInfo->maxIds,
        tropical);
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
