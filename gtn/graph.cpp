/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "graph.h"
#include "cuda/cuda.h"

namespace gtn {

namespace{

void push_back(float** arr, size_t size, float val) {
  // If the current size is a power of 2, then we need to allocate more space
  if (!(size & (size - 1))) {
    size_t space = size ? (size << 1) : 1;
    auto newarr = new float[space];
    std::copy(*arr, *arr + size, newarr);
    delete[] *arr;
    (*arr) = newarr;
  }
  (*arr)[size] = val;
}

void push_back(int** arr, size_t size, int val) {
  // If the current size is a power of 2, then we need to allocate more space
  if (!(size & (size - 1))) {
    size_t space = size ? (size << 1) : 1;
    auto newarr = new int[space];
    std::copy(*arr, *arr + size, newarr);
    delete[] *arr;
    (*arr) = newarr;
  }
  (*arr)[size] = val;
}

} // namespace

Graph::Graph(GradFunc gradFunc, std::vector<Graph> inputs) {
  sharedGrad_->calcGrad = false;
  // If any inputs require a gradient, then this should
  // also compute a gradient.
  for (auto& g : inputs) {
    sharedGrad_->calcGrad |= g.calcGrad();
  }
  if (!inputs.empty() && inputs[0].isCuda()) {
    sharedGraph_ = cuda(inputs[0].device()).sharedGraph_;
  }
  if (calcGrad()) {
    sharedGrad_->gradFunc = std::move(gradFunc);
    sharedGrad_->inputs = std::move(inputs);
  }
}

Graph::Graph(bool calcGrad /* = true */) {
  sharedGrad_->calcGrad = calcGrad;
}

int Graph::addNode(bool start /* = false */, bool accept /* = false */) {
  int idx = static_cast<int>(numNodes());
  push_back(&sharedGraph_->start, numNodes(), start);
  push_back(&sharedGraph_->accept, numNodes(), accept);
  sharedGraph_->numNodes++;
  if (start) {
    push_back(&sharedGraph_->startIds, numStart(), idx);
    sharedGraph_->numStart++;
  }
  if (accept) {
    push_back(&sharedGraph_->acceptIds, numAccept(), idx);
    sharedGraph_->numAccept++;
  }
  sharedGraph_->ilabelSorted = false;
  sharedGraph_->olabelSorted = false;
  uncompile();
  return idx;
}

size_t Graph::addArc(size_t srcNode, size_t dstNode, int label) {
  return addArc(srcNode, dstNode, label, label);
}

size_t Graph::addArc(
    size_t srcNode,
    size_t dstNode,
    int ilabel,
    int olabel,
    float weight /* = 0 */) {
  assert(ilabel >= epsilon && olabel >= epsilon);
  int idx = static_cast<int>(numArcs());
  push_back(&sharedWeights_->weights, numArcs(), weight);
  push_back(&sharedGraph_->ilabels, numArcs(), ilabel);
  push_back(&sharedGraph_->olabels, numArcs(), olabel);
  push_back(&sharedGraph_->srcNodes, numArcs(), srcNode);
  push_back(&sharedGraph_->dstNodes, numArcs(), dstNode);
  sharedGraph_->numArcs++;
  sharedGraph_->ilabelSorted = false;
  sharedGraph_->olabelSorted = false;
  uncompile();
  return idx;
}

void Graph::makeAccept(size_t i) {
  if (!sharedGraph_->accept[i]) {
    push_back(&sharedGraph_->acceptIds, numAccept(), static_cast<int>(i));
    sharedGraph_->accept[i] = true;
    sharedGraph_->numAccept++;
  }
};

void Graph::compile() const {
  sharedGraph_->compiled = true;

  auto computeArcsAndOffsets = [numNodes=numNodes(), numArcs=numArcs()](
      int* arcNodes,
      int** offsets,
      int** arcs) {
    // Count the number of arcs for each node
    std::vector<int> counts(numNodes, 0);
    for (int i = 0; i < numArcs; ++i) {
      counts[arcNodes[i]] += 1;
    }
    // Prefix sum scan to compute the offsets
    int sum = 0;
    *offsets = new int[numNodes + 1];
    for (int i = 0; i < counts.size(); ++i) {
      int c = counts[i];
      counts[i] = 0;
      (*offsets)[i] = sum;
      sum += c;
    }
    (*offsets)[numNodes] = sum;
    // Record the index of each node's arcs
    *arcs = new int[numArcs];
    for (int i = 0; i < numArcs; ++i) {
      auto n = arcNodes[i];
      (*arcs)[(*offsets)[n] + counts[n]] = i;
      counts[n] += 1;
    }
  };

  computeArcsAndOffsets(
      sharedGraph_->dstNodes,
      &sharedGraph_->inArcOffset,
      &sharedGraph_->inArcs);
  computeArcsAndOffsets(
      sharedGraph_->srcNodes,
      &sharedGraph_->outArcOffset,
      &sharedGraph_->outArcs);
}

float Graph::item() const {
  if (isCuda()) {
    throw std::invalid_argument(
      "[Graph::item] Can only get scalar from CPU graphs");
  }
  if (numArcs() != 1) {
    throw std::invalid_argument(
        "[Graph::item] Cannot convert Graph with more than 1 arc to a scalar.");
  }
  return weight(0);
}

Graph& Graph::grad() {
  return const_cast<Graph&>(static_cast<const Graph&>(*this).grad());
}

const Graph& Graph::grad() const {
  if (!calcGrad()) {
    throw std::logic_error("[Graph::grad] Gradient calculation disabled.");
  }
  if (!sharedGrad_->grad) {
    throw std::logic_error("[Graph::grad] Gradient not calculated yet.");
  }
  return *sharedGrad_->grad;
}

void Graph::addGrad(const std::vector<float>& other) {
  if (isCuda()) {
    throw std::logic_error(
      "[Graph::addGrad] Use addGrad(const float*) for GPU graphs.");
  }
  if (calcGrad()) {
    if (other.size() != numArcs()) {
      throw std::logic_error("[Graph::addGrad] Invalid grad size.");
    }
    std::lock_guard<std::mutex> lock(sharedGrad_->grad_lock);
    if (isGradAvailable()) {
      auto w = grad().weights();
      std::transform(
          other.begin(), other.end(), w, w, std::plus<>());
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      sharedGrad_->grad->sharedWeights_->deepCopy(
          other.data(), numArcs(), isCuda(), device());
    }
  }
}

void Graph::addGrad(const float* other) {
  if (calcGrad()) {
    std::lock_guard<std::mutex> lock(sharedGrad_->grad_lock);
    if (isGradAvailable()) {
      cuda::detail::add(
          other, grad().weights(), grad().weights(), numArcs(), isCuda());
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      sharedGrad_->grad->sharedWeights_->deepCopy(
          other, numArcs(), isCuda(), device());
    }
  }
}

void Graph::addGrad(const Graph& other) {
  if (other.isCuda() != isCuda() || (isCuda() && device() != other.device())) {
    throw std::invalid_argument("[Graph::addGrad] device mismach");
  }
  if (calcGrad() & other.numArcs() != numArcs()) {
    throw std::logic_error("[Graph::addGrad] Invalid grad size.");
  }
  addGrad(other.weights());
}

void Graph::setCalcGrad(bool calcGrad) {
  sharedGrad_->calcGrad = calcGrad;
  if (!calcGrad) {
    sharedGrad_->gradFunc = nullptr;
    sharedGrad_->inputs.clear();
    sharedGrad_->grad.reset();
  }
}

void Graph::setInputs(std::vector<Graph> inputs) {
  sharedGrad_->inputs = std::move(inputs);
}

void Graph::zeroGrad() {
  sharedGrad_->grad.reset();
}

std::uintptr_t Graph::id() {
  return reinterpret_cast<std::uintptr_t>(sharedGrad_.get());
}

namespace {
inline int sizeUp(size_t size) {
  auto n = static_cast<int>(ceil(log2(size)));
  return 1 << n;
}
} // namespace

void Graph::SharedGraph::allocHost() {
  startIds = new int[sizeUp(numStart)];
  acceptIds = new int[sizeUp(numAccept)];
  start = new int[sizeUp(numNodes)];
  accept = new int[sizeUp(numNodes)];
  inArcOffset = new int[sizeUp(numNodes + 1)];
  outArcOffset = new int[sizeUp(numNodes + 1)];
  inArcs = new int[sizeUp(numArcs)];
  outArcs = new int[sizeUp(numArcs)];
  ilabels = new int[sizeUp(numArcs)];
  olabels = new int[sizeUp(numArcs)];
  srcNodes = new int[sizeUp(numArcs)];
  dstNodes = new int[sizeUp(numArcs)];
}

void Graph::SharedGraph::freeHost() {
  delete[] startIds;
  delete[] acceptIds;
  delete[] start;
  delete[] accept;
  delete[] ilabels;
  delete[] olabels;
  delete[] srcNodes;
  delete[] dstNodes;
  if (compiled) {
    delete[] inArcOffset;
    delete[] outArcOffset;
    delete[] inArcs;
    delete[] outArcs;
  }
}

void Graph::SharedWeights::allocHost(size_t numArcs) {
  weights = new float[sizeUp(numArcs)];
}

Graph::SharedWeights::~SharedWeights() {
  if (isCuda) {
    cuda::detail::free(weights);
  } else {
    delete[] weights;
  }
}

Graph Graph::deepCopy(const Graph& src) {
  src.maybeCompile();
  Graph out(src.calcGrad());
  out.sharedGraph_->compiled = src.sharedGraph_->compiled;
  out.sharedGraph_->isCuda = src.isCuda();
  out.sharedGraph_->device = src.device();
  out.sharedGraph_->deepCopy(*(src.sharedGraph_));
  out.sharedWeights_->deepCopy(
      src.sharedWeights_->weights,
      src.numArcs(),
      src.isCuda(),
      src.device());
  return out;
}

void Graph::arcSort(bool olabel /* = false */) {
  if (isCuda()) {
    throw std::invalid_argument("[Graph::arcSort] Can only sort CPU graphs");
  }
  if ((olabel && sharedGraph_->olabelSorted) ||
      (!olabel && sharedGraph_->ilabelSorted)) {
    return;
  }
  maybeCompile();
  sharedGraph_->olabelSorted = olabel;
  sharedGraph_->ilabelSorted = !olabel;
  auto& labels = olabel ? sharedGraph_->olabels : sharedGraph_->ilabels;
  auto sortFn = [&labels](int a, int b) {
    return labels[a] < labels[b];
  };
  for (int i = 0; i < numNodes(); ++i) {
    auto start = sharedGraph_->inArcOffset[i];
    auto end = sharedGraph_->inArcOffset[i + 1];
    std::sort(
        sharedGraph_->inArcs + start,
        sharedGraph_->inArcs + end,
        sortFn);
    start = sharedGraph_->outArcOffset[i];
    end = sharedGraph_->outArcOffset[i + 1];
    std::sort(
        sharedGraph_->outArcs + start,
        sharedGraph_->outArcs + end,
        sortFn);
  }
}

void Graph::setWeights(float* weights) {
  // TODO free old weights
  sharedWeights_->deepCopy(weights, numArcs(), isCuda(), device());
}

void Graph::setWeights(const float* weights) {
  // TODO free old weights
  sharedWeights_->deepCopy(weights, numArcs(), isCuda(), device());
}

void Graph::labelsToArray(int* out, bool ilabel) {
  if (isCuda()) {
    throw std::invalid_argument(
      "[Graph::labelsToArray] Labels can only be retrieved on CPU graphs");
  }
  // TODO std::copy the vectors/arrays directly here
  for (int i = 0; i < numArcs(); ++i) {
    out[i] = ilabel ? this->ilabel(i) : olabel(i);
  }
}

std::vector<int> Graph::labelsToVector(bool ilabel) {
  std::vector<int> out(numArcs());
  labelsToArray(out.data(), ilabel);
  return out;
}

Graph Graph::cpu() const {
  // No-op if already on CPU
  if (!sharedGraph_->isCuda) {
    return *this;
  }
  //maybeCompile(); TODO cuda graphs must always be compiled
  Graph g;
  auto& od = *(g.sharedGraph_);
  od.isCuda = false;
  od.compiled = true;
  g.setCalcGrad(calcGrad());
  g.sharedGraph_->deepCopy(*sharedGraph_);
  g.sharedWeights_->deepCopy(
      sharedWeights_->weights, numArcs(), false, 0);
  return g;
}

Graph Graph::cuda(int device_) const {
  if (!cuda::isAvailable()) {
    std::logic_error("[Graph::cuda] CUDA not available.");
  }
  // No-op if already on GPU
  if (isCuda() && device() == device_) {
    return *this;
  }
  maybeCompile();
  Graph g;
  auto& od = *(g.sharedGraph_);
  od.isCuda = true;
  od.compiled = true;
  g.setCalcGrad(calcGrad());
  g.sharedGraph_->device = device_;
  g.sharedGraph_->deepCopy(*sharedGraph_);
  g.sharedWeights_->deepCopy(
      sharedWeights_->weights, numArcs(), g.isCuda(), g.device());
  return g;
}

Graph Graph::cuda() const {
  return cuda(cuda::getDevice());
}

} // namespace gtn
