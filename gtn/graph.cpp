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

using namespace gtn::detail;

namespace gtn {

namespace{

auto makeSharedGraph(Device device) {
  return std::shared_ptr<Graph::SharedGraph>{
    new Graph::SharedGraph{device},
    [](Graph::SharedGraph* g) {
      g->free();
      delete g;}};
}

auto makeSharedWeights(Device device) {
  return std::shared_ptr<detail::HDSpan<float>>{
      new detail::HDSpan<float>{device},
      [](detail::HDSpan<float>* w) {
        w->clear();
        delete w;}};
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
    auto g = cuda(inputs[0].device());
    sharedGraph_ = g.sharedGraph_;
    sharedWeights_ = g.sharedWeights_;
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
  sharedGraph_->start.push_back(start);
  sharedGraph_->accept.push_back(accept);
  if (start) {
    sharedGraph_->startIds.push_back(idx);
  }
  if (accept) {
    sharedGraph_->acceptIds.push_back(idx);
  }
  sharedGraph_->ilabelSorted = false;
  sharedGraph_->olabelSorted = false;
  uncompile();
  sharedGraph_->numNodes++;
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
  sharedWeights_->push_back(weight);
  sharedGraph_->ilabels.push_back(ilabel);
  sharedGraph_->olabels.push_back(olabel);
  sharedGraph_->srcNodes.push_back(srcNode);
  sharedGraph_->dstNodes.push_back(dstNode);
  sharedGraph_->ilabelSorted = false;
  sharedGraph_->olabelSorted = false;
  uncompile();
  sharedGraph_->numArcs++;
  return numArcs() - 1;
}

void Graph::makeAccept(size_t i) {
  if (!sharedGraph_->accept[i]) {
    sharedGraph_->acceptIds.push_back(static_cast<int>(i));
    sharedGraph_->accept[i] = true;
  }
};

void Graph::compile() const {
  sharedGraph_->compiled = true;

  auto computeArcsAndOffsets = [numNodes=numNodes(), numArcs=numArcs()](
      const detail::HDSpan<int>& arcNodes,
      detail::HDSpan<int>& offsets,
      detail::HDSpan<int>& arcs) {
    // Count the number of arcs for each node
    std::vector<int> counts(numNodes, 0);
    for (int i = 0; i < numArcs; ++i) {
      counts[arcNodes[i]] += 1;
    }
    // Prefix sum scan to compute the offsets
    int sum = 0;
    offsets.resize(numNodes + 1);
    for (int i = 0; i < counts.size(); ++i) {
      int c = counts[i];
      counts[i] = 0;
      offsets[i] = sum;
      sum += c;
    }
    offsets[numNodes] = sum;
    // Record the index of each node's arcs
    arcs.resize(numArcs);
    for (int i = 0; i < numArcs; ++i) {
      auto n = arcNodes[i];
      arcs[offsets[n] + counts[n]] = i;
      counts[n] += 1;
    }
  };

  computeArcsAndOffsets(
      sharedGraph_->dstNodes,
      sharedGraph_->inArcOffset,
      sharedGraph_->inArcs);
  computeArcsAndOffsets(
      sharedGraph_->srcNodes,
      sharedGraph_->outArcOffset,
      sharedGraph_->outArcs);
}

float Graph::item() const {
  if (numArcs() > 1) {
    throw std::invalid_argument(
        "[Graph::item] Cannot convert Graph with more than 1 arc to a scalar.");
  }
  if (numArcs() == 0) {
    throw std::invalid_argument(
        "[Graph::item] Cannot convert Graph with no arcs to a scalar.");
  }
  if (isCuda()) {
    HDSpan<float> w(1);
    w = getWeights();
    return w[0];
  } else {
    return weight(0);
  }
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
      sharedGrad_->grad->sharedWeights_ = makeSharedWeights(device());
      sharedGrad_->grad->setWeights(other.data());
    }
  }
}

void Graph::addGrad(const float* other) {
  if (calcGrad()) {
    std::lock_guard<std::mutex> lock(sharedGrad_->grad_lock);
    if (isGradAvailable()) {
      add(
          HDSpan<float>(numArcs(), const_cast<float*>(other), device()),
          grad().getWeights(),
          grad().getWeights());
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      sharedGrad_->grad->sharedWeights_ = makeSharedWeights(device());
      sharedGrad_->grad->setWeights(other);
    }
  }
}

void Graph::addGrad(const Graph& other) {
  if (device() != other.device()) {
    throw std::invalid_argument("[Graph::addGrad] device mismach");
  }
  if (calcGrad() && other.numArcs() != numArcs()) {
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

Graph Graph::deepCopy(const Graph& src) {
  return deepCopy(src, src.device());
}

Graph Graph::deepCopy(const Graph& src, Device device_) {
  if (device_.isCuda()) {
    src.maybeCompile();
  }
  Graph out(src.calcGrad());
  out.sharedGraph_ = makeSharedGraph(device_);
  out.sharedGraph_->numNodes = src.numNodes();
  out.sharedGraph_->numArcs = src.numArcs();
  out.sharedGraph_->compiled = src.sharedGraph_->compiled;
  out.sharedGraph_->startIds = src.sharedGraph_->startIds;
  out.sharedGraph_->acceptIds = src.sharedGraph_->acceptIds;
  out.sharedGraph_->start = src.sharedGraph_->start;
  out.sharedGraph_->accept = src.sharedGraph_->accept;
  out.sharedGraph_->ilabels = src.sharedGraph_->ilabels;
  out.sharedGraph_->olabels = src.sharedGraph_->olabels;
  out.sharedGraph_->srcNodes = src.sharedGraph_->srcNodes;
  out.sharedGraph_->dstNodes = src.sharedGraph_->dstNodes;
  out.sharedGraph_->inArcOffset = src.sharedGraph_->inArcOffset;
  out.sharedGraph_->outArcOffset = src.sharedGraph_->outArcOffset;
  out.sharedGraph_->inArcs = src.sharedGraph_->inArcs;
  out.sharedGraph_->outArcs = src.sharedGraph_->outArcs;
  out.sharedGraph_->ilabelSorted = src.ilabelSorted();
  out.sharedGraph_->olabelSorted = src.olabelSorted();
  out.sharedWeights_ = makeSharedWeights(device_);
  *(out.sharedWeights_) = *(src.sharedWeights_);
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
        sharedGraph_->inArcs.begin() + start,
        sharedGraph_->inArcs.begin() + end,
        sortFn);
    start = sharedGraph_->outArcOffset[i];
    end = sharedGraph_->outArcOffset[i + 1];
    std::sort(
        sharedGraph_->outArcs.begin() + start,
        sharedGraph_->outArcs.begin() + end,
        sortFn);
  }
}

void Graph::setWeights(const float* weights) {
  sharedWeights_->resize(numArcs());
  sharedWeights_->copy(weights);
}

std::vector<int> Graph::labelsToVector(bool ilabel) {
  if (isCuda()) {
    throw std::invalid_argument(
      "[Graph::labelsToVector] Labels can only be retrieved on CPU graphs");
  }
  std::vector<int> out(numArcs());
  auto labels = ilabel ?
    sharedGraph_->ilabels.data() : sharedGraph_->olabels.data();
  std::copy(labels, labels + numArcs(), out.begin());
  return out;
}

Graph Graph::to(const Device& device_) const {
  if (device_.isCuda() && !cuda::isAvailable()) {
    std::logic_error("[Graph::to] CUDA not available.");
  }
  // No-op if already on device_
  if (device() == device_) {
    return *this;
  }
  return deepCopy(*this, device_);
}

Graph Graph::cpu() const {
  return to(Device::CPU);
}

Graph Graph::cuda(const Device& device_) const {
  return to(device_);
}

Graph Graph::cuda() const {
  return cuda(Device::CUDA);
}

} // namespace gtn
