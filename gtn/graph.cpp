/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "graph.h"

namespace gtn {

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
  sharedGraph_->numNodes++;
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
  sharedGraph_->compiled = false;
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
  sharedGraph_->numArcs++;
  sharedWeights_->weights.push_back(weight);
  sharedGraph_->ilabels.push_back(ilabel);
  sharedGraph_->olabels.push_back(olabel);
  sharedGraph_->srcNodes.push_back(srcNode);
  sharedGraph_->dstNodes.push_back(dstNode);
  sharedGraph_->ilabelSorted = false;
  sharedGraph_->olabelSorted = false;
  sharedGraph_->compiled = false;
  return idx;
}

void Graph::compile() const {
  sharedGraph_->compiled = true;

  auto computeArcsAndOffsets = [numNodes=numNodes(), numArcs=numArcs()](
      const std::vector<int>& arcNodes,
      std::vector<int>& offsets,
      std::vector<int>& arcs) {
    offsets.resize(numNodes + 1);
    // Count the number of arcs for each node
    std::vector<int> counts(numNodes, 0);
    for (int i = 0; i < numArcs; ++i) {
      counts[arcNodes[i]] += 1;
    }
    // Prefix sum scan to compute the offsets
    int sum = 0;
    for (int i = 0; i < counts.size(); ++i) {
      int c = counts[i];
      counts[i] = 0;
      offsets[i] = sum;
      sum += c;
    }
    offsets.back() = sum;
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

void Graph::addGrad(std::vector<float>&& other) {
  if (isCuda()) {
    throw std::logic_error(
      "[Graph::addGrad] Use addGrad(const float*) for GPU graphs.");
  }
  if (calcGrad()) {
    if (other.size() != numArcs()) {
      throw std::logic_error("[Graph::addGrad] Invalid grad size.");
    }
    std::lock_guard<std::mutex> lock(sharedGraph_->grad_lock);
    if (isGradAvailable()) {
      for (int i = 0; i < numArcs(); i++) {
        grad().setWeight(i, grad().weight(i) + other[i]);
      }
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      sharedGrad_->grad->sharedWeights_->weights = std::move(other);
    }
  }
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
    std::lock_guard<std::mutex> lock(sharedGraph_->grad_lock);
    if (isGradAvailable()) {
      for (int i = 0; i < numArcs(); i++) {
        grad().setWeight(i, grad().weight(i) + other[i]);
      }
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      sharedGrad_->grad->sharedWeights_->weights = other;
    }
  }
}

void Graph::addGrad(const Graph& other) {
  if (other.isCuda() != isCuda() || (isCuda() && device() != other.device())) {
    throw std::invalid_argument("[Graph::addGrad] device mismach");
  }

  if (isCuda()) {
    addGrad(other.weights());
  } else {
    addGrad(other.sharedWeights_->weights);
  }
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
  Graph out(src.calcGrad());
  out.sharedGraph_->numNodes = src.numNodes();
  out.sharedGraph_->numArcs = src.numArcs();
  out.sharedGraph_->compiled = src.sharedGraph_->compiled;
  out.sharedGraph_->isCuda = src.isCuda();
  out.sharedGraph_->start = src.sharedGraph_->start;
  out.sharedGraph_->startIds = src.sharedGraph_->startIds;
  out.sharedGraph_->accept = src.sharedGraph_->accept;
  out.sharedGraph_->acceptIds = src.sharedGraph_->acceptIds;
  out.sharedGraph_->inArcOffset = src.sharedGraph_->inArcOffset;
  out.sharedGraph_->outArcOffset = src.sharedGraph_->outArcOffset;
  out.sharedGraph_->inArcs = src.sharedGraph_->inArcs;
  out.sharedGraph_->outArcs = src.sharedGraph_->outArcs;
  out.sharedGraph_->ilabels = src.sharedGraph_->ilabels;
  out.sharedGraph_->olabels = src.sharedGraph_->olabels;
  out.sharedGraph_->srcNodes = src.sharedGraph_->srcNodes;
  out.sharedGraph_->dstNodes = src.sharedGraph_->dstNodes;
  out.sharedWeights_->weights = src.sharedWeights_->weights;
  if (out.isCuda()) {
    out.sharedGraph_->device = src.device();
    out.sharedGraph_->deviceData.deepCopy(
        src.sharedGraph_->deviceData,
        src.device());
    out.sharedWeights_->deepCopy(
        src.sharedWeights_->deviceWeights,
        src.numArcs(),
        src.device());
  }
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

void Graph::setWeights(float* weights) {
  if (isCuda()) {
    sharedWeights_->deviceWeights = weights;
  } else {
    std::copy(weights, weights + numArcs(), this->weights());
  }
}

void Graph::setWeights(const float* weights) {
  if (isCuda()) {
    throw std::invalid_argument(
      "[Graph::setWeights] Weights can only be set on CPU graphs");
  }
  setWeights(const_cast<float*>(weights));
}

void Graph::labelsToArray(int* out, bool ilabel) {
  if (isCuda()) {
    throw std::invalid_argument(
      "[Graph::labelsToArray] Labels can only be retrieved on CPU graphs");
  }
  // TODO copy the vectors/arrays directly here
  for (int i = 0; i < numArcs(); ++i) {
    out[i] = ilabel ? this->ilabel(i) : olabel(i);
  }
}

std::vector<int> Graph::labelsToVector(bool ilabel) {
  std::vector<int> out(numArcs());
  labelsToArray(out.data(), ilabel);
  return out;
}

} // namespace gtn
