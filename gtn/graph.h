/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "hd_span.h"

namespace gtn {

/** The index of the epsilon label. */
constexpr int epsilon{-1};

/**
 * A `Graph` class to perform automatic differentiation with weighted
 * finite-state acceptors (WFSAs) and transducers (WFSTs).
 *
 * Example:
 *
 * \code{.cpp}
 * Graph graph;
 * graph.addNode(true); // Add a start node
 * graph.addNode(); // Add an internal node
 * graph.addNode(false, true); // Add an accept node
 *
 * // Add an arc from node 0 to 1 with ilabel 0, olabel 1 and weight 2.0
 * graph.addArc(0, 1, 0, 1, 2.0);
 *
 * // Add an arc from node 1 to 2 with ilabel 1, olabel 2 and weight 1.0
 * graph.addArc(1, 2, 1, 2, 1.0);
 *
 * // Compute the Viterbi score of the graph
 * auto score = viterbiScore(graph);
 *
 * print(score); // Print the score graph to std out
 *
 * backward(score); // Compute the gradient
 * graph.grad(); // Access the gradient
 * graph.zeroGrad(); // Clear the gradient
 * \endcode
 *
 * All operations are in the log or tropical semirings. The default score
 * for an arc is `0` (e.g. the multiplicative identity) and the additive
 * identity is `-infinity`. Path scores are accumulated with log-sum-exp or
 * max operations and the score for a path is accumulated with addition.
 */
class Graph {

 public:
  struct ArcPtr {
    int* ptr_;
    int n_;
    int operator[](const int idx) const { return ptr_[idx]; };
    const int* begin() const { return ptr_; };
    const int* end() const { return ptr_ + n_; };
    size_t size() const { return n_; };
  };

  struct SharedGraph {
    SharedGraph(Device device = Device::CPU) :
      device(device) { }

    // Device data
    Device device;

    /// Underlying graph data
    size_t numNodes{0};
    size_t numArcs{0};

    detail::HDSpan<int> startIds{device};
    detail::HDSpan<int> acceptIds{device};
    detail::HDSpan<bool> accept{device};
    detail::HDSpan<bool> start{device};

    // One value per node - i-th value corresponds to i-th node
    // Last element is the total number of arcs, so that
    // each element and its neighbor forms a range
    detail::HDSpan<int> inArcOffset{device};
    detail::HDSpan<int> outArcOffset{device};

    // One value per arc
    detail::HDSpan<int> inArcs{device};
    detail::HDSpan<int> outArcs{device};

    // One value per arc
    // i-th value corresponds to i-th arc
    detail::HDSpan<int> ilabels{device};
    detail::HDSpan<int> olabels{device};
    detail::HDSpan<int> srcNodes{device};
    detail::HDSpan<int> dstNodes{device};

    // Some optional metadata about the graph
    bool ilabelSorted{false};
    bool olabelSorted{false};
    bool compiled{device.isCuda()};

    void free() {
      startIds.clear();
      acceptIds.clear();
      start.clear();
      accept.clear();
      ilabels.clear();
      olabels.clear();
      srcNodes.clear();
      dstNodes.clear();
      inArcOffset.clear();
      outArcOffset.clear();
      inArcs.clear();
      outArcs.clear();
    };
  };

  using GradFunc =
      std::function<void(std::vector<Graph>& inputs, Graph& deltas)>;
  Graph(GradFunc gradFunc, std::vector<Graph> inputs);

  /**
   * \defgroup graphMethods Graph-level methods
   * @{
   */

  /** Construct a `Graph`.
   * @param calcGrad Whether or not to compute gradients with respect to this
   *   graph when calling `gtn::backward`.
   */
  Graph(bool calcGrad = true);

  /**
   * Adds a node to the graph.
   * @param start Indicates if the node is a starting node.
   * @param accept Indicates if the node is an accepting node.
   * @return The id of the node (used for e.g. adding arcs).
   */
  int addNode(bool start = false, bool accept = false);

  /**
   * Add a arc between two nodes. This assumes the graph is an acceptor, the
   * input label on the arc is the same as the output label.
   * @param srcNode The id of the source node.
   * @param dstNode The id of the destination node.
   * @param label The arc label.
   * @return The id of the added arc.
   */
  size_t addArc(size_t srcNode, size_t dstNode, int label);

  /**
   * Add a arc between two nodes.
   * @param srcNode The id of the source node.
   * @param dstNode The id of the destination node.
   * @param ilabel The arc input label.
   * @param olabel The arc output label.
   * @param weight The arc weight.
   * @return The id of the added arc.
   */
  size_t addArc(
      size_t srcNode,
      size_t dstNode,
      int ilabel,
      int olabel,
      float weight = 0.0);

  /** The number of arcs in the graph. */
  size_t numArcs() const {
    return sharedGraph_->numArcs;
  };
  /** The number of nodes in the graph. */
  size_t numNodes() const {
    return sharedGraph_->numNodes;
  };
  /** The number of starting nodes in the graph. */
  size_t numStart() const {
    return sharedGraph_->startIds.size();;
  };
  /** The number of accepting nodes in the graph. */
  size_t numAccept() const {
    return sharedGraph_->acceptIds.size();
  };

  /** Get the weight on a single arc graph.  */
  float item() const;

  /**
   * A deep copy of a graph `src` which is not recorded in the
   * autograd tape. For a version which is recorded in the
   * autograd tape see `gtn::clone`.
   */
  static Graph deepCopy(const Graph& src);

  /**
   * A deep copy of a graph `src` which is not recorded in the
   * autograd tape. For a version which is recorded in the
   * autograd tape see `gtn::clone`.
   *
   * @param src The source graph to copy from.
   * @param device The device to place the copy on
   */
  static Graph deepCopy(const Graph& src, Device device);

  /**
   * Sort the arcs entering and exiting a node in increasing order by arc in
   * label or out label if `olabel == true`. This function is intended
   * to be used prior to calls to `intersect` and `compose` to improve the
   * efficiency of the algorithm.
   */
  void arcSort(bool olabel = false);

  /**
   * Mark a graph's arcs as sorted.
   * If `olabel == false` then the graph will be marked as sorted by
   * arc input labels, otherwise it will be marked as sorted by the arc output
   * labels.
   */
  void markArcSorted(bool olabel = false) {
    if (olabel) {
      sharedGraph_->olabelSorted = true;
    } else {
      sharedGraph_->ilabelSorted = true;
    }
  }

  /**
   * Check if the arcs entering and exiting every node are sorted by input
   * label.
   */
  bool ilabelSorted() const {
    return sharedGraph_->ilabelSorted;
  }

  /**
   * Check if the arcs entering and exiting every node are sorted by output
   * label.
   */
  bool olabelSorted() const {
    return sharedGraph_->olabelSorted;
  }

  const detail::HDSpan<float>& getWeights() const {
    return *sharedWeights_;
  }

  detail::HDSpan<float>& getWeights() {
    return *sharedWeights_;
  }

  /**
   * Returns an array of weights from a graph. The array will contain
   * `Graph::numArcs()` elements.
   */
  float* weights() {
    assert(sharedWeights_ != nullptr);
    return sharedWeights_->data();
  }
  /**
   * A `const` version of `Graph::weights`.
   */
  const float* weights() const {
    assert(sharedWeights_ != nullptr);
    return sharedWeights_->data();
  }

  /**
   * Set the arc weights on a graph. The `weights` array must have
   * `Graph::numArcs()` elements.
   */
  void setWeights(const float* weights);

  /**
   * Extract a `std::vector` of labels from the graph.
   *
   * @param[in] ilabel Retreive ilabels if true, otherwise gets olabels.
   */
  std::vector<int> labelsToVector(bool ilabel = true);

  /**
   * Return a copy of the graph on the given device.
   *
   * The original graph is returned if it is already on the specified device.
   */
  Graph to(const Device& device) const;

  /**
   * Return a copy of the graph on the CPU.
   *
   * The original graph is returned if it is already on the CPU.
   */
  Graph cpu() const;

  /**
   * Return a copy of the graph on the currently active GPU.
   */
  Graph cuda() const;

  /**
   * Return a copy of the graph on the GPU specified by `device`.
   *
   * The original graph is returned if it is already on the specified device.
   */
  Graph cuda(const Device& device) const;

  /**
   * Get the `SharedGraph` object for the graph.
   */
  const Graph::SharedGraph& getData() const {
    return *sharedGraph_;
  }

  /**
   * Get a modifiable `SharedGraph` object for the graph.
   */
  Graph::SharedGraph& getData() {
    return *sharedGraph_;
  }

  /**
   * Returns true if the graph is on the GPU.
   */
  bool isCuda() const {
    return sharedGraph_->device.isCuda();
  }

  /**
   * Get the GPU device the graph is on.
   */
  Device device() const {
    return sharedGraph_->device;
  }

  /** @}*/

  /**
   * \defgroup gradMethods Autograd methods
   * @{
   */

  /**
   * Add a `std::vector` of gradients to the gradient graph weights. The
   * `Graph::addGrad` methods are intended for use by the autograd.
   */
  void addGrad(const std::vector<float>& other);

  /**
   * Add a `float*` of gradients to the gradient graph weights. The gradient
   * must be on the same device as the graph.
   **/
  void addGrad(const float* grad);

  /**
   * Add a `Graph` of gradients to the gradient graph. The `Graph::addGrad`
   * methods are intended for use by the autograd. The input graph `other` must
   * be on the same device as the graph.
   */
  void addGrad(const Graph& other);

  /** Check if a graph requires a gradient. */
  bool calcGrad() const {
    return sharedGrad_->calcGrad;
  };
  /** Check if a graph's gradient is computed. */
  bool isGradAvailable() const {
    return sharedGrad_->grad != nullptr;
  }
  /** Get the gradient graph. */
  Graph& grad();

  /** A `const` version of `Graph::grad`. */
  const Graph& grad() const;

  /** Specify if the gradient for this graph should be computed. */
  void setCalcGrad(bool calcGrad);

  /** Clear the graph's gradients. */
  void zeroGrad();

  /**
   * A unique identifier for a graph. Intended for use by the autograd.
   */
  std::uintptr_t id();

  /**
   * Get the gradient function of a graph. Intended for use by the autograd.
   */
  GradFunc gradFunc() {
    return sharedGrad_->gradFunc;
  };

  /**
   * Set the gradient function of a graph. Intended for use by the autograd.
   */
  void setGradFunc(GradFunc gradFunc) {
    if (calcGrad()) {
      sharedGrad_->gradFunc = gradFunc;
    }
  }

  /**
   * Get the vector of inputs used in the autograd computation graph. Intended
   * for use by the autograd.
   */
  std::vector<Graph>& inputs() const {
    return sharedGrad_->inputs;
  };

  /**
   * Sets the vector of inputs used in the autograd computation graph. Intended
   * for use by the autograd.
   */
  void setInputs(std::vector<Graph> inputs);

  /**
   * Clear the weights on a graph if they are no longer needed. Intended for
   * use by the autograd.
   */
  Graph withoutWeights() const {
    Graph other = *this;
    other.sharedWeights_ = nullptr;
    return other;
  }

  /** @} */

  /** \defgroup nodeAccess Node accessors
   *  @{
   */

  /** Get the indices of the start nodes of the graph. */
  std::vector<int> start() const {
    return std::vector<int>(
        sharedGraph_->startIds.begin(), sharedGraph_->startIds.end());
  };
  /** Get the indices of the accepting nodes of the graph. */
  std::vector<int> accept() const {
    return std::vector<int>(
        sharedGraph_->acceptIds.begin(), sharedGraph_->acceptIds.end());
  };
  /** Check if the `i`-th node is a start node. */
  bool isStart(size_t i) const {
    return sharedGraph_->start[i];
  };
  /** Check if the `i`-th node is an accepting node. */
  bool isAccept(size_t i) const {
    return sharedGraph_->accept[i];
  };
  /** Make the the `i`-th node an accepting node. */
  void makeAccept(size_t i);

  /** The number of outgoing arcs from the `i`-th node. */
  size_t numOut(size_t i) const {
    maybeCompile();
    return sharedGraph_->outArcOffset[i+1] - sharedGraph_->outArcOffset[i];
  }
  /** Get the indices of outgoing arcs from the `i`-th node. */
  ArcPtr out(size_t i) const {
    maybeCompile();
    auto start = sharedGraph_->outArcOffset[i];
    auto end = sharedGraph_->outArcOffset[i + 1];
    return ArcPtr{sharedGraph_->outArcs.begin() + start, end - start};
  }
  /** Get the index of the `j`-th outgoing arc from the `i`-th node. */
  int out(size_t i, size_t j) const {
    maybeCompile();
    return sharedGraph_->outArcs[sharedGraph_->outArcOffset[i] + j];
  }
  /** The number of incoming arcs to the `i`-th node. */
  size_t numIn(size_t i) const {
    maybeCompile();
    return sharedGraph_->inArcOffset[i+1] - sharedGraph_->inArcOffset[i];
  }
  /** Get the indices of incoming arcs to the `i`-th node. */
  ArcPtr in(size_t i) const {
    maybeCompile();
    auto start = sharedGraph_->inArcOffset[i];
    auto end = sharedGraph_->inArcOffset[i + 1];
    return ArcPtr{sharedGraph_->inArcs.begin() + start, end - start};
  }
  /** Get the index of the `j`-th incoming arc to the `i`-th node. */
  size_t in(size_t i, size_t j) const {
    maybeCompile();
    return sharedGraph_->inArcs[sharedGraph_->inArcOffset[i] + j];
  }

  /** @}*/

  /** \defgroup arcAccess Arc accessors
   *  @{
   */

  /** The source node of the `i`-th arc. */
  int srcNode(size_t i) const {
    return sharedGraph_->srcNodes[i];
  }
  /** The destination node of the `i`-th arc. */
  int dstNode(size_t i) const {
    return sharedGraph_->dstNodes[i];
  }
  /** The label of the `i`-th arc (use this for acceptors). */
  int label(size_t i) const {
    return sharedGraph_->ilabels[i];
  }
  /** The input label of the `i`-th arc. */
  int ilabel(size_t i) const {
    return sharedGraph_->ilabels[i];
  }
  /** The output label of the `i`-th arc. */
  int olabel(size_t i) const {
    return sharedGraph_->olabels[i];
  }

  /** The weight of the `i`-th arc. */
  float weight(size_t i) const {
    assert(sharedWeights_ != nullptr);
    return (*sharedWeights_)[i];
  }
  /** Set the weight of the `i`-th arc. */
  void setWeight(size_t i, float weight) {
    assert(sharedWeights_ != nullptr);
    (*sharedWeights_)[i] = weight;
  }
  /** @}*/

 private:
  // Attempt to keep code like `g.addArc(n1, n2, 0, 2.0)` from compiling
  size_t addArc(size_t srcNode, size_t dstNode, int label, float) = delete;
  size_t addArc(size_t srcNode, size_t dstNode, int label, double) = delete;

  // Semantically const
  void compile() const;

  // Semantically const
  void maybeCompile() const {
    if (!sharedGraph_->compiled) {
      compile();
    }
  }

  void uncompile() {
    sharedGraph_->inArcs.clear();
    sharedGraph_->outArcs.clear();
    sharedGraph_->inArcOffset.clear();
    sharedGraph_->outArcOffset.clear();
    sharedGraph_->compiled = false;
  }

  struct SharedGrad {
    /// Underlying grad data
    GradFunc gradFunc{nullptr};
    std::vector<Graph> inputs;
    std::unique_ptr<Graph> grad{nullptr};
    bool calcGrad;
    std::mutex grad_lock;
  };

  std::shared_ptr<SharedGraph> sharedGraph_{
    new SharedGraph{},
    [](SharedGraph* g) {
      g->free();
      delete g;}
  };
  std::shared_ptr<detail::HDSpan<float>> sharedWeights_{
    new detail::HDSpan<float>{},
    [](detail::HDSpan<float>* w) {
      w->clear();
      delete w;}
  };
  std::shared_ptr<SharedGrad> sharedGrad_{std::make_shared<SharedGrad>()};
};

} // namespace gtn
