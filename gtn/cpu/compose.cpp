/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <queue>

#include "gtn/cpu/compose.h"

namespace gtn {
namespace cpu {
namespace {

struct ExploreState {
    ExploreState(
        int first,
        int second,
        bool followFirst = false,
        bool followSecond = false) :
      first(first),
      second(second),
      followFirst(followFirst),
      followSecond(followSecond) {};
    int first;
    int second;
    bool followFirst;
    bool followSecond;
};

inline size_t toIndex(int n1, int n2, const Graph& g, bool first, bool second) {
  int offset = first ? 1 : 0;
  offset = second ? 2 : offset;
  return offset + 3 * (n1 + g.numNodes() * n2);
}

inline size_t toIndex(const ExploreState& state, const Graph& g) {
  return toIndex(
      state.first, state.second, g, state.followFirst, state.followSecond);
}


/* Check reachability via edges with epsilon labels */
void epsilonReachable(
    const Graph& first,
    const Graph& second,
    const ExploreState& state,
    std::vector<bool>& reachable,
    std::queue<ExploreState>& toExplore) {
  bool secondOrFirst = state.followSecond;
  auto edges =
      secondOrFirst ? second.in(state.second) : first.in(state.first);
  auto isSorted =
      secondOrFirst ? second.ilabelSorted() : first.olabelSorted();

  for (auto i : edges) {
    auto label = secondOrFirst ? second.ilabel(i) : first.olabel(i);
    if (label != epsilon) {
      if (isSorted) {
        break;
      } else {
        continue;
      }
    }
    auto fn = secondOrFirst ? state.first : first.srcNode(i);
    auto sn = secondOrFirst ? second.srcNode(i) : state.second;
    auto idx = toIndex(fn, sn, first, false, false);

    // If we haven't seen states before, explore them.
    if (!reachable[idx]) {
      toExplore.emplace(fn, sn);
      reachable[idx] = true;
    }
    int offset = secondOrFirst ? 2 : 1;
    if (!reachable[idx + offset]) {
      toExplore.emplace(fn, sn, !secondOrFirst, secondOrFirst);
      reachable[idx + offset] = true;
    }
  }
}

/*
 * Find any state in the new composed graph which can reach
 * an accepting state.
 *
 * This is accomplished by iteratively following backwards pairwise arc paths
 * from the first and second graphs where the olabel for the first arc equals
 * the ilabel for the second arc.
 */
auto findReachable(
    const Graph& first,
    const Graph& second,
    std::shared_ptr<ArcMatcher> matcher) {
  std::vector<bool> reachable(first.numNodes() * second.numNodes() * 3, false);
  std::queue<ExploreState> toExplore;
  // toExplore -- add accepting node pairs
  for (auto f : first.accept()) {
    for (auto s : second.accept()) {
      auto idx = toIndex(f, s, first, false, false);
      toExplore.emplace(f, s);
      reachable[idx] = true;

      toExplore.emplace(f, s, true, false);
      reachable[idx + 1] = true;

      toExplore.emplace(f, s, false, true);
      reachable[idx + 2] = true;
    }
  }
  while (!toExplore.empty()) {
    auto curr = toExplore.front();
    toExplore.pop();
    if (!curr.followFirst && !curr.followSecond) {
      matcher->match(curr.first, curr.second, true);
      int i, j;
      // Iterate through arcs that end with the curr node - the first arc's olabel
      // is the same as the second arc's ilabel per the matcher
      while (matcher->hasNext()) {
        std::tie(i, j) = matcher->next(); // arcs ending with curr
        // Starting nodes for i and j arcs
        auto un1 = first.srcNode(i);
        auto un2 = second.srcNode(j);
        auto idx = toIndex(un1, un2, first, false, false);
        if (!reachable[idx]) {
          // If we haven't seen this state before, explore it.
          toExplore.emplace(un1, un2);
          reachable[idx] = true;
        }
        // For non epsilon matches, explore the epsilon-following states.
        if (first.olabel(i) != epsilon) {
          if (!reachable[idx + 1]) {
            toExplore.emplace(un1, un2, true, false);
            reachable[idx + 1] = true;
          }
          if (!reachable[idx + 2]) {
            toExplore.emplace(un1, un2, false, true);
            reachable[idx + 2] = true;
          }
        }
      }
    } else {
      // Check for reachable nodes via single epsilon transitions
      epsilonReachable(first, second, curr, reachable, toExplore);
    }
  }
  return reachable;
}

/* Add a node and arc to the new graph if it is reachable.
 * Returns if node is reachable. */
bool addReachableNodeAndArc(
    const Graph& first,
    const Graph& second,
    int currNode,
    const ExploreState& dst,
    float weight,
    int ilabel,
    int olabel,
    const std::vector<bool>& reachable,
    std::queue<ExploreState>& toExplore,
    std::vector<int>& newNodes,
    Graph& ngraph) {
  // Prospective new dest node in the composed graph. Ignore if we can't get to
  // an accept state.
  auto idx = toIndex(dst, first);
  if (reachable[idx]) {
    // Build the node - val of -1 --> uninitialized
    if (newNodes[idx] < 0) {
      newNodes[idx] = ngraph.addNode(
          false,
          first.isAccept(dst.first) && second.isAccept(dst.second));
      // Explore forward
      toExplore.push(dst);
    }
    auto newarc =
        ngraph.addArc(currNode, newNodes[idx], ilabel, olabel, weight);
  }
  return reachable[idx];
}

/*
 * Follow epsilon transitions in either the first or second graph.
 */
void addEpsilonReachableNodes(
    bool secondOrFirst,
    const Graph& first,
    const Graph& second,
    int currNode, // in the composed graph
    const ExploreState& state, // in the input graphs
    const std::vector<bool>& reachable,
    std::queue<ExploreState>& toExplore,
    std::vector<int>& newNodes,
    Graph& ngraph,
    std::vector<std::pair<int, int>>& gradInfo) {
  auto edges =
      secondOrFirst ? second.out(state.second) : first.out(state.first);
  // If epsilon is the output of an arc in the first graph's current node,
  // add an edge from the current node in the composed graph that takes epsilon
  // --> the first graph's olabel; if the second graph contains an input
  // epsilon, add an edge that takes the second graph's ilabel --> epsilon.
  // Traverse the epsilon edge in the graph and explore it forward
  // since the subgraph reachable from it is valid in the composed graph
  for (auto i : edges) {
    auto label = secondOrFirst ? second.ilabel(i) : first.olabel(i);
    auto isSorted =
        secondOrFirst ? second.ilabelSorted() : first.olabelSorted();
    if (label != epsilon) {
      if (isSorted) {
        // epsilon < 0 - can early-stop since we've reached a non-epsilon node
        // which will appear after epsilons in the sorted order
        break;
      } else {
        continue; // might find a future epsilon
      }
    }

    // Make the next state to explore
    auto nextState = ExploreState(
        secondOrFirst ? state.first : first.dstNode(i),
        secondOrFirst ? second.dstNode(i) : state.second,
        !secondOrFirst,
        secondOrFirst);

    // The destination node in the composed graph
    bool isReachable = addReachableNodeAndArc(
        first,
        second,
        currNode,
        nextState,
        secondOrFirst ? second.weight(i) : first.weight(i),
        secondOrFirst ? epsilon : first.ilabel(i),
        secondOrFirst ? second.olabel(i) : epsilon,
        reachable,
        toExplore,
        newNodes,
        ngraph);

    if (isReachable) {
      // Keep track of the edge in the composed graph for gradient computation
      if (secondOrFirst) {
        gradInfo.emplace_back(-1, i);
      } else {
        gradInfo.emplace_back(i, -1);
      }
    }
  }
}
} // namespace

void UnsortedMatcher::match(int lnode, int rnode, bool matchIn /* = false*/) {
  lv_ = matchIn ? g1_.in(lnode) : g1_.out(lnode);
  rv_ = matchIn ? g2_.in(rnode) : g2_.out(rnode);
  lIt_ = lv_.begin();
  rIt_ = rv_.begin();
}

bool UnsortedMatcher::hasNext() {
  for (; lIt_ != lv_.end(); ++lIt_) {
    for (; rIt_ != rv_.end(); ++rIt_) {
      if (g1_.olabel(*lIt_) == g2_.ilabel(*rIt_)) {
        return true;
      }
    }
    rIt_ = rv_.begin();
  }
  return false;
}

std::pair<int, int> UnsortedMatcher::next() {
  return std::make_pair(*lIt_, *rIt_++);
}

SinglySortedMatcher::SinglySortedMatcher(
    const Graph& g1,
    const Graph& g2,
    bool searchG1 /* = false */)
    : g1_(g1), g2_(g2), searchG1_(searchG1) {}

void SinglySortedMatcher::match(
    int lnode,
    int rnode,
    bool matchIn /* = false */) {
  searchV_ = matchIn ? g1_.in(lnode) : g1_.out(lnode);
  queryV_ = matchIn ? g2_.in(rnode) : g2_.out(rnode);
  if (!searchG1_) {
    // Swap based on the graph we are searching
    std::swap(searchV_, queryV_);
  }
  searchIt_ = searchV_.begin();
  queryIt_ = queryV_.begin();
}

bool SinglySortedMatcher::hasNext() {
  if (queryIt_ == queryV_.end()) {
    return false;
  }
  if (searchIt_ != searchV_.end()) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);
    auto sl = searchG1_ ? g1_.olabel(*searchIt_) : g2_.ilabel(*searchIt_);
    if (ql == sl) {
      return true;
    }
  }
  if (searchIt_ != searchV_.begin()) {
    // Not at the start of the search
    ++queryIt_;
  }

  // Update the query pointer and the start of the search range pointer
  for (; queryIt_ != queryV_.end(); ++queryIt_) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);
    // Set the comparison function appropriately
    auto comparisonFn = [this](int arc, int val) {
      return searchG1_ ? g1_.olabel(arc) < val : g2_.ilabel(arc) < val;
    };
    searchIt_ =
        std::lower_bound(searchV_.begin(), searchV_.end(), ql, comparisonFn);

    if (searchIt_ == searchV_.end()) {
      continue;
    }

    auto sl = searchG1_ ? g1_.olabel(*searchIt_) : g2_.ilabel(*searchIt_);
    if (sl == ql) {
      return true;
    }
  }
  return false;
}

std::pair<int, int> SinglySortedMatcher::next() {
  if (searchG1_) {
    return std::make_pair(*searchIt_++, *queryIt_);
  } else {
    return std::make_pair(*queryIt_, *searchIt_++);
  }
}

void DoublySortedMatcher::match(
    int lnode,
    int rnode,
    bool matchIn /* = false */) {
  searchV_ = matchIn ? g1_.in(lnode) : g1_.out(lnode);
  queryV_ = matchIn ? g2_.in(rnode) : g2_.out(rnode);

  searchG1_ = searchV_.size() > queryV_.size();
  if (!searchG1_) {
    // Swap based on the graph we are searching
    std::swap(searchV_, queryV_);
  }
  searchItBegin_ = searchIt_ = searchV_.begin();
  queryIt_ = queryV_.begin();
}

bool DoublySortedMatcher::hasNext() {
  if (queryIt_ == queryV_.end()) {
    return false;
  }
  if (searchIt_ != searchV_.end()) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);
    auto sl = searchG1_ ? g1_.olabel(*searchIt_) : g2_.ilabel(*searchIt_);
    if (ql == sl) {
      return true;
    }
  }
  if (searchIt_ != searchItBegin_) {
    // Not at the start of the search
    ++queryIt_;
  }

  // Update the query pointer and the start of the search range pointer
  for (; queryIt_ != queryV_.end(); ++queryIt_) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);

    // Set the comparison function appropriately
    auto comparisonFn = [this](int arc, int val) {
      return searchG1_ ? g1_.olabel(arc) < val : g2_.ilabel(arc) < val;
    };
    // Allowed because the query vector is sorted.
    searchItBegin_ =
        std::lower_bound(searchItBegin_, searchV_.end(), ql, comparisonFn);
    if (searchItBegin_ == searchV_.end()) {
      return false;
    }

    auto sl =
        searchG1_ ? g1_.olabel(*searchItBegin_) : g2_.ilabel(*searchItBegin_);
    if (sl == ql) {
      searchIt_ = searchItBegin_;
      return true;
    }
  }
  return false;
}

std::pair<int, int> DoublySortedMatcher::next() {
  if (searchG1_) {
    return std::make_pair(*searchIt_++, *queryIt_);
  } else {
    return std::make_pair(*queryIt_, *searchIt_++);
  }
}

// Composes two graphs and returns a new graph
Graph compose(
    const Graph& first,
    const Graph& second,
    std::shared_ptr<ArcMatcher> matcher) {
  // Compute reachable nodes from any accept state in the new graph
  auto reachable = findReachable(first, second, matcher);
  // Compose the graphs
  Graph ngraph(nullptr, {first, second});
  // Flat representation of nodes in both graphs for all three possible epsilon
  // matched states, index with toIndex.
  std::vector<int> newNodes(3 * first.numNodes() * second.numNodes(), -1);

  std::queue<ExploreState> toExplore;
  // Compile starting nodes that are reachable. If any pairs of reachable start
  // nodes in the input graph are also both accept nodes, make these accept
  // nodes in the composed graph.
  for (auto s1 : first.start()) {
    for (auto s2 : second.start()) {
      auto idx = toIndex(s1, s2, first, false, false);
      if (reachable[idx]) {
        newNodes[idx] =
            ngraph.addNode(true, first.isAccept(s1) && second.isAccept(s2));
        toExplore.emplace(s1, s2);
      }
    }
  }

  // The index of a particlar pair entry in gradInfo corresponds to an arc in
  // the composed graph - at gradient computation time, this facilitates
  // efficiently mapping an arc in the composed graph to the corresponding arcs
  // in the first and second graphs
  std::vector<std::pair<int, int>> gradInfo;
  // Explore the graph starting from the collection of start nodes
  while (!toExplore.empty()) {
    auto curr = toExplore.front();
    toExplore.pop();
    // A node in the composed graph
    auto currNode = newNodes[toIndex(curr, first)];
    int i, j;
    matcher->match(curr.first, curr.second);
    // Each pair of nodes in the initial graph may have multiple outgoing arcs
    // that should be combined in the composed graph
    while (matcher->hasNext()) {
      // The matcher invariant remains: arc i's olabel (from the first graph) is
      // arc j's ilabel (from the second graph)
      std::tie(i, j) = matcher->next();

      // If in a following first or second only state then can only follow
      // non-epsilon matches.
      if (first.olabel(i) == epsilon && (curr.followFirst || curr.followSecond)) {
        continue;
      }
      // Make the next state to explore
      auto next = ExploreState(first.dstNode(i), second.dstNode(j), false, false);
      bool isReachable = addReachableNodeAndArc(
          first,
          second,
          currNode,
          next,
          first.weight(i) + second.weight(j),
          first.ilabel(i),
          second.olabel(j),
          reachable,
          toExplore,
          newNodes,
          ngraph);

      if (isReachable) {
        // Arcs remember where they came from for easy gradient computation.
        gradInfo.emplace_back(i, j);
      }
    }
    if (!curr.followSecond) {
      addEpsilonReachableNodes(
          false,
          first,
          second,
          currNode, // in the composed graph
          curr, // in the input graphs
          reachable,
          toExplore,
          newNodes,
          ngraph,
          gradInfo);
    }
    if (!curr.followFirst) {
      addEpsilonReachableNodes(
          true,
          first,
          second,
          currNode, // in the composed graph
          curr, // in the input graphs
          reachable,
          toExplore,
          newNodes,
          ngraph,
          gradInfo);
    }
  }

  /*
   * Here we assume deltas is the output (e.g. ngraph) and we know where
   * each arc came from. This makes it possible to disambiguate two arcs in the
   * composed graph with the same label and the same src and destination nodes.
   */
  auto gradFunc = [gradInfo = std::move(gradInfo)](
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

  ngraph.setGradFunc(std::move(gradFunc));
  return ngraph;
}

} // namespace cpu
} // namespace gtn
