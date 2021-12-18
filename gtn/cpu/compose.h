/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <utility>
#include <vector>

#include "gtn/graph.h"

namespace gtn {
namespace cpu {

class ArcMatcher {
 public:
  virtual void match(int lnode, int rnode, bool matchIn = false) = 0;
  virtual bool hasNext() = 0;
  virtual std::pair<int, int> next() = 0;
};

class UnsortedMatcher : public ArcMatcher {
 public:
  UnsortedMatcher(const Graph& g1, const Graph& g2) : g1_(g1), g2_(g2){};

  /* Match the arcs on the left node `lnode` and the right node `rnode`. If
   * `matchIn = false` (default) then arcs will be matched by `olabel`
   * otherwise they will be matched by `ilabel`.
   */
  void match(int lnode, int rnode, bool matchIn /* = false*/) override;
  bool hasNext() override;
  std::pair<int, int> next() override;

 private:
  const Graph& g1_;
  const Graph& g2_;
  Graph::ArcPtr lv_;
  Graph::ArcPtr  rv_;
  const int* lIt_;
  const int* rItBegin_;
  const int* rIt_;
};

class SinglySortedMatcher : public ArcMatcher {
 public:
  SinglySortedMatcher(const Graph& g1, const Graph& g2, bool searchG1 = false);

  void match(int lnode, int rnode, bool matchIn /* = false */) override;

  bool hasNext() override;

  std::pair<int, int> next() override;

 private:
  const Graph& g1_;
  const Graph& g2_;
  bool searchG1_;
  Graph::ArcPtr searchV_;
  Graph::ArcPtr queryV_;
  const int* searchIt_;
  const int* queryIt_;
};

class DoublySortedMatcher : public ArcMatcher {
 public:
  DoublySortedMatcher(const Graph& g1, const Graph& g2) : g1_(g1), g2_(g2){};

  void match(int lnode, int rnode, bool matchIn /* = false */) override;

  bool hasNext() override;

  std::pair<int, int> next() override;

 private:
  const Graph& g1_;
  const Graph& g2_;
  bool searchG1_;
  Graph::ArcPtr searchV_;
  Graph::ArcPtr queryV_;
  const int* searchIt_;
  const int* searchItBegin_;
  const int* queryIt_;
};

/* Composes two transducers. */
Graph compose(
    const Graph& g1,
    const Graph& g2,
    std::shared_ptr<ArcMatcher> matcher);

} // namespace cpu
} // namespace gtn
