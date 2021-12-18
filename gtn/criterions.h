/*
 * Copyright (c) Meta, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "graph.h"

namespace gtn {
namespace criterion {

/** \addtogroup criterions
 *  @{
 */

/**
 * An implementation of Connectionist Temporal Classification (CTC)
 * in the GTN framework. See e.g.
 * https://www.cs.toronto.edu/~graves/icml_2006.pdf
 * Computes the CTC Loss of emission graph and return the output loss as a graph
 * NB: This assumes the weights on the emission graph are in log probabilities.
 * This is typically done by adding a LogSoftmax layer in the last layer of the
 * network.
 */
Graph ctcLoss(
    const Graph& logProbs,
    const std::vector<int>& target,
    const int blankIdx);

} // namespace criterion
} // namespace gtn