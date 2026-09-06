#pragma once

// Flattening lower bound.
//
// The "flattening" of a 3-tensor T along its A-axis is the NA × (NB·NC) matrix
// whose row i is the (flattened) bc-slice T[i]. Its rank over 𝔽_q is a lower
// bound on the tensor rank R(T), because a rank-r decomposition of T gives a
// rank-r factorisation of every flattening. We take the best of the three
// cyclic positions (flatten along A, B, then C).
//
// Split out of the matrix-mult code's rank_lower_bound_basic_technics.h: there
// the tensor axes had composite dimensions (n0·n1, n1·n2, n2·n0); here they are
// the plain problem dimensions (NA, NB, NC). CyclicTranspose rotates the three
// axes so the single FlattenTensorA covers all three positions.

#include <algorithm>
#include <cstddef>

#include "core/dynamic_matrix.h"
#include "core/tensor.h"

// Rotate the three tensor axes: result[j][k][i] = tensor[i][j][k].
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
Tensor<P, NB, NC, NA>
CyclicTranspose(const Tensor<P, NA, NB, NC> &tensor) {
  Tensor<P, NB, NC, NA> result = {};
  for (std::size_t i = 0; i < NA; ++i) {
    for (std::size_t j = 0; j < NB; ++j) {
      for (std::size_t k = 0; k < NC; ++k) {
        result[j][k][i] = tensor[i][j][k];
      }
    }
  }
  return result;
}

// Flatten along the A-axis into an NA × (NB·NC) matrix over 𝔽_q
// (q = P) with entry (i, j·NC + k) = tensor[i][j][k].
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
DynamicMatrix<P> FlattenTensorA(const Tensor<P, NA, NB, NC> &tensor) {
  DynamicMatrix<P> result(NA, NB * NC);
  for (std::size_t i = 0; i < NA; ++i) {
    for (std::size_t j = 0; j < NB; ++j) {
      for (std::size_t k = 0; k < NC; ++k) {
        result(static_cast<int>(i), static_cast<int>(j * NC + k)) =
            tensor[i][j][k];
      }
    }
  }
  return result;
}

// Largest flattening rank over the three cyclic positions. `target_rank` lets
// the caller short-circuit: as soon as one position already meets the target we
// return it without computing the others. Mirrors RankLowerBoundFlattenMatrix.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
int RankLowerBoundFlatten(const Tensor<P, NA, NB, NC> &tensor,
                          int target_rank = std::numeric_limits<int>::max()) {
  Tensor<P, NB, NC, NA> tensor1 = CyclicTranspose<P, NA, NB, NC>(tensor);
  int rank1 = FlattenTensorA<P, NB, NC, NA>(tensor1).Rank();
  if (rank1 >= target_rank) {
    return rank1;
  }

  Tensor<P, NC, NA, NB> tensor2 = CyclicTranspose<P, NB, NC, NA>(tensor1);
  int rank2 = FlattenTensorA<P, NC, NA, NB>(tensor2).Rank();
  if (rank2 >= target_rank) {
    return rank2;
  }

  int rank0 = FlattenTensorA<P, NA, NB, NC>(tensor).Rank();
  return std::max({rank0, rank1, rank2});
}
