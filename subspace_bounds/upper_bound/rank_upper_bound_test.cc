#include "subspace_bounds/upper_bound/rank_upper_bound.h"

#include <cstddef>

#include "gtest/gtest.h"

#include "core/tensor.h"

namespace upper_bound {
namespace {

// The NxNxN matrix-multiplication tensor over F_2.
template <std::size_t N> Tensor<2, N * N, N * N, N * N> MatMulTensor() {
  Tensor<2, N * N, N * N, N * N> t = {};
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      for (std::size_t k = 0; k < N; ++k)
        t[i * N + j][j * N + k][k * N + i] = 1;
  return t;
}

TEST(RankUpperBoundTest, MatMul2x2FindsStrassen) {
  Tensor<2, 4, 4, 4> t = MatMulTensor<2>();

  UpperBoundOptions opts;
  UpperBoundResult<4, 4, 4> result = RankUpperBound(t, opts);

  EXPECT_EQ(result.rank, 7);
  EXPECT_FALSE(result.scheme.empty());
}

TEST(RankUpperBoundTest, MatMul3x3FindsRank23) {
  Tensor<2, 9, 9, 9> t = MatMulTensor<3>();

  UpperBoundOptions opts;
  opts.path_limit = 100'000;
  opts.max_steps_at_a_rank = 2'000;
  opts.num_paths = 10;

  UpperBoundResult<9, 9, 9> result = RankUpperBound(t, opts);

  EXPECT_EQ(result.rank, 23);
  EXPECT_FALSE(result.scheme.empty());
}

} // namespace
} // namespace upper_bound
