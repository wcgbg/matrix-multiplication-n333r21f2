#include "matrix/f2_matrix.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace matrix {
namespace {

// Exhaustive checks of the square-matrix operators for n ≤ 4 (n² ≤ 16 bits).
template <int n> void CheckSquareOps() {
  using Mat = F2Matrix<n, n>;
  const Mat id = Mat::Identity();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(id.Get(i, j), i == j ? 1 : 0);
    }
  }

  for (uint32_t d = 0; d < (uint32_t{1} << (n * n)); ++d) {
    const Mat m(static_cast<uint16_t>(d));

    // Transpose: involution + entrywise definition.
    EXPECT_EQ(m.Transposed().Transposed().Data(), m.Data());
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        EXPECT_EQ(m.Transposed().Get(j, i), m.Get(i, j));
      }
    }

    // Identity is a left/right unit.
    EXPECT_EQ((m * id).Data(), m.Data());
    EXPECT_EQ((id * m).Data(), m.Data());

    // Inverse: M·M⁻¹ = M⁻¹·M = I when invertible; the zero sentinel otherwise.
    const Mat mi = m.Inversed();
    if (m.IsInvertible()) {
      EXPECT_EQ((m * mi).Data(), id.Data());
      EXPECT_EQ((mi * m).Data(), id.Data());
    } else {
      EXPECT_EQ(mi.Data(), 0);
    }
  }
}

TEST(F2MatrixTest, SquareOps) {
  CheckSquareOps<1>();
  CheckSquareOps<2>();
  CheckSquareOps<3>();
  CheckSquareOps<4>();
}

// |GL(n, 𝔽₂)| via Rank()/IsInvertible() must match the known group orders.
template <int n> int CountInvertible() {
  int count = 0;
  for (uint32_t d = 0; d < (uint32_t{1} << (n * n)); ++d) {
    if (F2Matrix<n, n>(static_cast<uint16_t>(d)).IsInvertible()) {
      ++count;
    }
  }
  return count;
}

TEST(F2MatrixTest, GLSizes) {
  EXPECT_EQ(CountInvertible<1>(), 1);
  EXPECT_EQ(CountInvertible<2>(), 6);
  EXPECT_EQ(CountInvertible<3>(), 168);
  EXPECT_EQ(CountInvertible<4>(), 20160);
}

// The row-vector × matrix product the tables rely on: (1×n)·(n×n) = the XOR of
// the matrix rows selected by the row vector. Exhaustive for n ≤ 4.
template <int n> void CheckRowTimesMatrix() {
  for (unsigned v = 0; v < (1u << n); ++v) {
    const F2Matrix<1, n> row(static_cast<uint16_t>(v));
    for (uint32_t d = 0; d < (uint32_t{1} << (n * n)); ++d) {
      const F2Matrix<n, n> m(static_cast<uint16_t>(d));
      unsigned expect = 0;
      for (int j = 0; j < n; ++j) {
        uint8_t acc = 0;
        for (int k = 0; k < n; ++k) {
          acc ^= static_cast<uint8_t>(((v >> k) & 1u) & m.Get(k, j));
        }
        if (acc) {
          expect |= 1u << j;
        }
      }
      EXPECT_EQ((row * m).Data(), expect) << "v=" << v << " m=" << d;
    }
  }
}

TEST(F2MatrixTest, RowTimesMatrix) {
  CheckRowTimesMatrix<2>();
  CheckRowTimesMatrix<3>();
  CheckRowTimesMatrix<4>();
}

} // namespace
} // namespace matrix
