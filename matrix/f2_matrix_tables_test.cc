#include "matrix/f2_matrix_tables.h"

#include <cstdint>
#include <set>
#include <vector>

#include "gtest/gtest.h"

#include "matrix/f2_matrix.h"

// Exhaustive checks of every table F2MatrixTables exposes, using the
// independently-tested F2Matrix operators (operator*, Inversed, Transposed) as
// the reference. The shapes deliberately span N0<N1, N0>N1, N0=N1 and the
// degenerate row/column shapes, because the table indexing differs per
// orientation. Non-square Mult011/Mult001 cases check operand shapes as well
// as multiplication order; square cases alone cannot detect shape mismatches.

namespace matrix {
namespace {

// Verifies one square side: the GL vector must be identity-first, hold every
// invertible matrix exactly once and nothing else, and match the known group
// order; the inverse table must agree with F2Matrix::Inversed() on every matrix
// (including the 0 sentinel for singular inputs) and actually invert it.
template <int n, typename InverseFn>
void CheckSquareSide(const std::vector<F2Matrix<n, n>> &gl, InverseFn inverse,
                     int expected_order) {
  ASSERT_FALSE(gl.empty());
  EXPECT_EQ(gl.front().Data(), (F2Matrix<n, n>::Identity().Data()))
      << "identity must be listed first";
  EXPECT_EQ(static_cast<int>(gl.size()), expected_order);

  std::set<uint16_t> present;
  for (const F2Matrix<n, n> &m : gl) {
    EXPECT_TRUE(m.IsInvertible()) << "GL holds a singular matrix: " << m.Data();
    EXPECT_TRUE(present.insert(m.Data()).second)
        << "GL holds a duplicate: " << m.Data();
  }

  const uint16_t id = F2Matrix<n, n>::Identity().Data();
  int invertible = 0;
  for (uint32_t d = 0; d < (uint32_t{1} << (n * n)); ++d) {
    const F2Matrix<n, n> m(static_cast<uint16_t>(d));
    EXPECT_EQ(inverse(m).Data(), m.Inversed().Data())
        << "inverse table disagrees with reference at d=" << d;
    if (m.IsInvertible()) {
      ++invertible;
      EXPECT_TRUE(present.count(m.Data()) != 0)
          << "GL is missing invertible matrix d=" << d;
      EXPECT_EQ((m * inverse(m)).Data(), id) << "inverse is wrong at d=" << d;
    }
  }
  EXPECT_EQ(invertible, expected_order) << "GL coverage count";
}

template <int N0, int N1> void CheckGroupTables(int order0, int order1) {
  const F2MatrixTables<N0, N1> tables;
  EXPECT_EQ(tables.Identity0().Data(), (F2Matrix<N0, N0>::Identity().Data()));
  EXPECT_EQ(tables.Identity1().Data(), (F2Matrix<N1, N1>::Identity().Data()));
  CheckSquareSide<N0>(
      tables.Gl0(), [&](F2Matrix<N0, N0> m) { return tables.Inverse0(m); },
      order0);
  CheckSquareSide<N1>(
      tables.Gl1(), [&](F2Matrix<N1, N1> m) { return tables.Inverse1(m); },
      order1);
}

// tables.Mult011(m01, m11) == m01·m11 for every (N0×N1, N1×N1) pair. Stops at
// the first mismatch so a systematic error reports one clear example rather
// than flooding millions of failures.
template <int N0, int N1> void CheckMult011() {
  const F2MatrixTables<N0, N1> tables;
  for (uint32_t b = 0; b < (uint32_t{1} << (N1 * N1)); ++b) {
    const F2Matrix<N1, N1> m11(static_cast<uint16_t>(b));
    for (uint32_t a = 0; a < (uint32_t{1} << (N0 * N1)); ++a) {
      const F2Matrix<N0, N1> m01(static_cast<uint16_t>(a));
      const uint16_t got = tables.Mult011(m01, m11).Data();
      const uint16_t want = (m01 * m11).Data();
      if (got != want) {
        ADD_FAILURE() << "Mult011<" << N0 << "," << N1 << "> m01=" << a
                      << " m11=" << b << " got=" << got << " want=" << want;
        return;
      }
    }
  }
}

// tables.Mult001(m00, m01) == m00·m01 for every (N0×N0, N0×N1) pair.
template <int N0, int N1> void CheckMult001() {
  const F2MatrixTables<N0, N1> tables;
  for (uint32_t b = 0; b < (uint32_t{1} << (N0 * N0)); ++b) {
    const F2Matrix<N0, N0> m00(static_cast<uint16_t>(b));
    for (uint32_t a = 0; a < (uint32_t{1} << (N0 * N1)); ++a) {
      const F2Matrix<N0, N1> m01(static_cast<uint16_t>(a));
      const uint16_t got = tables.Mult001(m00, m01).Data();
      const uint16_t want = (m00 * m01).Data();
      if (got != want) {
        ADD_FAILURE() << "Mult001<" << N0 << "," << N1 << "> m00=" << b
                      << " m01=" << a << " got=" << got << " want=" << want;
        return;
      }
    }
  }
}

template <int N0, int N1> void CheckTranspose() {
  const F2MatrixTables<N0, N1> tables;
  for (uint32_t a = 0; a < (uint32_t{1} << (N0 * N1)); ++a) {
    const F2Matrix<N0, N1> m01(static_cast<uint16_t>(a));
    EXPECT_EQ(tables.TransposeConstraint(m01).Data(), m01.Transposed().Data())
        << "transpose mismatch at " << a;
  }
}

TEST(F2MatrixTablesTest, GroupAndInverseTables) {
  CheckGroupTables<1, 1>(1, 1);
  CheckGroupTables<2, 2>(6, 6);
  CheckGroupTables<3, 3>(168, 168);
  CheckGroupTables<4, 4>(20160, 20160);
  CheckGroupTables<2, 3>(6, 168);
  CheckGroupTables<3, 2>(168, 6);
  CheckGroupTables<2, 4>(6, 20160);
  CheckGroupTables<4, 2>(20160, 6);
  CheckGroupTables<3, 4>(168, 20160);
  CheckGroupTables<4, 3>(20160, 168);
}

TEST(F2MatrixTablesTest, Mult011MatchesReferenceProduct) {
  CheckMult011<1, 1>();
  CheckMult011<2, 2>();
  CheckMult011<3, 3>();
  CheckMult011<2, 3>(); // N0 < N1
  CheckMult011<3, 2>(); // N0 > N1
  CheckMult011<4, 2>(); // N0 > N1
  CheckMult011<4, 3>(); // N0 > N1, largest tractable shape
  CheckMult011<1, 4>(); // degenerate row shape
}

TEST(F2MatrixTablesTest, Mult001MatchesReferenceProduct) {
  CheckMult001<1, 1>();
  CheckMult001<2, 2>();
  CheckMult001<3, 3>();
  CheckMult001<2, 3>(); // N0 < N1
  CheckMult001<3, 2>(); // N0 > N1
  CheckMult001<2, 4>(); // N0 < N1
  CheckMult001<3, 4>(); // N0 < N1, largest tractable shape
  CheckMult001<4, 1>(); // degenerate column shape
}

TEST(F2MatrixTablesTest, TransposeConstraintMatchesReference) {
  CheckTranspose<2, 2>();
  CheckTranspose<2, 3>();
  CheckTranspose<3, 2>();
  CheckTranspose<3, 3>();
  CheckTranspose<2, 4>();
  CheckTranspose<4, 2>();
  CheckTranspose<3, 4>();
  CheckTranspose<4, 3>();
  CheckTranspose<4, 4>();
}

} // namespace
} // namespace matrix
