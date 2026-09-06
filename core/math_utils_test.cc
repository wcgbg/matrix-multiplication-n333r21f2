#include "core/math_utils.h"

#include <vector>

#include "gtest/gtest.h"

#include "core/gf_vec.h"

namespace {

TEST(GaussJordanEliminationF2Test, EmptyMatrix) {
  std::vector<GFVec<2, 4>> matrix;
  EXPECT_EQ(GaussJordanEliminationF2<4>(&matrix), 0);
}

TEST(GaussJordanEliminationF2Test, ZeroMatrix) {
  std::vector<GFVec<2, 4>> matrix(3);
  EXPECT_EQ(GaussJordanEliminationF2<4>(&matrix), 0);
  EXPECT_EQ(matrix[0].data, 0);
  EXPECT_EQ(matrix[1].data, 0);
  EXPECT_EQ(matrix[2].data, 0);
}

TEST(GaussJordanEliminationF2Test, IdentityMatrix) {
  // Identity matrix 3x3: rows are [100, 010, 001] in binary
  // bit 0 = col 0, bit 1 = col 1, bit 2 = col 2
  std::vector<GFVec<2, 3>> matrix = {
      {0b001}, // row 0: [1, 0, 0]
      {0b010}, // row 1: [0, 1, 0]
      {0b100}, // row 2: [0, 0, 1]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 3);
  // Identity should remain identity
  EXPECT_EQ(matrix[0].data, 0b001);
  EXPECT_EQ(matrix[1].data, 0b010);
  EXPECT_EQ(matrix[2].data, 0b100);
}

TEST(GaussJordanEliminationF2Test, FullRankMatrix) {
  // Matrix:
  // 110
  // 101
  // 010
  std::vector<GFVec<2, 3>> matrix = {
      {0b011}, // row 0: [1, 1, 0]
      {0b101}, // row 1: [1, 0, 1]
      {0b010}, // row 2: [0, 1, 0]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 3);
  // After RREF, should become identity matrix
  EXPECT_EQ(matrix[0].data, 0b001);
  EXPECT_EQ(matrix[1].data, 0b010);
  EXPECT_EQ(matrix[2].data, 0b100);
}

TEST(GaussJordanEliminationF2Test, Rank2Matrix) {
  // Matrix:
  // 110
  // 101
  // 011
  std::vector<GFVec<2, 3>> matrix = {
      {0b011}, // row 0: [1, 1, 0]
      {0b101}, // row 1: [1, 0, 1]
      {0b110}, // row 2: [0, 1, 1]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 2);
  EXPECT_EQ(matrix[0].data, 0);
  EXPECT_EQ(matrix[1].data, 0b011);
  EXPECT_EQ(matrix[2].data, 0b101);
}

TEST(GaussJordanEliminationF2Test, Rank1Matrix) {
  // Matrix:
  // 111
  // 111
  // 111
  // All rows identical, rank should be 1
  std::vector<GFVec<2, 3>> matrix = {
      {0b111}, // row 0: [1, 1, 1]
      {0b111}, // row 1: [1, 1, 1]
      {0b111}, // row 2: [1, 1, 1]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 1);
  EXPECT_EQ(matrix[0].data, 0);
  EXPECT_EQ(matrix[1].data, 0);
  EXPECT_EQ(matrix[2].data, 0b111);
}

TEST(GaussJordanEliminationF2Test, SingleRow) {
  std::vector<GFVec<2, 3>> matrix = {{0b011}};
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 1);
  EXPECT_EQ(matrix[0].data, 0b011);
}

TEST(GaussJordanEliminationF2Test, SingleRowZero) {
  std::vector<GFVec<2, 3>> matrix = {{0b000}};
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 0);
  EXPECT_EQ(matrix[0].data, 0);
}

TEST(GaussJordanEliminationF2Test, MoreRowsThanColumns) {
  // Matrix 4x3:
  // 110
  // 010
  // 001
  // 111
  std::vector<GFVec<2, 3>> matrix = {
      {0b011}, // row 0: [1, 1, 0]
      {0b010}, // row 1: [0, 1, 0]
      {0b100}, // row 2: [0, 0, 1]
      {0b111}, // row 3: [1, 1, 1]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 3);
  EXPECT_EQ(matrix[0].data, 0);
  EXPECT_EQ(matrix[1].data, 0b001);
  EXPECT_EQ(matrix[2].data, 0b010);
  EXPECT_EQ(matrix[3].data, 0b100);
}

TEST(GaussJordanEliminationF2Test, MoreColumnsThanRows) {
  // Matrix 2x4:
  // 0110
  // 1010
  std::vector<GFVec<2, 4>> matrix = {
      {0b0110}, // row 0: [0, 1, 1, 0]
      {0b0101}, // row 1: [1, 0, 1, 0]
  };
  EXPECT_EQ(GaussJordanEliminationF2<4>(&matrix), 2);
  EXPECT_EQ(matrix[0].data, 0b0011);
  EXPECT_EQ(matrix[1].data, 0b0101);
}

TEST(GaussJordanEliminationF2Test, PermutedIdentity) {
  // Matrix with rows swapped:
  // 010
  // 001
  // 100
  std::vector<GFVec<2, 3>> matrix = {
      {0b010}, // row 0: [0, 1, 0]
      {0b100}, // row 1: [0, 0, 1]
      {0b001}, // row 2: [1, 0, 0]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 3);
  // Should be reordered to identity-like form
  EXPECT_EQ(matrix[0].data, 0b001);
  EXPECT_EQ(matrix[1].data, 0b010);
  EXPECT_EQ(matrix[2].data, 0b100);
}

TEST(GaussJordanEliminationF2Test, UpperTriangular) {
  // Matrix:
  // 111
  // 011
  // 001
  std::vector<GFVec<2, 3>> matrix = {
      {0b111}, // row 0: [1, 1, 1]
      {0b110}, // row 1: [0, 1, 1]
      {0b100}, // row 2: [0, 0, 1]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 3);
  // Should become identity
  EXPECT_EQ(matrix[0].data, 0b001);
  EXPECT_EQ(matrix[1].data, 0b010);
  EXPECT_EQ(matrix[2].data, 0b100);
}

TEST(GaussJordanEliminationF2Test, LowerTriangular) {
  // Matrix:
  // 100
  // 110
  // 111
  std::vector<GFVec<2, 3>> matrix = {
      {0b001}, // row 0: [1, 0, 0]
      {0b011}, // row 1: [1, 1, 0]
      {0b111}, // row 2: [1, 1, 1]
  };
  EXPECT_EQ(GaussJordanEliminationF2<3>(&matrix), 3);
  // Should become identity
  EXPECT_EQ(matrix[0].data, 0b001);
  EXPECT_EQ(matrix[1].data, 0b010);
  EXPECT_EQ(matrix[2].data, 0b100);
}

TEST(GaussJordanEliminationF2Test, BitwidthFour) {
  std::vector<GFVec<2, 4>> matrix = {
      {0b0011}, // row 0: [1, 1, 0, 0]
      {0b0101}, // row 1: [1, 0, 1, 0]
      {0b1100}, // row 2: [0, 0, 1, 1]
  };
  EXPECT_EQ(GaussJordanEliminationF2<4>(&matrix), 3);
  EXPECT_EQ(matrix[0].data, 0b0011);
  EXPECT_EQ(matrix[1].data, 0b0101);
  EXPECT_EQ(matrix[2].data, 0b1001);
}

TEST(GaussJordanEliminationF2Test, LargeBitwidth) {
  std::vector<GFVec<2, 16>> matrix = {
      {0b1111111100000000},
      {0b0000000011111111},
      {0b1010101010101010},
  };
  EXPECT_EQ(GaussJordanEliminationF2<16>(&matrix), 3);
  EXPECT_EQ(matrix[0].data, 0b0000000011111111);
  EXPECT_EQ(matrix[1].data, 0b0101010101010101);
  EXPECT_EQ(matrix[2].data, 0b1010101001010101);
}

TEST(GaussJordanEliminationFqTest, RankZeroOnEmptyMatrix) {
  std::vector<GFVec<3, 3>> rows;
  EXPECT_EQ((GaussJordanEliminationFq<3, 3>(&rows)), 0);
}

TEST(GaussJordanEliminationFqTest, RankZeroOnZeroRows) {
  std::vector<GFVec<3, 3>> rows = {GFVec<3, 3>{{0, 0, 0}}};
  EXPECT_EQ((GaussJordanEliminationFq<3, 3>(&rows)), 0);
  // Zero row is parked at the front (reversed-row convention).
  EXPECT_TRUE(rows[0].IsZero());
}

TEST(GaussJordanEliminationFqTest, IdentityHasFullRank) {
  // For NA=3: three pivot rows. Column order reversed means pivot order is
  // bit 2, bit 1, bit 0 from current_row decreasing.
  std::vector<GFVec<3, 3>> rows = {
      GFVec<3, 3>{{1, 0, 0}},
      GFVec<3, 3>{{0, 1, 0}},
      GFVec<3, 3>{{0, 0, 1}},
  };
  EXPECT_EQ((GaussJordanEliminationFq<3, 3>(&rows)), 3);
}

TEST(GaussJordanEliminationFqTest, DependentRowGetsZeroed) {
  // Row 1 = 2 * Row 0 mod 3. Rank should be 1, zero row parked at the front.
  std::vector<GFVec<3, 3>> rows = {
      GFVec<3, 3>{{1, 2, 0}},
      GFVec<3, 3>{{2, 1, 0}}, // = 2 * row 0 mod 3
  };
  EXPECT_EQ((GaussJordanEliminationFq<3, 3>(&rows)), 1);
  EXPECT_TRUE(rows[0].IsZero());
  EXPECT_FALSE(rows[1].IsZero());
}

TEST(GaussJordanEliminationFqTest, NormalizesPivotToOne) {
  // Single row with pivot value 2 (not 1). RREF normalizes it to 1.
  std::vector<GFVec<3, 3>> rows = {GFVec<3, 3>{{2, 1, 0}}};
  EXPECT_EQ((GaussJordanEliminationFq<3, 3>(&rows)), 1);
  // Highest-order pivot: index 1, with value 2 originally → normalized to 1.
  // Inverse of 2 mod 3 is 2; 2*2=1, 2*1=2 → row becomes {1, 2, 0} but column-
  // reversed RREF picks the highest-order nonzero entry as the pivot. Here
  // index 1 is the pivot (value 1 after normalization); index 0 must be the
  // scaled free coordinate.
  EXPECT_EQ(rows[0][1], 1);
  EXPECT_EQ(rows[0][0], 2);
}

TEST(GaussJordanEliminationFqTest, EliminatesAbovePivot) {
  // Two rows. After RREF, the pivots are at the highest set indices, and
  // entries above pivots are zero.
  std::vector<GFVec<3, 3>> rows = {
      GFVec<3, 3>{{1, 0, 1}}, // pivot at index 2
      GFVec<3, 3>{{0, 1, 1}}, // pivot at index 2 too — dependent on row 0
  };
  EXPECT_EQ((GaussJordanEliminationFq<3, 3>(&rows)), 2);
  // The second pivot must be at index 1, with the index-2 entry eliminated.
  // Specifically, row 1 originally had {0, 1, 1}, row 0 had {1, 0, 1}; pivot
  // at index 2 picks row 1 first (column-reversed iteration from
  // current_row = 1). After elimination row 0's index-2 entry becomes 0.
  // One row ends with leading 1 at index 2, the other at index 1.
  int pivot_at_2 = -1;
  int pivot_at_1 = -1;
  for (int i = 0; i < 2; ++i) {
    if (rows[i].LeadingNonzeroIdx() == 2) {
      pivot_at_2 = i;
    } else if (rows[i].LeadingNonzeroIdx() == 1) {
      pivot_at_1 = i;
    }
  }
  ASSERT_GE(pivot_at_2, 0);
  ASSERT_GE(pivot_at_1, 0);
  EXPECT_EQ(rows[pivot_at_2][2], 1);
  EXPECT_EQ(rows[pivot_at_1][1], 1);
  EXPECT_EQ(rows[pivot_at_1][2], 0); // above-pivot entry eliminated
}

TEST(IsLinearIndependentFqTest, IndependentTrue) {
  std::vector<GFVec<3, 3>> rows = {
      GFVec<3, 3>{{1, 0, 0}},
      GFVec<3, 3>{{0, 1, 0}},
  };
  EXPECT_TRUE((IsLinearIndependentFq<3, 3>(rows)));
}

TEST(IsLinearIndependentFqTest, DependentFalse) {
  std::vector<GFVec<3, 3>> rows = {
      GFVec<3, 3>{{1, 2, 0}},
      GFVec<3, 3>{{2, 1, 0}}, // = 2 * row 0 mod 3
  };
  EXPECT_FALSE((IsLinearIndependentFq<3, 3>(rows)));
}

} // namespace
