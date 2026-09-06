#include "core/constraints.h"

#include <cstdint>

#include "gtest/gtest.h"

#include "core/gf_vec.h"
#include "core/tensor.h"
#include "core/tensor_utils.h"

namespace {

// Step 1 plumbed F_2; Step 3 adds F_3 coverage. The F_2 expectations below
// are the regression baseline; the F_3 expectations exercise the new path.
constexpr int kP2 = 2;
constexpr int kP3 = 3;

// Helper: F₂ GFVec row of width N from an integer bit pattern.
template <int N> GFVec<kP2, N> F2Row(uint64_t bits) {
  return GFVec<kP2, N>{static_cast<BitVec<N>>(bits)};
}

TEST(ConstraintToStringTest, SingleAndMultipleTerms) {
  // Most significant bit first; coordinate i lands at position N-1-i.
  EXPECT_EQ((ConstraintToString<kP2, 6>(F2Row<6>(0b000001))), "000001");
  EXPECT_EQ((ConstraintToString<kP2, 6>(F2Row<6>(0b001001))), "001001");
  EXPECT_EQ((ConstraintToString<kP2, 6>(F2Row<6>(0b111111))), "111111");
}

TEST(ConstraintToStringTest, ZeroFunctional) {
  EXPECT_EQ((ConstraintToString<kP2, 6>(F2Row<6>(0))), "000000");
}

TEST(ConstraintToStringTest, WiderBitVec) {
  // N = 12 uses a uint16_t BitVec; high coordinate still prints.
  EXPECT_EQ((ConstraintToString<kP2, 12>(F2Row<12>(0b100000000001))),
            "100000000001");
}

TEST(ConstraintsBytesTest, EmptyRoundTrip) {
  const Constraints<kP2, 3> empty;
  EXPECT_EQ((ConstraintsToBytes<kP2, 3>(empty)), "");
  EXPECT_EQ((ConstraintsFromBytes<kP2, 3>("")), empty);
}

TEST(ConstraintsToStringTest, EmptyAndCommaJoined) {
  EXPECT_EQ((ConstraintsToString<kP2, 6>(Constraints<kP2, 6>{})),
            "EMPTY");
  EXPECT_EQ((ConstraintsToString<kP2, 6>(
                Constraints<kP2, 6>{F2Row<6>(0b000001)})),
            "000001");
  EXPECT_EQ((ConstraintsToString<kP2, 6>(Constraints<kP2, 6>{
                F2Row<6>(0b000001), F2Row<6>(0b000011)})),
            "000001,000011");
}

TEST(ApplyConstraintsToTensorTest, EmptyLeavesTensorUnchanged) {
  const auto tensor =
      SparseStringToTensor<kP2, 2, 2, 2>("a0*b0*c0 + a1*b1*c1 + a1*b0*c1");
  EXPECT_EQ((ApplyConstraintsToTensor<kP2, 2, 2, 2>(Constraints<kP2, 2>{},
                                                       tensor)),
            tensor);
}

TEST(ApplyConstraintsToTensorTest, SingleCoordinatePivotZeroesSlice) {
  // Constraint a0 = 0 (row 0b01, pivot bit 0, no free coordinates): the whole
  // A-slice i=0 is zeroed; the i=1 slice is untouched.
  const auto tensor = SparseStringToTensor<kP2, 2, 2, 2>(
      "a0*b0*c0 + a0*b1*c1 + a1*b0*c1 + a1*b1*c0");
  const Constraints<kP2, 2> constraints = {F2Row<2>(0b01)};
  EXPECT_EQ(TensorToSparseString(
                ApplyConstraintsToTensor<kP2, 2, 2, 2>(constraints, tensor)),
            "a1*b0*c1 + a1*b1*c0");
}

TEST(ApplyConstraintsToTensorTest, SubstitutionFoldsPivotSlice) {
  // Constraint a0 + a1 = 0 (row 0b11): pivot is the MSB (a1), free coordinate
  // is a0, so a1 := a0. The a1 slice folds into the a0 slice (XOR), then a1 is
  // zeroed. Here a0*b0*c0 and a1*b0*c0 cancel.
  const auto tensor =
      SparseStringToTensor<kP2, 2, 2, 2>("a0*b0*c0 + a1*b0*c0 + a1*b1*c1");
  const Constraints<kP2, 2> constraints = {F2Row<2>(0b11)};
  EXPECT_EQ(TensorToSparseString(
                ApplyConstraintsToTensor<kP2, 2, 2, 2>(constraints, tensor)),
            "a0*b1*c1");
}

TEST(ApplyConstraintsToTensorTest, RectangularDimensions) {
  // NA=3, NB=2, NC=2. Constraint a0 + a2 = 0 (row 0b101): pivot a2, free a0,
  // so the a2 slice folds into the a0 slice and a2 is zeroed.
  const auto tensor =
      SparseStringToTensor<kP2, 3, 2, 2>("a2*b0*c0 + a2*b1*c1 + a0*b0*c1");
  const Constraints<kP2, 3> constraints = {F2Row<3>(0b101)};
  EXPECT_EQ(TensorToSparseString(
                ApplyConstraintsToTensor<kP2, 3, 2, 2>(constraints, tensor)),
            "a0*b0*c0 + a0*b0*c1 + a0*b1*c1");
}

TEST(ApplyConstraintsToTensorTest, MultipleConstraints) {
  // NA=3 in RREF: rows {0b110, 0b001} encode a1 + a2 = 0 (pivot a2, free a1)
  // and a0 = 0 (pivot a0). So a2 folds into a1, and the a0 slice is zeroed.
  const auto tensor =
      SparseStringToTensor<kP2, 3, 2, 2>("a2*b0*c0 + a1*b1*c1 + a0*b0*c1");
  const Constraints<kP2, 3> constraints = {F2Row<3>(0b110), F2Row<3>(0b001)};
  EXPECT_EQ(TensorToSparseString(
                ApplyConstraintsToTensor<kP2, 3, 2, 2>(constraints, tensor)),
            "a1*b0*c0 + a1*b1*c1");
}

// --- F_3 coverage --------------------------------------------------------

TEST(ConstraintToStringF3Test, DigitsBigEndian) {
  // F_3 row [c0=2, c1=0, c2=1] prints highest-coordinate first: "102".
  GFVec<kP3, 3> v{{2, 0, 1}};
  EXPECT_EQ((ConstraintToString<kP3, 3>(v)), "102");
}

TEST(ConstraintsBytesF3Test, RoundTrip) {
  const Constraints<kP3, 3> r = {GFVec<kP3, 3>{{1, 0, 2}},
                                    GFVec<kP3, 3>{{0, 1, 2}}};
  const std::string bytes = ConstraintsToBytes<kP3, 3>(r);
  // sizeof(FpVec<3, 3>) == 3 → two rows × 3 bytes.
  EXPECT_EQ(bytes.size(), 6u);
  EXPECT_EQ((ConstraintsFromBytes<kP3, 3>(bytes)), r);
}

TEST(ApplyConstraintsToTensorF3Test, ZeroesPivotSlice) {
  // NA=2, constraint a0 = 0 (row [1, 0]): pivot at index 0, no free
  // coordinates; the i=0 slice is wiped out entirely.
  const auto tensor =
      SparseStringToTensor<kP3, 2, 2, 2>("a0*b0*c0 + 2*a0*b1*c1 + a1*b0*c1");
  const Constraints<kP3, 2> constraints = {GFVec<kP3, 2>{{1, 0}}};
  EXPECT_EQ(TensorToSparseString(
                ApplyConstraintsToTensor<kP3, 2, 2, 2>(constraints, tensor)),
            "a1*b0*c1");
}

TEST(ApplyConstraintsToTensorF3Test, NonOneCoefficientSubstitution) {
  // NA=2, constraint 2*a0 + a1 = 0 (canonical RREF rescales to row [2, 1]
  // with pivot at index 1, coefficient 1). The substitution reads a1 = -2*a0
  // = a0 (mod 3), so the a1 slice folds into the a0 slice (coefficient 1×
  // each).
  //
  // T = a0*b0*c0 + a1*b0*c0 + 2*a1*b1*c1
  // After substitution (a1 → -2·a0 = 1·a0 mod 3):
  //   result[a0] = T[a0] + Sub(0, Mul(2, T[a1])) where row[0]=2
  //              = a0*b0*c0 - 2*a1's slice
  //              row's free coord is i=0 with row[0]=2; pivot is i=1.
  //   result[0] = T[0] - 2·T[1]  (per j,k)
  //   T[0]: b0*c0
  //   T[1]: b0*c0 (coeff 1), b1*c1 (coeff 2)
  //   result[0]: (1, 0, 0, 0)[b0c0] - 2*(1, 0, 0, 0) = 1 - 2 = 2 → 2*a0*b0*c0
  //              (0, 0, 0, 0)[b1c1] - 2*(0, 0, 0, 2) = -4 = 2 → 2*a0*b1*c1
  //   result[1] = 0
  const auto tensor =
      SparseStringToTensor<kP3, 2, 2, 2>("a0*b0*c0 + a1*b0*c0 + 2*a1*b1*c1");
  const Constraints<kP3, 2> constraints = {GFVec<kP3, 2>{{2, 1}}};
  EXPECT_EQ(TensorToSparseString(
                ApplyConstraintsToTensor<kP3, 2, 2, 2>(constraints, tensor)),
            "2*a0*b0*c0 + 2*a0*b1*c1");
}

TEST(ApplyConstraintsToTensorF3Test, EmptyLeavesTensorUnchanged) {
  const auto tensor =
      SparseStringToTensor<kP3, 2, 2, 2>("2*a0*b0*c0 + a1*b1*c1");
  EXPECT_EQ((ApplyConstraintsToTensor<kP3, 2, 2, 2>(Constraints<kP3, 2>{},
                                                       tensor)),
            tensor);
}

} // namespace
