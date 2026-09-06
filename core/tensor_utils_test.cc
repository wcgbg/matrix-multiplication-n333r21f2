#include "tensor_utils.h"

#include "gtest/gtest.h"

// Convenience: all tests are over F_2 (P = 2).
template <std::size_t NA, std::size_t NB, std::size_t NC>
using TensorF2 = Tensor<2, NA, NB, NC>;

TEST(TensorToSparseStringTest, AllZeros) {
  TensorF2<2, 2, 2> tensor{};
  EXPECT_EQ((TensorToSparseString<2, 2, 2, 2>(tensor)), "0");
}

TEST(TensorToSparseStringTest, SingleTermF2) {
  TensorF2<2, 2, 2> tensor{};
  tensor[0][1][1] = 1;
  EXPECT_EQ((TensorToSparseString<2, 2, 2, 2>(tensor)), "a0*b1*c1");
}

TEST(TensorToSparseStringTest, MultipleTermsF2) {
  TensorF2<2, 2, 4> tensor{};
  tensor[0][1][3] = 1;
  tensor[1][0][2] = 1;
  EXPECT_EQ((TensorToSparseString<2, 2, 2, 4>(tensor)),
            "a0*b1*c3 + a1*b0*c2");
}

TEST(TensorToSparseStringTest, CoefficientNotOne) {
  TensorF2<2, 2, 4> tensor{};
  tensor[0][1][3] = 2;
  EXPECT_EQ((TensorToSparseString<2, 2, 2, 4>(tensor)), "2*a0*b1*c3");
}

TEST(TensorToSparseStringTest, MixedCoefficients) {
  TensorF2<2, 2, 3> tensor{};
  tensor[0][0][0] = 1;
  tensor[1][1][2] = 3;
  EXPECT_EQ((TensorToSparseString<2, 2, 2, 3>(tensor)),
            "a0*b0*c0 + 3*a1*b1*c2");
}

TEST(TensorToSparseStringTest, OrderingIsIJK) {
  TensorF2<2, 2, 2> tensor{};
  tensor[1][0][0] = 1;
  tensor[0][1][0] = 1;
  tensor[0][0][1] = 1;
  EXPECT_EQ((TensorToSparseString<2, 2, 2, 2>(tensor)),
            "a0*b0*c1 + a0*b1*c0 + a1*b0*c0");
}

TEST(SparseStringToTensorTest, Zero) {
  TensorF2<2, 2, 2> expected{};
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 2>("0")), expected);
}

TEST(SparseStringToTensorTest, Empty) {
  TensorF2<2, 2, 2> expected{};
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 2>("")), expected);
}

TEST(SparseStringToTensorTest, SingleTermF2) {
  TensorF2<2, 2, 2> expected{};
  expected[0][1][1] = 1;
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 2>("a0*b1*c1")), expected);
}

TEST(SparseStringToTensorTest, MultipleTermsF2) {
  TensorF2<2, 2, 4> expected{};
  expected[0][1][3] = 1;
  expected[1][0][2] = 1;
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 4>("a0*b1*c3 + a1*b0*c2")),
            expected);
}

TEST(SparseStringToTensorTest, CoefficientNotOne) {
  TensorF2<2, 2, 4> expected{};
  expected[0][1][3] = 2;
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 4>("2*a0*b1*c3")), expected);
}

TEST(SparseStringToTensorTest, MixedCoefficients) {
  TensorF2<2, 2, 3> expected{};
  expected[0][0][0] = 1;
  expected[1][1][2] = 3;
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 3>("a0*b0*c0 + 3*a1*b1*c2")),
            expected);
}

TEST(SparseStringToTensorTest, ToleratesExtraWhitespace) {
  TensorF2<2, 2, 2> expected{};
  expected[1][0][1] = 1;
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 2>("  a1 * b0 * c1  ")),
            expected);
}

TEST(SparseStringToTensorTest, NoSpacesAroundPlus) {
  TensorF2<2, 2, 2> expected{};
  expected[0][0][0] = 1;
  expected[1][1][1] = 1;
  EXPECT_EQ((SparseStringToTensor<2, 2, 2, 2>("a0*b0*c0+a1*b1*c1")),
            expected);
}

// Round-trip: tensor -> string -> tensor should be the identity.
TEST(TensorRoundTripTest, F2) {
  TensorF2<3, 2, 4> tensor{};
  tensor[0][1][3] = 1;
  tensor[2][0][1] = 1;
  tensor[1][1][2] = 1;
  EXPECT_EQ((SparseStringToTensor<2, 3, 2, 4>(
                TensorToSparseString<2, 3, 2, 4>(tensor))),
            tensor);
}

TEST(TensorRoundTripTest, WithCoefficients) {
  TensorF2<3, 3, 3> tensor{};
  tensor[0][0][0] = 1;
  tensor[1][2][0] = 4;
  tensor[2][1][2] = 6;
  EXPECT_EQ((SparseStringToTensor<2, 3, 3, 3>(
                TensorToSparseString<2, 3, 3, 3>(tensor))),
            tensor);
}

TEST(TensorRoundTripTest, LargeN) {
  TensorF2<13, 14, 15> tensor{};
  tensor[10][1][12] = 1;
  tensor[2][11][14] = 1;
  tensor[12][13][2] = 1;
  std::string string = TensorToSparseString<2, 13, 14, 15>(tensor);
  EXPECT_EQ(string, "a2*b11*c14 + a10*b1*c12 + a12*b13*c2");
  EXPECT_EQ((SparseStringToTensor<2, 13, 14, 15>(string)), tensor);
}

TEST(TensorRoundTripTest, AllZeros) {
  TensorF2<2, 3, 2> tensor{};
  EXPECT_EQ((TensorToSparseString<2, 2, 3, 2>(tensor)), "0");
  EXPECT_EQ((SparseStringToTensor<2, 2, 3, 2>("0")), tensor);
}
