#include "matrix/tensor.h"

#include <string>

#include "gtest/gtest.h"

#include "core/tensor.h"
#include "core/tensor_utils.h"

namespace matrix {
namespace {

// ⟨2,2,2⟩ is the order-2 matrix-multiplication (Strassen) tensor:
// T[i*2+j][j*2+k][k*2+i] = 1 for i,j,k in {0,1}, i.e. 8 nonzero cells.
TEST(BuildMulTensorTest, Strassen222) {
  EXPECT_EQ((TensorToSparseString<2, 4, 4, 4>(BuildMulTensor<2, 2, 2, 2>())),
            "a0*b0*c0 + a0*b1*c2 + a1*b2*c0 + a1*b3*c2 + "
            "a2*b0*c1 + a2*b1*c3 + a3*b2*c1 + a3*b3*c3");
}

// Non-cube ⟨2,2,3⟩: kNA=4, kNB=6, kNC=6; round-trips through the sparse string.
TEST(BuildMulTensorTest, NonCube223RoundTrip) {
  using T = Tensor<2, 4, 6, 6>;
  const T tensor = BuildMulTensor<2, 2, 2, 3>();
  const std::string text = TensorToSparseString<2, 4, 6, 6>(tensor);
  EXPECT_EQ((SparseStringToTensor<2, 4, 6, 6>(text)), tensor);

  // Exactly N0*N1*N2 = 12 nonzero structure constants.
  int nonzeros = 0;
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 6; ++b)
      for (int c = 0; c < 6; ++c)
        if (tensor[a][b][c].value != 0)
          ++nonzeros;
  EXPECT_EQ(nonzeros, 12);
}

} // namespace
} // namespace matrix
