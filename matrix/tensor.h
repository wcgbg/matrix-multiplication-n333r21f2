#pragma once

#include <cstddef>

#include "core/tensor.h"

namespace matrix {

// The bilinear tensor of ⟨N0, N1, N2⟩ matrix multiplication over 𝔽_q (q = P):
// multiplying an N0×N1 matrix by an N1×N2 matrix. The A factor is the space of
// N0×N1 matrices (flat index ij = i·N1 + j), B is N1×N2 (jk = j·N2 + k), C is
// N2×N0 (ki = k·N0 + i). The structure constant is 1 exactly at the diagonal
// triple (ij, jk, ki):
//   T[i·N1+j][j·N2+k][k·N0+i] = 1.
// (A direct port of the matrix-mult repo's MatrixMultiplicationTensor.)
template <int P, std::size_t N0, std::size_t N1, std::size_t N2>
Tensor<P, N0 * N1, N1 * N2, N2 * N0> BuildMulTensor() {
  Tensor<P, N0 * N1, N1 * N2, N2 * N0> tensor{};
  for (std::size_t i = 0; i < N0; ++i) {
    for (std::size_t j = 0; j < N1; ++j) {
      for (std::size_t k = 0; k < N2; ++k) {
        tensor[i * N1 + j][j * N2 + k][k * N0 + i] = 1;
      }
    }
  }
  return tensor;
}

} // namespace matrix
