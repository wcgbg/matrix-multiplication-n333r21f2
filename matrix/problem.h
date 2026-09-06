#pragma once

// ⟨N0, N1, N2⟩ matrix multiplication over the prime field 𝔽_P.
//
//   T_mat(N0,N1,N2) : 𝔽_q^{N0×N1} × 𝔽_q^{N1×N2} → 𝔽_q^{N0×N2}
//   (X, Y) ↦ X · Y
//
// As a tensor (the A/B/C factors flattened row-major; see tensor.h):
//   T_mat = Σ_{i<N0, j<N1, k<N2} x_{ij} ⊗ y_{jk} ⊗ z_{ki},
// so kNA = N0·N1, kNB = N1·N2, kNC = N2·N0.
//
// Its A-side symmetry group is the √|G| meet-in-the-middle GL_{N0} × GL_{N1}
// split that core/symmetry.h was designed around — see f2_symmetry.h and
// fp_symmetry.h.

#include <format>
#include <string>
#include <type_traits>

#include "core/constraints.h"
#include "core/gf_vec.h" // for IntPow
#include "core/tensor.h"
#include "matrix/fp_symmetry.h"
#include "matrix/f2_symmetry.h"
#include "matrix/tensor.h"

namespace matrix {

template <int P, int N0, int N1, int N2> struct Problem {
  static_assert(N0 >= 1 && N1 >= 1 && N2 >= 1);
  static_assert(N0 * N1 <= 32); // kNA backtracking ceiling
  // Prime fields only: 𝔽₂ uses the lookup-table symmetry group
  // (matrix/f2_symmetry.h), odd primes the arithmetic one
  // (matrix/fp_symmetry.h). The flip-graph upper bound stays 𝔽₂-only.

  static constexpr int kNA = N0 * N1;
  static constexpr int kNB = N1 * N2;
  static constexpr int kNC = N2 * N0;
  static constexpr int kP = P;
  static constexpr int kQ = P;

  // The three matrix-multiplication dimensions, exposed for tooling.
  static constexpr int kN0 = N0;
  static constexpr int kN1 = N1;
  static constexpr int kN2 = N2;

  using Vec = GFVec<P, kNA>;
  using SymmetryGroup =
      std::conditional_t<P == 2, matrix::SymmetryGroup<P, N0, N1, N2>,
                         matrix::FpSymmetryGroup<P, N0, N1, N2>>;

  static Tensor<P, kNA, kNB, kNC> MakeTensor() {
    return BuildMulTensor<P, N0, N1, N2>();
  }

  // <family>_q<QQ>_n<N0><N1><N2>, e.g. matrix_q02_n333. All feasible formats
  // have single-digit dimensions (mirroring the source repo's rmms_n333).
  static std::string Name() {
    return std::format("matrix_q{:02}_n{}{}{}", kQ, N0, N1, N2);
  }
};

} // namespace matrix
