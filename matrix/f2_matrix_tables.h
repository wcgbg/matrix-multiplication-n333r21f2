#pragma once

// All lookup tables the matrix-multiplication symmetry group needs, for one
// constraint shape ⟨N0, N1⟩, over 𝔽₂. Replaces the former GLTablesF2<n> +
// ConstraintTablesF2<N0,N1> pair. This is a plain class (no singleton):
// SymmetryGroup holds one instance and lends it to its Query/Store sides.
//
// The tables are stored compactly (uint8/uint16 arrays indexed by a matrix's
// raw uint16 data) and built once in the constructor using F2Matrix algebra
// (operator*, Inversed, Transposed, IsInvertible) rather than inline bit loops.
// The public API speaks F2Matrix and wraps the raw tables via .Data() /
// F2Matrix(uint16_t).
//
//   - Gl0()/Gl1()              : GL(N0)/GL(N1) over 𝔽₂, identity first.
//   - Inverse0()/Inverse1()    : matrix inverse (from a table).
//   - Mult011(m01, m11) = m01·m11    : N0 row lookups (m01: N0×N1, m11: N1×N1).
//   - Mult001(m00, m01) = m00·m01    : N0 row lookups (m00: N0×N0, m01: N0×N1).
//   - TransposeConstraint(m01)   : the N0×N1 → N1×N0 transpose (cubic action).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "matrix/f2_matrix.h"

namespace matrix {

template <int N0, int N1> class F2MatrixTables {
  static_assert(N0 >= 1 && N1 >= 1);
  static_assert(N0 <= 4 && N1 <= 4, "GL lookup tables cap N0, N1 ≤ 4");
  static_assert(N0 * N1 <= 16, "constraint must fit in a uint16");

public:
  static constexpr int kN01 = N0 * N1;
  static constexpr int kN11 = N1 * N1;

  F2MatrixTables() {
    gl0_ = EnumerateGL<N0>();
    gl1_ = EnumerateGL<N1>();
    inverse0_ = BuildInverse<N0>();
    inverse1_ = BuildInverse<N1>();

    // v1: 1xN1 matrix
    // m11: N1xN1 matrix
    // v1_x_m11_[m11][v1] = v1·m11
    v1_x_m11_.resize(std::size_t{1} << (N1 + kN11));
    for (uint32_t m11d = 0; m11d < (uint32_t{1} << kN11); ++m11d) {
      F2Matrix<N1, N1> m11(static_cast<uint16_t>(m11d));
      for (uint16_t v1d = 0; v1d < (uint16_t{1} << N1); ++v1d) {
        F2Matrix<1, N1> v1(v1d);
        F2Matrix<1, N1> prod = v1 * m11;
        size_t index = (static_cast<std::size_t>(m11d) << N1) | v1d;
        v1_x_m11_[index] = static_cast<uint8_t>(prod.Data());
      }
    }

    // v0: 1xN0 matrix
    // m01: N0xN1 matrix
    // v0_x_m01_[m01][v0] = v0·m01
    v0_x_m01_.resize(std::size_t{1} << (N0 + kN01));
    for (uint32_t m01d = 0; m01d < (uint32_t{1} << kN01); ++m01d) {
      F2Matrix<N0, N1> m01(static_cast<uint16_t>(m01d));
      for (uint16_t v0d = 0; v0d < (uint16_t{1} << N0); ++v0d) {
        F2Matrix<1, N0> v0(v0d);
        F2Matrix<1, N1> prod = v0 * m01;
        size_t index = (static_cast<std::size_t>(m01d) << N0) | v0d;
        v0_x_m01_[index] = static_cast<uint8_t>(prod.Data());
      }
    }

    // transpose_01_[m] = mᵀ  (N0×N1 → N1×N0).
    transpose_01_.resize(uint32_t{1} << kN01);
    for (uint32_t m01d = 0; m01d < transpose_01_.size(); ++m01d) {
      F2Matrix<N0, N1> m01(static_cast<uint16_t>(m01d));
      transpose_01_[m01d] = m01.Transposed();
    }
  }

  const std::vector<F2Matrix<N0, N0>> &Gl0() const { return gl0_; }
  const std::vector<F2Matrix<N1, N1>> &Gl1() const { return gl1_; }

  F2Matrix<N0, N0> Identity0() const { return F2Matrix<N0, N0>::Identity(); }
  F2Matrix<N1, N1> Identity1() const { return F2Matrix<N1, N1>::Identity(); }

  F2Matrix<N0, N0> Inverse0(F2Matrix<N0, N0> m00) const {
    return inverse0_[m00.Data()];
  }
  F2Matrix<N1, N1> Inverse1(F2Matrix<N1, N1> m11) const {
    return inverse1_[m11.Data()];
  }

  // m01·m11: walk the N0 rows of m01 through v1_x_m11_. Row i of the product is
  // (row i of m01)·m11, looked up at index (m11 << N1) | row_i(m01) — matching
  // the table layout v1_x_m11_[(m11d << N1) | v1d] = v1·m11.
  F2Matrix<N0, N1> Mult011(F2Matrix<N0, N1> m01, F2Matrix<N1, N1> m11) const {
    const uint16_t m11d = m11.Data();
    const uint16_t m01d = m01.Data();
    const std::size_t base = static_cast<std::size_t>(m11d) << N1;
    uint16_t out = 0;
    for (int i = 0; i < N0; ++i) {
      const auto m01row = (m01d >> (i * N1)) & kN1Mask;
      size_t index = base | m01row;
      out |= static_cast<uint16_t>(v1_x_m11_[index]) << (i * N1);
    }
    return F2Matrix<N0, N1>(out);
  }

  // m00·m01: walk the N0 rows of m00 through v0_x_m01_.
  F2Matrix<N0, N1> Mult001(F2Matrix<N0, N0> m00, F2Matrix<N0, N1> m01) const {
    const uint16_t m00d = m00.Data();
    const uint16_t m01d = m01.Data();
    const std::size_t base = static_cast<std::size_t>(m01d) << N0;
    uint16_t out = 0;
    for (int i = 0; i < N0; ++i) {
      const auto m00row = (m00d >> (i * N0)) & kN0Mask;
      size_t index = base | m00row;
      out |= static_cast<uint16_t>(v0_x_m01_[index]) << (i * N1);
    }
    return F2Matrix<N0, N1>(out);
  }

  F2Matrix<N1, N0> TransposeConstraint(F2Matrix<N0, N1> m01) const {
    return transpose_01_[m01.Data()];
  }

private:
  static constexpr unsigned kN0Mask = (1u << N0) - 1;
  static constexpr unsigned kN1Mask = (1u << N1) - 1;

  template <int n> static std::vector<F2Matrix<n, n>> EnumerateGL() {
    std::vector<F2Matrix<n, n>> out;
    const F2Matrix<n, n> id = F2Matrix<n, n>::Identity();
    out.push_back(id);
    for (uint32_t d = 0; d < (uint32_t{1} << (n * n)); ++d) {
      const F2Matrix<n, n> mm(static_cast<uint16_t>(d));
      if (!(mm == id) && mm.IsInvertible()) {
        out.push_back(mm);
      }
    }
    return out;
  }

  template <int n> static std::vector<F2Matrix<n, n>> BuildInverse() {
    std::vector<F2Matrix<n, n>> inv(uint32_t{1} << (n * n));
    for (uint32_t d = 0; d < inv.size(); ++d) {
      inv[d] = F2Matrix<n, n>(static_cast<uint16_t>(d)).Inversed();
    }
    return inv;
  }

  std::vector<F2Matrix<N0, N0>> gl0_;      // GL(N0, 𝔽₂), identity first
  std::vector<F2Matrix<N1, N1>> gl1_;      // GL(N1, 𝔽₂), identity first
  std::vector<F2Matrix<N0, N0>> inverse0_; // [M] -> M⁻¹ (N0×N0), 0 if singular
  std::vector<F2Matrix<N1, N1>> inverse1_; // [M] -> M⁻¹ (N1×N1), 0 if singular
  std::vector<uint8_t> v1_x_m11_;          // [m11,v1] -> v1·m11
  std::vector<uint8_t> v0_x_m01_;          // [m01,v0] -> v0·m01
  std::vector<F2Matrix<N1, N0>> transpose_01_; // [m01] -> m01ᵀ (N0×N1 -> N1×N0)
};

} // namespace matrix
