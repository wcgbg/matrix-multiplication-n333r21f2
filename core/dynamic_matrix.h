#pragma once

// Matrix over 𝔽_P of size n0 × n1.
//
// Fieldent type is `GF<P>` (one byte; F_q index in [0, q)). For
// P = 2 the entry is a 0/1 bit and addition is XOR (`Plus`); for odd P the GF
// operator overloads use direct mod-P arithmetic.
//
// `Rank` runs Gauss-Jordan elimination over 𝔽_q (pivot row normalized to
// leading 1, then eliminate above and below).
//
// Used by:
//   - core/rank_lower_bound_flatten.h: rank of the A-flattening matrix.
//   - core/rank_lower_bound_forced_product.h: rank of the forced-product
//     collection and combinations.

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "core/gf.h"

template <int P> class DynamicMatrix {
public:
  static_assert(P >= 2 && P < 256);

  using Field = GF<P>;

  DynamicMatrix(int n0, int n1);
  template <size_t n0, size_t n1>
  explicit DynamicMatrix(const std::array<std::array<Field, n1>, n0> &data);

  void ResizeRows(int n0);

  // Each element is rendered via `Field::ToString()`; elements within a row are
  // ','-separated, rows are ';'-separated, whole matrix wrapped in `[...]`.
  // E.g. "[1,0,0,1;1,1,0,1;0,0,0,1]" for a 3×4 matrix over 𝔽_2, or
  // "[10,01,11;10,01,00]" for a 2×3 matrix over 𝔽_4 (digits low-coef first,
  // so 0 → "00", 1 → "10", y → "01", y+1 → "11").
  std::string ToString() const;

  Field operator()(int i0, int i1) const { return data_[Index(i0, i1)]; }
  Field &operator()(int i0, int i1) { return data_[Index(i0, i1)]; }

  DynamicMatrix Plus(const DynamicMatrix &other) const;

  bool IsZero() const;

  // Rank over 𝔽_q via Gauss-Jordan elimination.
  int Rank() const;

  int rows() const { return n0_; }
  int cols() const { return n1_; }

  const Field *data() const { return data_.data(); }

private:
  int Index(int i0, int i1) const { return i0 * n1_ + i1; }

  int n0_ = 0;
  int n1_ = 0;
  std::vector<Field> data_;
};

template <int P>
template <size_t n0, size_t n1>
DynamicMatrix<P>::DynamicMatrix(
    const std::array<std::array<Field, n1>, n0> &data)
    : n0_(n0), n1_(n1), data_(n0 * n1) {
  for (size_t i0 = 0; i0 < n0; ++i0) {
    for (size_t i1 = 0; i1 < n1; ++i1) {
      (*this)(static_cast<int>(i0), static_cast<int>(i1)) = data[i0][i1];
    }
  }
}
