#pragma once

// A small, dense matrix over 𝔽₂, packed into a uint16: bit (i*N1 + j) is
// the entry at row i, column j (row-major; the same layout as the framework's
// constraint Vec.data and the matrix-mult repo's StaticMatrix<n0,n1>). Only
// N0*N1 ≤ 16 is supported (one uint16). This is a plain value type
// used to build the symmetry-group lookup tables
// (matrix/f2_matrix_tables.h) and to make the small-matrix algebra
// (transpose, product, inverse) explicit instead of inline bit-twiddling.

#include <array>
#include <cstdint>
#include <utility>

namespace matrix {

template <int N0, int N1 = N0> class F2Matrix {
  static_assert(N0 >= 1 && N1 >= 1);
  static_assert(N0 * N1 <= 16, "F2Matrix must fit in a uint16");

public:
  F2Matrix() = default;
  // Kept constexpr because Identity() constructs through it.
  explicit constexpr F2Matrix(uint16_t data) : data_(data) {}

  uint16_t Data() const { return data_; }

  uint8_t Get(int i, int j) const {
    return static_cast<uint8_t>((data_ >> (i * N1 + j)) & 1u);
  }
  void Set(int i, int j, uint8_t bit) {
    const uint16_t mask = static_cast<uint16_t>(1u << (i * N1 + j));
    data_ = static_cast<uint16_t>(bit ? (data_ | mask) : (data_ & ~mask));
  }

  static constexpr F2Matrix Identity() {
    static_assert(N0 == N1, "Identity requires a square matrix");
    uint16_t d = 0;
    for (int i = 0; i < N0; ++i) {
      d = static_cast<uint16_t>(d | (1u << (i * N1 + i)));
    }
    return F2Matrix(d);
  }

  F2Matrix<N1, N0> Transposed() const {
    F2Matrix<N1, N0> out;
    for (int i = 0; i < N0; ++i) {
      for (int j = 0; j < N1; ++j) {
        if (Get(i, j)) {
          out.Set(j, i, 1);
        }
      }
    }
    return out;
  }

  // Standard 𝔽₂ matrix product (N0×N1)·(N1×N2) = (N0×N2).
  template <int N2>
  F2Matrix<N0, N2> operator*(const F2Matrix<N1, N2> &m12) const {
    F2Matrix<N0, N2> out;
    for (int i = 0; i < N0; ++i) {
      for (int j = 0; j < N2; ++j) {
        uint8_t acc = 0;
        for (int k = 0; k < N1; ++k) {
          acc ^= static_cast<uint8_t>(Get(i, k) & m12.Get(k, j));
        }
        if (acc) {
          out.Set(i, j, 1);
        }
      }
    }
    return out;
  }

  // Rank over 𝔽₂ via Gaussian elimination on the N0 row-vectors.
  int Rank() const {
    std::array<uint16_t, N0> rows{};
    constexpr uint16_t kColMask = static_cast<uint16_t>((1u << N1) - 1);
    for (int i = 0; i < N0; ++i) {
      rows[i] = static_cast<uint16_t>((data_ >> (i * N1)) & kColMask);
    }
    int rank = 0;
    int col = 0;
    int row = 0;
    while (row < N0 && col < N1) {
      int pivot = -1;
      for (int r = row; r < N0; ++r) {
        if (rows[r] & (1u << col)) {
          pivot = r;
          break;
        }
      }
      if (pivot < 0) {
        ++col;
        continue;
      }
      std::swap(rows[row], rows[pivot]);
      for (int r = 0; r < N0; ++r) {
        if (r != row && (rows[r] & (1u << col))) {
          rows[r] = static_cast<uint16_t>(rows[r] ^ rows[row]);
        }
      }
      ++rank;
      ++row;
      ++col;
    }
    return rank;
  }

  bool IsInvertible() const {
    static_assert(N0 == N1, "IsInvertible requires a square matrix");
    return Rank() == N0;
  }

  // Gauss-Jordan inverse over 𝔽₂. Returns the zero matrix if singular (a valid
  // inverse is never zero, so 0 is an unambiguous sentinel).
  F2Matrix Inversed() const {
    static_assert(N0 == N1, "Inversed requires a square matrix");
    constexpr uint16_t kColMask = static_cast<uint16_t>((1u << N1) - 1);
    std::array<uint16_t, N0> rows{};
    std::array<uint16_t, N0> inv{};
    for (int i = 0; i < N0; ++i) {
      rows[i] = static_cast<uint16_t>((data_ >> (i * N1)) & kColMask);
      inv[i] = static_cast<uint16_t>(1u << i);
    }
    for (int col = 0; col < N0; ++col) {
      int pivot = -1;
      for (int r = col; r < N0; ++r) {
        if (rows[r] & (1u << col)) {
          pivot = r;
          break;
        }
      }
      if (pivot < 0) {
        return F2Matrix(0);
      }
      std::swap(rows[col], rows[pivot]);
      std::swap(inv[col], inv[pivot]);
      for (int r = 0; r < N0; ++r) {
        if (r != col && (rows[r] & (1u << col))) {
          rows[r] = static_cast<uint16_t>(rows[r] ^ rows[col]);
          inv[r] = static_cast<uint16_t>(inv[r] ^ inv[col]);
        }
      }
    }
    uint16_t out = 0;
    for (int i = 0; i < N0; ++i) {
      out = static_cast<uint16_t>(out | (inv[i] << (i * N1)));
    }
    return F2Matrix(out);
  }

  bool operator==(const F2Matrix &other) const { return data_ == other.data_; }
  bool operator<(const F2Matrix &other) const { return data_ < other.data_; }

private:
  uint16_t data_ = 0;
};

} // namespace matrix
