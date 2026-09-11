#pragma once

#include "ng-log/logging.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace profiles::fp {

template <int P, int N0, int N1> struct Geometry {
  static constexpr int NA = N0 * N1;
  static constexpr int Q = [] {
    int q = 1;
    for (int i = 0; i < NA; ++i)
      q *= P;
    return q;
  }();
  using Digits = std::array<uint8_t, NA>;

  // ------------------------------------------------------------ field
  static uint8_t Inv(uint8_t a) {
    for (int b = 1; b < P; ++b) {
      if (a * b % P == 1)
        return static_cast<uint8_t>(b);
    }
    CHECK(false) << "no inverse";
    return 0;
  }
  static Digits Decode(uint32_t code) {
    Digits d;
    for (int i = 0; i < NA; ++i) {
      d[i] = static_cast<uint8_t>(code % P);
      code /= P;
    }
    return d;
  }
  static uint32_t Encode(const Digits &d) {
    uint32_t code = 0;
    for (int i = NA - 1; i >= 0; --i)
      code = code * P + d[i];
    return code;
  }
  static bool IsZero(const Digits &d) {
    for (int i = 0; i < NA; ++i) {
      if (d[i])
        return false;
    }
    return true;
  }
  // a - c*b
  static void SubScaled(Digits &a, int c, const Digits &b) {
    for (int i = 0; i < NA; ++i)
      a[i] = static_cast<uint8_t>((a[i] + (P - c) * b[i] % P) % P);
  }
  static Digits Normalize(Digits d) { // first nonzero digit -> 1
    for (int i = 0; i < NA; ++i) {
      if (d[i]) {
        const int s = Inv(d[i]);
        for (int j = 0; j < NA; ++j)
          d[j] = static_cast<uint8_t>(d[j] * s % P);
        return d;
      }
    }
    return d;
  }
  // rank of the N0 x N1 matrix with entry (i, j) = d[i*N1 + j]
  static int MatRank(const Digits &d) {
    std::array<std::array<int, N1>, N0> m;
    for (int i = 0; i < N0; ++i)
      for (int j = 0; j < N1; ++j)
        m[i][j] = d[i * N1 + j];
    int r = 0;
    for (int col = 0; col < N1 && r < N0; ++col) {
      int piv = -1;
      for (int i = r; i < N0; ++i) {
        if (m[i][col]) {
          piv = i;
          break;
        }
      }
      if (piv < 0)
        continue;
      std::swap(m[r], m[piv]);
      const int s = Inv(static_cast<uint8_t>(m[r][col]));
      for (int j = 0; j < N1; ++j)
        m[r][j] = m[r][j] * s % P;
      for (int i = 0; i < N0; ++i) {
        if (i == r || !m[i][col])
          continue;
        const int f = m[i][col];
        for (int j = 0; j < N1; ++j)
          m[i][j] = (m[i][j] + (P - f) * m[r][j]) % P;
      }
      ++r;
    }
    return r;
  }

  // -------------------------------------------------------- subspaces
  static constexpr int kMaxDim = NA - 1; // dim NA is the whole space: cap = r
  struct Span {
    uint8_t dim = 0;
    Digits rows[NA] = {};
    int piv[NA] = {};
  };
  static constexpr int kRowBits = [] {
    int b = 0;
    while ((1u << b) <= static_cast<unsigned>(Q))
      ++b;
    return b;
  }();
  static_assert(kRowBits * NA + 4 <= 64, "span key must fit in 64 bits");

  // RREF (pivot = first nonzero column, pivot entry 1, pivot columns cleared),
  // rows in ascending pivot order.
  static Span Rref(std::vector<Digits> rows) {
    int r = 0;
    Span s;
    for (int col = 0; col < NA; ++col) {
      int p = -1;
      for (int k = r; k < static_cast<int>(rows.size()); ++k) {
        if (rows[k][col]) {
          p = k;
          break;
        }
      }
      if (p < 0)
        continue;
      std::swap(rows[r], rows[p]);
      const int inv = Inv(rows[r][col]);
      for (int j = 0; j < NA; ++j)
        rows[r][j] = static_cast<uint8_t>(rows[r][j] * inv % P);
      for (int k = 0; k < static_cast<int>(rows.size()); ++k) {
        if (k != r && rows[k][col])
          SubScaled(rows[k], rows[k][col], rows[r]);
      }
      s.piv[r] = col;
      ++r;
    }
    // Copy only now: later pivot columns keep eliminating earlier rows.
    for (int i = 0; i < r; ++i)
      s.rows[i] = rows[i];
    s.dim = static_cast<uint8_t>(r);
    return s;
  }
  static uint64_t Key(const Span &s) {
    uint64_t key = s.dim;
    for (int i = 0; i < s.dim; ++i)
      key = (key << kRowBits) | Encode(s.rows[i]);
    return key;
  }
  static bool InSpan(const Span &s, Digits v) {
    for (int i = 0; i < s.dim; ++i) {
      if (v[s.piv[i]])
        SubScaled(v, v[s.piv[i]], s.rows[i]);
    }
    return IsZero(v);
  }
  static Span Join(const Span &s, const Digits &v) {
    std::vector<Digits> rows(s.rows, s.rows + s.dim);
    rows.push_back(v);
    return Rref(rows);
  }
};

} // namespace profiles::fp
