#pragma once

// A-side symmetry group for ⟨N0, N1, N2⟩ matrix multiplication over a prime
// field 𝔽_P with P odd (the 𝔽₂ case keeps its lookup-table implementation in
// symmetry.h; this file is its arithmetic counterpart for P ≥ 3).
//
// Same group and the same √|G| split as symmetry.h:
//   Store := { X ↦ X·R       : R ∈ GL_{N1}(𝔽_P) }
//   Query := { X ↦ L·τᵗ(X)   : L ∈ GL_{N0}(𝔽_P), t ∈ {0, 1 iff cubic} }
// so Query⁻¹·Store = {X ↦ L⁻¹·τᵗ(X)·R} covers (GL_{N0} × GL_{N1}) ⋊ C₂, the
// rank-preserving first-argument symmetries of T_mat (scalars are included
// redundantly: X ↦ λX acts trivially on subspaces).
//
// A constraint is an N0×N1 matrix of base-field digits (flat index
// ij = i·N1 + j, the framework's GFVec coordinate order). Group elements are
// encoded as the base-P digit expansion of the square matrix (row-major), so
// an element IS its matrix and serialises into the certificate's fixed32
// witness fields; inverses are precomputed per element. The products are
// computed directly (no lookup tables): for the intended sizes (P = 3,
// N0, N1 ≤ 3) that is at most 27 multiplications per action.

#include <array>
#include <cstdint>
#include <vector>

#include "core/gf.h"
#include "core/gf_vec.h"

namespace matrix {

template <int P, int N0, int N1, int N2> class FpSymmetryGroup {
  static_assert(P >= 3 && P < 256, "FpSymmetryGroup is for odd primes; 𝔽₂ uses SymmetryGroup");
  static_assert(N0 >= 1 && N1 >= 1 && N2 >= 1);
  static_assert(IntPow(P, N0 * N0) <= (1 << 24) && IntPow(P, N1 * N1) <= (1 << 24),
                "GL enumeration by matrix code caps P^(n^2) at 2^24");

public:
  static constexpr int kNA = N0 * N1;
  static constexpr bool kCubic = (N0 == N1 && N1 == N2);
  using Vec = GFVec<P, kNA>;
  using F = GF<P>;
  using Code = uint32_t; // base-P digits of a square matrix, row-major

  template <int R, int C> using Mat = std::array<F, static_cast<std::size_t>(R * C)>;

  template <int N> static Code Encode(const Mat<N, N> &m) {
    Code code = 0;
    for (int i = N * N - 1; i >= 0; --i) {
      code = code * P + m[i].value;
    }
    return code;
  }
  template <int N> static Mat<N, N> Decode(Code code) {
    Mat<N, N> m;
    for (int i = 0; i < N * N; ++i) {
      m[i] = F{static_cast<uint8_t>(code % P)};
      code /= P;
    }
    return m;
  }

  template <int R, int K, int C>
  static Mat<R, C> Mul(const Mat<R, K> &a, const Mat<K, C> &b) {
    Mat<R, C> c;
    for (int i = 0; i < R; ++i) {
      for (int j = 0; j < C; ++j) {
        F acc = F::Zero();
        for (int k = 0; k < K; ++k) {
          acc = F::Add(acc, F::Mul(a[i * K + k], b[k * C + j]));
        }
        c[i * C + j] = acc;
      }
    }
    return c;
  }

  template <int N> static Mat<N, N> IdentityMat() {
    Mat<N, N> m;
    m.fill(F::Zero());
    for (int i = 0; i < N; ++i) m[i * N + i] = F::One();
    return m;
  }

  // Gauss-Jordan over 𝔽_P; returns false iff singular.
  template <int N> static bool TryInverse(Mat<N, N> a, Mat<N, N> *out) {
    Mat<N, N> inv = IdentityMat<N>();
    for (int col = 0; col < N; ++col) {
      int piv = -1;
      for (int r = col; r < N; ++r) {
        if (a[r * N + col].value != 0) { piv = r; break; }
      }
      if (piv < 0) return false;
      if (piv != col) {
        for (int j = 0; j < N; ++j) {
          std::swap(a[piv * N + j], a[col * N + j]);
          std::swap(inv[piv * N + j], inv[col * N + j]);
        }
      }
      const F s = F::Inverse(a[col * N + col]);
      for (int j = 0; j < N; ++j) {
        a[col * N + j] = F::Mul(a[col * N + j], s);
        inv[col * N + j] = F::Mul(inv[col * N + j], s);
      }
      for (int r = 0; r < N; ++r) {
        if (r == col || a[r * N + col].value == 0) continue;
        const F f = a[r * N + col];
        for (int j = 0; j < N; ++j) {
          a[r * N + j] = F::Sub(a[r * N + j], F::Mul(f, a[col * N + j]));
          inv[r * N + j] = F::Sub(inv[r * N + j], F::Mul(f, inv[col * N + j]));
        }
      }
    }
    *out = inv;
    return true;
  }

  static Mat<N0, N1> ToMat(const Vec &v) {
    Mat<N0, N1> m;
    for (int i = 0; i < kNA; ++i) m[i] = v[i];
    return m;
  }
  static Vec FromMat(const Mat<N0, N1> &m) {
    Vec v{};
    for (int i = 0; i < kNA; ++i) v.Set(i, m[i]);
    return v;
  }
  static Mat<N0, N1> Transpose(const Mat<N0, N1> &m) {
    static_assert(N0 == N1, "transpose as a constraint map needs N0 == N1");
    Mat<N0, N1> t;
    for (int i = 0; i < N0; ++i) {
      for (int j = 0; j < N1; ++j) t[j * N1 + i] = m[i * N1 + j];
    }
    return t;
  }

  // All of GL_N(𝔽_P) as codes, plus the inverse of every element by code.
  template <int N> struct GlTable {
    std::vector<Code> codes;
    std::vector<Code> inverse; // indexed by code; 0 for singular matrices
    GlTable() {
      const Code total = static_cast<Code>(IntPow(P, N * N));
      inverse.assign(total, 0);
      for (Code c = 0; c < total; ++c) {
        Mat<N, N> inv;
        if (TryInverse<N>(Decode<N>(c), &inv)) {
          codes.push_back(c);
          inverse[c] = Encode<N>(inv);
        }
      }
    }
  };

  class StoreSet {
  public:
    using Elem = Code;
    explicit StoreSet(const GlTable<N1> &gl) : gl_(gl) {}
    int Size() const { return static_cast<int>(gl_.codes.size()); }
    Elem At(int j) const { return gl_.codes[j]; }
    Elem Identity() const { return Encode<N1>(IdentityMat<N1>()); }
    Vec Apply(Elem r, Vec v) const {
      return FromMat(Mul<N0, N1, N1>(ToMat(v), Decode<N1>(r)));
    }
    Vec ApplyInverse(Elem r, Vec v) const {
      return FromMat(Mul<N0, N1, N1>(ToMat(v), Decode<N1>(gl_.inverse[r])));
    }

  private:
    const GlTable<N1> &gl_;
  };

  // Packs into the certificate's fixed32 query_elem: bit 31 = transpose flag,
  // the low 31 bits = the code of L.
  struct QueryElem {
    uint32_t l = 0;
    uint32_t transpose = 0;

    constexpr QueryElem() = default;
    constexpr QueryElem(uint32_t l_value, uint32_t transpose_flag)
        : l(l_value), transpose(transpose_flag) {}
    constexpr explicit QueryElem(uint32_t packed)
        : l(packed & 0x7FFFFFFFu), transpose(packed >> 31) {}
    constexpr explicit operator uint32_t() const { return l | (transpose << 31); }
  };

  class QuerySet {
  public:
    using Elem = QueryElem;
    explicit QuerySet(const GlTable<N0> &gl) : gl_(gl) {
      const int t_count = kCubic ? 2 : 1;
      for (int t = 0; t < t_count; ++t) {
        for (Code c : gl_.codes) elems_.push_back(QueryElem{c, static_cast<uint32_t>(t)});
      }
    }
    int Size() const { return static_cast<int>(elems_.size()); }
    Elem At(int i) const { return elems_[i]; }
    Elem Identity() const { return QueryElem{Encode<N0>(IdentityMat<N0>()), 0}; }
    Vec Apply(Elem e, Vec v) const {
      Mat<N0, N1> m = ToMat(v);
      if constexpr (kCubic) {
        if (e.transpose) m = Transpose(m);
      }
      return FromMat(Mul<N0, N0, N1>(Decode<N0>(e.l), m));
    }
    Vec ApplyInverse(Elem e, Vec v) const {
      Mat<N0, N1> w = Mul<N0, N0, N1>(Decode<N0>(gl_.inverse[e.l]), ToMat(v));
      if constexpr (kCubic) {
        if (e.transpose) w = Transpose(w);
      }
      return FromMat(w);
    }

  private:
    const GlTable<N0> &gl_;
    std::vector<QueryElem> elems_;
  };

private:
  // Constructed before query/store so their references bind to live tables.
  GlTable<N0> gl0_;
  GlTable<N1> gl1_;

public:
  QuerySet query;
  StoreSet store;

  FpSymmetryGroup() : query(gl0_), store(gl1_) {}
};

} // namespace matrix
