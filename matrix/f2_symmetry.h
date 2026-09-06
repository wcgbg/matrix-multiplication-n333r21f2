#pragma once

// A-side symmetry group for ⟨N0, N1, N2⟩ matrix multiplication over 𝔽₂.
//
//   T_mat = Σ_{i<N0, j<N1, k<N2} x_{ij} ⊗ y_{jk} ⊗ z_{ki}.
//
// The A factor is the space of N0×N1 matrices, flattened row-major to 𝔽₂^{kNA}
// with kNA = N0·N1 and flat index ij = i·N1 + j. The rank-preserving A-side
// symmetries (paper §3.1) are the sandwich X ↦ L·X·R (L ∈ GL_{N0}, R ∈ GL_{N1})
// and, for the cubic format N0=N1=N2, the transpose X ↦ Xᵀ. The projective
// group is G = ((GL_{N0} × GL_{N1}) ⋊ C₂) / 𝔽₂^* (the centre of GL(n,2) is
// trivial).
//
// √|G| meet-in-the-middle split (core/symmetry.h):
//   Store := { X ↦ X·R       : R ∈ GL_{N1} }                       (right
//   mult), Query := { X ↦ L·τᵗ(X)   : L ∈ GL_{N0}, t ∈ {0, 1 iff cubic} } (left
//   mult).
// Query⁻¹·Store = {X ↦ L⁻¹·X·R} ∪ {X ↦ L⁻¹·Xᵀ·R} covers G, which is what the
// orbit enumerator and OrbitMap::Get require. We use the *direct* action
// X ↦ L·τ(X)·R (matching the source repo's TransformRestrictions); it ranges
// over the same GL_{N0}×GL_{N1}⋊C₂ bi-action as the contragredient, hence the
// same orbits.
//
// The constraint is just the N0×N1 matrix of the framework's Vec.data bits, so
// the small-matrix algebra is delegated to a held F2MatrixTables<N0,N1>
// instance (matrix/f2_matrix_tables.h): every group action is a few
// table lookups via F2Matrix. This file is a thin adapter — it keeps the
// SymmetryGroupConcept surface and the compact Elem encodings (Store::Elem is
// the uint16 R; QuerySet::Elem is {uint16 l, uint16 transpose}) that the
// certificate witnesses serialise to, and converts Vec ↔ F2Matrix at the table
// boundary.

#include <cstdint>
#include <vector>

#include "core/bit_vec.h"
#include "core/gf_vec.h"
#include "matrix/f2_matrix.h"
#include "matrix/f2_matrix_tables.h"

namespace matrix {

template <int P, int N0, int N1, int N2> class SymmetryGroup {
  static_assert(P == 2, "matrix::SymmetryGroup is the F_2 lookup-table group; odd primes use FpSymmetryGroup");
  static_assert(N0 >= 1 && N1 >= 1 && N2 >= 1);
  static_assert(N0 <= 4 && N1 <= 4,
                "the GL lookup-table approach caps N0, N1 ≤ 4");
  static_assert(N0 * N1 <= 16, "constraint must fit in a uint16");

public:
  static constexpr int kNA = N0 * N1;
  static constexpr bool kCubic = (N0 == N1 && N1 == N2);
  using Vec = GFVec<P, kNA>;

private:
  using Constraint = BitVec<kNA>;
  using Tables = F2MatrixTables<N0, N1>;

  // The constraint's Vec.data bits ARE the N0×N1 matrix (ij = i·N1 + j layout).
  static F2Matrix<N0, N1> ToMat(Vec v) {
    return F2Matrix<N0, N1>(static_cast<uint16_t>(v.data));
  }
  static Vec FromMat(F2Matrix<N0, N1> m) {
    return Vec{static_cast<Constraint>(m.Data())};
  }

public:
  // Store side: right-multiplication by GL(N1). The element IS the matrix R
  // (its uint16 value), so it serialises straight into the certificate witness.
  class StoreSet {
  public:
    using Elem = uint16_t;

    explicit StoreSet(const Tables &tables) : tables_(tables) {}

    int Size() const { return static_cast<int>(tables_.Gl1().size()); }
    Elem At(int j) const { return tables_.Gl1()[j].Data(); }
    Elem Identity() const { return tables_.Identity1().Data(); }
    Vec Apply(Elem r, Vec v) const {
      return FromMat(tables_.Mult011(ToMat(v), F2Matrix<N1, N1>(r)));
    }
    Vec ApplyInverse(Elem r, Vec v) const {
      return FromMat(
          tables_.Mult011(ToMat(v), tables_.Inverse1(F2Matrix<N1, N1>(r))));
    }

  private:
    const Tables &tables_;
  };

  // A Query element: left-multiply by `l ∈ GL(N0)` after an optional transpose.
  // Packs into the certificate's `fixed32 query_elem` via static_cast (the only
  // conversion the serialization layer performs).
  struct QueryElem {
    uint16_t l = 0;
    uint16_t transpose = 0;

    constexpr QueryElem() = default;
    constexpr QueryElem(uint16_t l_value, uint16_t transpose_flag)
        : l(l_value), transpose(transpose_flag) {}
    constexpr explicit QueryElem(uint32_t packed)
        : l(static_cast<uint16_t>(packed & 0xFFFFu)),
          transpose(static_cast<uint16_t>((packed >> 16) & 0xFFFFu)) {}
    constexpr explicit operator uint32_t() const {
      return static_cast<uint32_t>(l) |
             (static_cast<uint32_t>(transpose) << 16);
    }
  };

  // Query side: { X ↦ l · τᵗ(X) : l ∈ GL(N0), t ∈ {0, 1 iff cubic} }.
  class QuerySet {
  public:
    using Elem = QueryElem;

    explicit QuerySet(const Tables &tables) : tables_(tables) {
      const int t_count = kCubic ? 2 : 1;
      for (int t = 0; t < t_count; ++t) {
        for (const F2Matrix<N0, N0> &l : tables_.Gl0()) {
          elems_.push_back(QueryElem{l.Data(), static_cast<uint16_t>(t)});
        }
      }
    }

    int Size() const { return static_cast<int>(elems_.size()); }
    Elem At(int i) const { return elems_[i]; }
    Elem Identity() const { return QueryElem{tables_.Identity0().Data(), 0}; }

    Vec Apply(Elem e, Vec v) const {
      F2Matrix<N0, N1> m = ToMat(v);
      if constexpr (kCubic) {
        if (e.transpose) {
          m = tables_.TransposeConstraint(m); // N0=N1: N1×N0 ≡ N0×N1
        }
      }
      return FromMat(tables_.Mult001(F2Matrix<N0, N0>(e.l), m));
    }

    Vec ApplyInverse(Elem e, Vec v) const {
      // Inverse of m ↦ l·τᵗ(m): τᵗ(l⁻¹·m).
      F2Matrix<N0, N1> w =
          tables_.Mult001(tables_.Inverse0(F2Matrix<N0, N0>(e.l)), ToMat(v));
      if constexpr (kCubic) {
        if (e.transpose) {
          w = tables_.TransposeConstraint(w);
        }
      }
      return FromMat(w);
    }

  private:
    const Tables &tables_;
    std::vector<QueryElem> elems_;
  };

private:
  // Declared before query/store so it is constructed first and their references
  // bind to a live object.
  Tables tables_;

public:
  QuerySet query;
  StoreSet store;

  SymmetryGroup() : query(tables_), store(tables_) {}
};

} // namespace matrix
