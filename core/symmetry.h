#pragma once

// The SymmetryGroup concept.
//
// A SymmetryGroup describes invertible A-side actions induced by tensor
// automorphisms. Actions are compared on constraint subspaces (row spans,
// keyed by column-reversed RREF), so scalar multiples act identically.
//
// The group is presented as two enumerated sets, `query` and `store`, for
// meet-in-the-middle orbit lookup. The split must satisfy:
//   1. Soundness: every element on either side is a genuine symmetry.
//   2. Coverage: the product set Query^-1 * Store covers the group on
//   subspaces.
// Neither side needs to be closed under multiplication or inversion, and a
// symmetry may have several representations as a query/store pair. In the
// odd-prime matrix implementation, scalar matrices give redundant actions.
//
// The orbit enumerators apply query.ApplyInverse(query_elem,
// store.Apply(store_elem, v)). OrbitMap::Get instead searches for a hit
//   query.Apply(query_elem, q) == store.Apply(store_elem, c)
// as subspaces, where c is a stored canonical representative. These use the
// same covering set: q == query_elem^-1 * store_elem * c.
//
// The verifier recovers c from a recorded witness by applying
// store.ApplyInverse(store_elem, query.Apply(query_elem, q)) and taking RREF
// (subspace_bounds/verifier/backtracking_verifier.h). ApplyInverse implements
// the inverse action; it does not require that inverse to appear in At().
//
// Cost model: OrbitMap::Set materializes each Store-image of a canonical form
// (up to Store.Size() entries per orbit), while OrbitMap::Get probes with each
// Query-image (up to Query.Size() probes). The split trades memory for lookup
// time; similarly sized factors give the square-root meet-in-the-middle cost.
//
// Each side exposes:
//   using Elem = ...;
//   int  Size() const;
//   Elem At(int i) const;                    // i-th element, 0 <= i < Size()
//   Elem Identity() const;
//   Elem DecodeChecked(uint32_t code) const; // validate a serialized witness
//   Vec  Apply(Elem e, Vec v) const;
//   Vec  ApplyInverse(Elem e, Vec v) const;
//
// At(i) returns an element value; the interface does not identify it with i.
// Matrix Store elements encode the right matrix; Query elements encode the
// left-multiplication matrix and an optional transpose. Witnesses serialize
// those values, and the verifier validates them with DecodeChecked before use.
//
// The outer class holds public `query` and `store` members of nested QuerySet
// and StoreSet types. Actions operate on a GFVec<P, N> row: bit-packed for P=2,
// and one field digit per coordinate for odd P. Concrete implementations live
// in matrix/f2_symmetry.h and matrix/fp_symmetry.h.

#include <concepts>

template <class G>
concept SymmetryGroupConcept =
    requires(const G g, typename G::QuerySet::Elem query,
             typename G::StoreSet::Elem store, typename G::Vec v, int i) {
      typename G::Vec;
      typename G::QuerySet;
      typename G::StoreSet;
      typename G::QuerySet::Elem;
      typename G::StoreSet::Elem;

      { g.query } -> std::same_as<const typename G::QuerySet &>;
      { g.store } -> std::same_as<const typename G::StoreSet &>;

      { g.query.Size() } -> std::convertible_to<int>;
      { g.store.Size() } -> std::convertible_to<int>;

      { g.query.At(i) } -> std::same_as<typename G::QuerySet::Elem>;
      { g.store.At(i) } -> std::same_as<typename G::StoreSet::Elem>;
      { g.query.Identity() } -> std::same_as<typename G::QuerySet::Elem>;
      { g.store.Identity() } -> std::same_as<typename G::StoreSet::Elem>;

      { g.query.Apply(query, v) } -> std::same_as<typename G::Vec>;
      { g.store.Apply(store, v) } -> std::same_as<typename G::Vec>;
      { g.query.ApplyInverse(query, v) } -> std::same_as<typename G::Vec>;
      { g.store.ApplyInverse(store, v) } -> std::same_as<typename G::Vec>;
    };
