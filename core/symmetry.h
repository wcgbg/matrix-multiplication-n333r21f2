#pragma once

// The SymmetryGroup concept.
//
// A SymmetryGroup describes the A-side symmetry group G of a problem — the
// linear maps on the dual A-space induced by tensor automorphisms. G acts on
// constraint *subspaces* (row spans, keyed by column-reversed RREF), so the
// scalar centre is quotiented out and the group is effectively projective. It
// is presented split into two sides, `query` and `store`, that the orbit-map
// machinery combines by a meet-in-the-middle: a group element is reached as
// "query after store", v ↦ query.Apply(query_elem, store.Apply(store_elem, v)).
//
// What the split must satisfy. This is WEAKER than "two subgroups whose product
// is G"; in particular neither side need be closed under multiplication, and
// the factorisation of a g ∈ G into (s, t) need not be unique:
//
//   1. Soundness — every query element and every store element is itself an
//      element of G (a genuine A-side symmetry acting on subspaces). Then every
//      meet-in-the-middle hit is a real orbit coincidence, never a spurious
//      merge of distinct orbits.
//   2. Coverage — the product *set* must cover G, so the orbit enumerator and
//      OrbitMap find every member of an orbit. Two product sets are in play:
//      the enumerator forms Query⁻¹ · Store directly, while OrbitMap::Get
//      matches query.Apply(query_elem, q) ≡ store.Apply(store_elem, c) and so
//      needs Query⁻¹ · Store to cover G. Both must hold; they coincide when
//      Query is inverse-closed, which is the case for both matrix symmetry
//      groups (GL_{N0}, extended by transpose). Store has NO closure requirement of
//      any kind.
//
// There is deliberately no separate "the inverse must be an element" rule. Each
// side exposes both `Apply` (the forward action of an element) and
// `ApplyInverse` (the action of that element's inverse, as a map). ApplyInverse
// is always implementable — every symmetry is an invertible linear map — and
// crucially does NOT require the inverse to be an enumerated element of the
// side. The verifier replays a witness as store.ApplyInverse(store_elem, …)
// (core/backtracking_verifier.h). A side that happens to be a closed group may
// implement ApplyInverse by looking up the inverse element in a precomputed
// table (as all three current problems do); a non-closed covering set would
// instead invert the action directly.
//
// Subgroups with G = Query⁻¹ · Store and Query ∩ Store = {e} (an exact /
// Zappa–Szép factorisation) satisfy soundness and coverage AND give a *unique*
// (s, t) per g. That is what all three current problems use: it is convenient
// (one witness per orbit, no double-counting, |G| = Query.Size()·Store.Size()
// with no collisions — see the *FactorizationIsUnique tests) but it is NOT
// required. A covering by two suitable subsets is enough. The relaxation
// matters when
// a problem has two large symmetry factors of comparable size (e.g. GL_n × GL_m
// for matrix multiplication): splitting them across the two sides is a genuine
// √|G| meet-in-the-middle — Store.Size() ≈ Query.Size() ≈ √|G| — rather than
// a "one big factor on store, one tiny factor on query" split.
//
// Cost model (independent of the algebra): OrbitMap::Set materialises every
// store-image of a canonical form (≈ Store.Size() memory per orbit), and
// OrbitMap::Get probes the map with every query element (O(Query.Size()) per
// Get). A problem trades memory for lookup time by how it assigns its factors
// to the two sides.
//
// Each side is its own nested struct exposing the small uniform interface:
//
//   using Elem = ...;
//   int  Size() const;            // number of elements on this side
//   Elem At(int i) const;         // i-th element, i ∈ [0, Size())
//   Elem Identity() const;        // identity element
//   Vec  Apply(Elem e, Vec v) const;         // action of e on a row vector
//   Vec  ApplyInverse(Elem e, Vec v) const;  // action of e⁻¹ on a row vector
//
// The outer SymmetryGroup class holds two public members `query` and `store`,
// instances of nested `QuerySet` and `StoreSet` structs respectively.
// Elements are addressed by integer index (At over [0, Size())) and act on a
// single constraint functional represented as a Vec — the row-vector type in
// (𝔽_q^N)* with q = P. For P = 2 it's a bit-packed BitVec<N>;
// for odd P it's a GFVec<P, N>. The concrete Vec type is
// supplied by the Problem. Concrete implementations live in
// matrix/f2_symmetry.h and matrix/fp_symmetry.h.

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
