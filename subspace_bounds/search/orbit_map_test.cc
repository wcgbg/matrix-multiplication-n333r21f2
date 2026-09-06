#include "subspace_bounds/search/orbit_map.h"

#include <cstdint>

#include "gtest/gtest.h"

#include "core/bit_vec.h"
#include "core/constraints.h"
#include "core/gf_vec.h"
#include "matrix/problem.h"

namespace {

// <2,2,2> over F_2: the first factor is a 2x2 matrix, NA = 4; bit ij = i*2+j.
using Problem = matrix::Problem<2, 2, 2, 2>;
using Group = Problem::SymmetryGroup;
using QueryElem = Group::QuerySet::Elem;
using StoreElem = Group::StoreSet::Elem;
constexpr int kP2 = 2;

using Row = GFVec<kP2, 4>;
using BV = BitVec<4>;

// Normalise a constraint set to the same canonical RREF the OrbitMap uses for
// keys (drop the zero rows the reversed elimination parks at the front).
Constraints<kP2, 4> Canonical(Constraints<kP2, 4> r) {
  const int rank = GaussJordanRREF<kP2, 4>(&r);
  r.erase(r.begin(), r.end() - rank);
  return r;
}

TEST(OrbitMapTest, SetThenGetReturnsStoredRank) {
  Group group;
  OrbitMap<Problem> om(&group);

  const Constraints<kP2, 4> r = {Row{BV{0b0001}}}; // x_00 = 0
  om.Set(r, 5);

  EXPECT_EQ(om.Get(r), 5);
}

TEST(OrbitMapTest, GetFindsAnyOrbitRepresentative) {
  Group group;
  OrbitMap<Problem> om(&group);

  const Constraints<kP2, 4> canonical = {Row{BV{0b0001}}};
  om.Set(canonical, 7);

  // Every query-image of the canonical set lies in the same orbit, so Get must
  // recover the same rank from any of them, and the witness must map it back.
  for (int i = 0; i < group.query.Size(); ++i) {
    const QueryElem query = group.query.At(i);
    Constraints<kP2, 4> moved;
    for (const auto v : canonical) {
      moved.push_back(group.query.Apply(query, v));
    }
    moved = Canonical(moved);

    QueryElem witness_query{};
    StoreElem witness_store{};
    EXPECT_EQ(om.Get(moved, &witness_query, &witness_store), 7);

    // c = store.ApplyInverse(witness_store, query.Apply(witness_query, q))
    // recovers the canonical form from the (query, store) witness — the
    // contract of OrbitMap::Get (subspace_bounds/search/orbit_map.h).
    Constraints<kP2, 4> back;
    for (const auto v : moved) {
      back.push_back(group.store.ApplyInverse(
          witness_store, group.query.Apply(witness_query, v)));
    }
    EXPECT_EQ(Canonical(back), canonical);
  }
}

TEST(OrbitMapTest, TwoOrbitsWithDifferentRanks) {
  Group group;
  OrbitMap<Problem> om(&group);

  // E_00 (rank one) and the identity (rank two) lie in different orbits.
  om.Set(Constraints<kP2, 4>{Row{BV{0b0001}}}, 4);
  om.Set(Constraints<kP2, 4>{Row{BV{0b1001}}}, 6);

  EXPECT_EQ(om.Get(Constraints<kP2, 4>{Row{BV{0b1001}}}), 6);
  EXPECT_EQ(om.Get(Constraints<kP2, 4>{Row{BV{0b0001}}}), 4);
}

TEST(OrbitMapTest, ClearEmptiesTheMap) {
  Group group;
  OrbitMap<Problem> om(&group);
  om.Set(Constraints<kP2, 4>{Row{BV{0b0001}}}, 3);
  EXPECT_EQ(om.Get(Constraints<kP2, 4>{Row{BV{0b0001}}}), 3);
  om.Clear();
  EXPECT_DEATH(om.Get(Constraints<kP2, 4>{Row{BV{0b0001}}}), "");
}

} // namespace
