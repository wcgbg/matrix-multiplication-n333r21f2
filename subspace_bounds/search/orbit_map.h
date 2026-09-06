#pragma once

// The orbit map: a hash map from canonical constraint sets to their known
// rank lower bound, supporting fast lookup under the action of a SymmetryGroup
// via the square-root trick.
//
// This is a refactor of rank_search/constraints_map.h. The interface is
// unchanged; the implementation calls the SymmetryGroup's nested
// query.Apply / store.Apply instead of the hard-coded GL_{n0} × GL_{n1}
// matrix product.
//
// Mechanism — a meet-in-the-middle, NOT a subgroup decomposition. `Set(c)`
// materialises every store-image store.Apply(store_elem, c) of a canonical form
// as a key; `Get(q)` probes the map with every query-image
// query.Apply(query_elem, q). A hit means query.Apply(query_elem, q) ≡
// store.Apply(store_elem, c) (as subspaces), i.e. q lies in c's orbit via q ≡
// query_elem⁻¹·store_elem·c. Correctness needs only (a) every query/store
// element is a genuine symmetry — so a hit is never a spurious merge — and (b)
// the product set Query⁻¹ · Store covers the group, so the hit is always found.
// Neither side has to be a subgroup; see core/symmetry.h for the full statement
// of the requirement.

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "boost/unordered/unordered_flat_map.hpp"
#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/bit_vec.h"
#include "core/constraints.h"
#include "core/symmetry.h"

template <class Problem> class OrbitMap {
public:
  static constexpr int kP = Problem::kP;
  using Vec = GFVec<kP, Problem::kNA>;
  using SymmetryGroup = typename Problem::SymmetryGroup;
  using QueryElem = typename SymmetryGroup::QuerySet::Elem;
  using StoreElem = typename SymmetryGroup::StoreSet::Elem;

  // Construct an empty map. The SymmetryGroup is borrowed; the caller owns it
  // and must keep it alive for the OrbitMap's lifetime.
  explicit OrbitMap(const SymmetryGroup *group);

  virtual ~OrbitMap() { Clear(); }

  // Insert (Store-orbit-of) the canonical form `c` ↦ `rank` for every
  // store-image of `c`. This is O(store.Size()) per call. Used during DP setup
  // and after each iteration of the outer DP loop, when an orbit gets a new
  // known lower bound.
  void Set(const Constraints<Problem::kP, Problem::kNA> &c,
           int rank);

  // Look up the rank lower bound of (any constraint in the orbit of) the query
  // `q`. Probes the map with each query element in order and returns the rank
  // of the first stored image found (every store-image of an orbit carries that
  // orbit's rank, so the first hit suffices).
  //
  // The witness output parameters report, verbatim, the two elements the query
  // found at the hit: the query element `query_elem` it enumerated, and the
  // store element `store_elem` that Set applied (read from the stored Value).
  // They satisfy the hit equation
  //
  //   query.Apply(query_elem, q) ≡ store.Apply(store_elem, c)  (as subspaces)
  //
  // where c is the canonical form of q's orbit given in Set. Equivalently:
  //
  //     q = query.ApplyInverse(query_elem, store.Apply(store_elem, c))
  //     c = store.ApplyInverse(store_elem, query.Apply(query_elem, q))
  int Get(const Constraints<Problem::kP, Problem::kNA> &q,
          QueryElem *witness_query = nullptr,
          StoreElem *witness_store = nullptr) const;

  void Clear();

private:
  // Sharded for parallel inserts; same constant as the matrix-mult code.
  static constexpr int kNumShards = 997;

  struct Value {
    // Rank lower bound (fits in uint8_t for all reasonable problem sizes).
    uint8_t rank = 0;
    // The StoreElem that takes the canonical form *to* this stored key, i.e.
    // store.Apply(store_elem, c) ≡ key. Reported verbatim by Get(). For groups
    // where the store action is a "small" object (Vec or int), this is
    // cheap to store.
    StoreElem store_elem{};
  };
  using Map = boost::unordered_flat_map<
      Constraints<Problem::kP, Problem::kNA>, Value,
      ConstraintsHash<Problem::kP, Problem::kNA>>;

  const SymmetryGroup *group_ = nullptr;
  std::array<Map, kNumShards> shards_;
  std::array<std::mutex, kNumShards> mutexes_;
};

template <class Problem>
OrbitMap<Problem>::OrbitMap(const SymmetryGroup *group) : group_(group) {
  CHECK_NOTNULL(group_);
}

template <class Problem>
void OrbitMap<Problem>::Set(
    const Constraints<Problem::kP, Problem::kNA> &c, int rank) {
  CHECK_GE(rank, 0);
  CHECK_LT(rank, 256);
  // Insert one entry per store-image of `c`. Each image is the subspace
  // store.Apply(store_elem, ·) applied to `c`, re-normalised to the canonical
  // (column-reversed) RREF so it can serve as a hash key. Because store.Apply
  // is invertible, the image keeps full rank — no zero rows to drop.
  const int store_size = group_->store.Size();
  Constraints<Problem::kP, Problem::kNA> image;
  for (int j = 0; j < store_size; ++j) {
    const StoreElem store_elem = group_->store.At(j);
    image.clear();
    image.reserve(c.size());
    for (const Vec v : c) {
      image.push_back(group_->store.Apply(store_elem, v));
    }
    const int image_rank =
        GaussJordanRREF<Problem::kP, Problem::kNA>(&image);
    CHECK_EQ(image_rank, static_cast<int>(c.size()));
    const std::size_t shard =
        ConstraintsHash<Problem::kP, Problem::kNA>{}(image) %
        kNumShards;
    std::lock_guard<std::mutex> lock(mutexes_[shard]);
    shards_[shard][image] = Value{static_cast<uint8_t>(rank), store_elem};
  }
}

template <class Problem>
int OrbitMap<Problem>::Get(
    const Constraints<Problem::kP, Problem::kNA> &q,
    QueryElem *witness_query, StoreElem *witness_store) const {
  // Probe the map: for each query element, normalise query.Apply(query, q)
  // to canonical RREF and look it up. A stored key K means some processed
  // orbit's canonical form c satisfies store.Apply(store_elem, c) ≡ K ≡
  // query.Apply(query_elem, q) (as subspaces), so q lies in that orbit. The
  // first hit settles the orbit; we return its rank and report the
  // (query_elem, store_elem) witness verbatim. No locking: Get is only called
  // in DP phases that do not run concurrently with Set.
  const int query_size = group_->query.Size();
  Constraints<Problem::kP, Problem::kNA> image;
  for (int i = 0; i < query_size; ++i) {
    const QueryElem query_elem = group_->query.At(i);
    image.clear();
    image.reserve(q.size());
    for (const Vec v : q) {
      image.push_back(group_->query.Apply(query_elem, v));
    }
    const int image_rank =
        GaussJordanRREF<Problem::kP, Problem::kNA>(&image);
    // Reversed RREF parks zero rows at the front; drop them so the key matches
    // the full-rank keys stored by Set.
    image.erase(image.begin(), image.end() - image_rank);
    const std::size_t shard =
        ConstraintsHash<Problem::kP, Problem::kNA>{}(image) %
        kNumShards;
    const auto &map = shards_[shard];
    const auto it = map.find(image);
    if (it != map.end()) {
      if (witness_query != nullptr) {
        *witness_query = query_elem;
      }
      if (witness_store != nullptr) {
        *witness_store = it->second.store_elem;
      }
      return it->second.rank;
    }
  }
  LOG(FATAL) << "Constraints not found in OrbitMap (size=" << q.size() << ")";
}

template <class Problem> void OrbitMap<Problem>::Clear() {
  tbb::parallel_for(0, kNumShards, [this](int i) {
    std::lock_guard<std::mutex> lock(mutexes_[i]);
    shards_[i].clear();
  });
}
