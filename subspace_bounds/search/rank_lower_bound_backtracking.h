#pragma once

// Substitution-with-Backtracking lower bound (Algorithm 2).
//
// Idea: starting from the current subspace, repeatedly substitute (add) extra
// constraints. A subset S of added constraints whose more-constrained orbit
// has rank lower bound r witnesses that the current tensor has rank ≥ |S| + r
// (each substitution can save at most one multiplication). A DFS explores
// growing chains of substitutions, pruning a branch once it can no longer beat
// the target. The deepest chain that beats `known_rank_lower_bound + 1` proves
// the improved bound.
//
// Almost-verbatim port of rank_search/rank_lower_bound_backtracking.h. The
// changes are mechanical: n0·n1 → NA, StaticMatrixData → Vec, and the map
// witness (transpose, gl_left, gl_right) → (query_elem, store_elem) from the
// OrbitMap.
//
// The search records the compressed DFS trace into a BacktrackingProof
// and returns it to the caller (which stores it in the per-certificate
// BacktrackingProofArchive); it also reports the trace length in
// pb::BacktrackingProof.proof_size. The witness payload per step is the
// (query_elem, store_elem) pair returned by OrbitMap::Get, which
// core/backtracking_verifier.h replays.

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "boost/unordered/unordered_flat_map.hpp"
#include "ng-log/logging.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "core/backtracking_proof.h"
#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "subspace_bounds/search/orbit_map.h"

template <class Problem> class RankLowerBoundBacktracking {
public:
  static constexpr int NA = Problem::kNA;
  static constexpr int P = Problem::kP;
  using Vec = GFVec<P, NA>;
  using SymmetryGroup = typename Problem::SymmetryGroup;
  using QueryElem = typename SymmetryGroup::QuerySet::Elem;
  using StoreElem = typename SymmetryGroup::StoreSet::Elem;

  // Returns (rank, proof_proto, trace): on improvement `trace` is the winning
  // DFS trace (the caller stores it in the per-certificate archive); on no
  // improvement the rank is 0 and `trace` is empty.
  static std::tuple<int, pb::BacktrackingProof, BacktrackingProof>
  Search(const Constraints<P, NA> &constraints,
         const OrbitMap<Problem> &orbit_map, int known_rank_lower_bound,
         uint64_t step_limit, size_t max_map_size) {
    return RankLowerBoundBacktracking(constraints, orbit_map,
                                      known_rank_lower_bound, step_limit,
                                      max_map_size)
        .Search();
  }

private:
  struct LocalMapValue {
    uint8_t rank = 0;
    QueryElem query_elem{};
    StoreElem store_elem{};
  };
  using LocalMap =
      boost::unordered_flat_map<Constraints<P, NA>, LocalMapValue,
                                ConstraintsHash<P, NA>>;

  RankLowerBoundBacktracking(const Constraints<P, NA> &constraints,
                             const OrbitMap<Problem> &orbit_map,
                             int known_rank_lower_bound, uint64_t step_limit,
                             size_t max_map_size)
      : base_constraints_(constraints), orbit_map_(orbit_map),
        step_limit_(step_limit), max_map_size_(max_map_size) {
    // Unlike the matrix-mult code we do not re-query the map for the base
    // orbit's bound here: the caller already passes the best bound found by the
    // flatten / forced-product / degenerate methods, and the base orbit is not
    // yet inserted into the OrbitMap during its own dimension's pass.
    known_rank_lower_bound_ = known_rank_lower_bound;
    target_rank_lower_bound_ = known_rank_lower_bound_ + 1;
    max_depth_ = known_rank_lower_bound_;

    CHECK_LT(max_depth_, 32);
    for (int bitwidth = 0; bitwidth < max_depth_; ++bitwidth) {
      std::vector<uint32_t> masks_i(uint32_t(1) << bitwidth);
      std::iota(masks_i.begin(), masks_i.end(), uint32_t(0));
      std::sort(masks_i.begin(), masks_i.end(), [&](uint32_t a, uint32_t b) {
        return std::popcount(a) < std::popcount(b);
      });
      masks_.push_back(std::move(masks_i));
    }

    // The candidate extra constraints are the dual vectors that are "minimal"
    // with respect to the base: subtracting any base constraint does not lower
    // them lex. This enumerates one representative per coset of base.
    //
    // For 𝔽₂: minimal iff XOR-ing in any base constraint does not lower the
    // candidate. Equivalently, the candidate has 0 at every base pivot column.
    // The XOR check is what the matrix-mult predecessor uses.
    //
    // For 𝔽_p (P>2): minimal iff the candidate has 0 at every base pivot.
    // Because base_constraints_ is in canonical RREF, each base vector has a
    // unique pivot column with coefficient 1; for c ∈ F_p^* the subtraction
    // r − c·b zeroes the candidate's value at that pivot column iff r had a
    // nonzero there. Lex order being "high coordinate first", this is exactly
    // the coset-min condition.
    static_assert(NA < 32, "Backtracking DFS path mask is uint32_t");
    // Enumerate q^NA = P^NA candidate row-vectors. For P = 2 the
    // value is the bit-packed integer (legacy F₂ path); otherwise we decode
    // the integer counter into an GFVec.
    const uint64_t num_candidates = IntPow(P, NA);
    std::vector<int> base_pivots;
    base_pivots.reserve(base_constraints_.size());
    for (const auto &b : base_constraints_) {
      base_pivots.push_back(b.LeadingNonzeroIdx());
    }
    for (uint64_t value = 1; value < num_candidates; ++value) {
      const Vec constraint = DecodeGFVec<P, NA>(value);
      if (constraint.LeadingNonzero() != GF<P>::One()) {
        continue;
      }
      bool is_minimal = true;
      for (int p_idx : base_pivots) {
        // Element value at the pivot column must be the field zero.
        if (constraint[p_idx] != GF<P>::Zero()) {
          is_minimal = false;
          break;
        }
      }
      if (is_minimal) {
        minimal_constraints_.push_back(constraint);
      }
    }
  }

  std::tuple<int, pb::BacktrackingProof, BacktrackingProof> Search() const {
    if (base_constraints_.size() == static_cast<std::size_t>(NA)) {
      return {0, {}, {}};
    }
    if (max_depth_ == 0) {
      return {0, {}, {}};
    }
    if (step_limit_ == 0) {
      return {0, {}, {}};
    }

    // Parallelise the top level of the DFS over the first added constraint.
    std::atomic<bool> early_break = false;
    std::atomic<uint64_t> step_count(0);
    std::vector<std::pair<int, BacktrackingProof>> rank_and_proof_list(
        minimal_constraints_.size());
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, minimal_constraints_.size()),
        [&](const tbb::blocked_range<size_t> &range) {
          Constraints<P, NA> dfs_constraints;
          LocalMap local_map;
          for (size_t i = range.begin(); i < range.end(); ++i) {
            if (early_break) {
              break;
            }
            dfs_constraints.push_back(minimal_constraints_[i]);
            rank_and_proof_list[i].first =
                Search(static_cast<int>(i), &step_count, &dfs_constraints,
                       &local_map, &rank_and_proof_list[i].second);
            dfs_constraints.pop_back();
            if (rank_and_proof_list[i].first <= known_rank_lower_bound_) {
              early_break = true;
            }
          }
        });

    int rank_lower_bound_child = std::numeric_limits<int>::max();
    size_t proof_size = 0;
    for (const auto &rank_and_proof : rank_and_proof_list) {
      rank_lower_bound_child =
          std::min(rank_lower_bound_child, rank_and_proof.first);
      proof_size += rank_and_proof.second.Size();
    }
    if (rank_lower_bound_child <= known_rank_lower_bound_) {
      return {0, {}, {}};
    }
    // Concatenate the per-branch traces in branch order (DFS pre-order
    // overall).
    BacktrackingProof proof;
    proof.Reserve(proof_size);
    for (const auto &rank_and_proof : rank_and_proof_list) {
      proof.Append(rank_and_proof.second);
    }
    CHECK_EQ(proof.Size(), proof_size);
    pb::BacktrackingProof proof_proto;
    proof_proto.set_proof_size(static_cast<uint32_t>(proof_size));
    return {rank_lower_bound_child, std::move(proof_proto), std::move(proof)};
  }

  // Recursive DFS. Returns the rank lower bound provable from the chain in
  // *dfs_constraints, appending to *proof one record per branch that reaches
  // the target (in DFS pre-order).
  int Search(int max_constraint_idx, std::atomic<uint64_t> *step_count,
             Constraints<P, NA> *dfs_constraints, LocalMap *local_map,
             BacktrackingProof *proof) const {
    if (step_count->fetch_add(1, std::memory_order_relaxed) >= step_limit_) {
      return 0;
    }
    int rank_lower_bound_self = 0;
    CHECK(!dfs_constraints->empty());
    if (local_map->size() >= max_map_size_) {
      // Cheap O(n) stochastic halving to bound memory (not LRU).
      unsigned int bit = local_map->size() & 1;
      for (auto it = local_map->begin(); it != local_map->end();) {
        if (bit == 0) {
          it = local_map->erase(it);
        } else {
          ++it;
        }
        bit ^= 1;
      }
    }
    const size_t dfs_size = dfs_constraints->size();
    const auto &masks = masks_[dfs_size - 1];
    Constraints<P, NA> constraints;
    constraints.reserve(dfs_size);
    for (uint32_t mask : masks) {
      constraints.clear();
      // Iterate only over set bits (LSB -> MSB).
      for (uint32_t m = mask; m != 0; m &= m - 1) {
        constraints.push_back((*dfs_constraints)[std::countr_zero(m)]);
      }
      constraints.push_back(dfs_constraints->back());
      auto [it, inserted] = local_map->try_emplace(constraints);
      if (inserted) {
        // Not cached: combine with the base subspace and look the orbit up.
        constraints.insert(constraints.end(), base_constraints_.begin(),
                           base_constraints_.end());
        it->second.rank = static_cast<uint8_t>(orbit_map_.Get(
            constraints, &it->second.query_elem, &it->second.store_elem));
      }
      const LocalMapValue &value = it->second;
      rank_lower_bound_self = std::max<int>(
          rank_lower_bound_self, std::popcount(mask) + 1 + value.rank);
      if (rank_lower_bound_self >= target_rank_lower_bound_) {
        // This branch proves the target; record one step. full_mask includes
        // the last dfs constraint (always part of `constraints` above), so
        // popcount(full_mask) + value.rank reproduces the bound on replay.
        const uint32_t full_mask = mask | (uint32_t(1) << (dfs_size - 1));
        proof->Append(static_cast<uint8_t>(dfs_size), full_mask,
                      static_cast<uint32_t>(value.query_elem),
                      static_cast<uint32_t>(value.store_elem));
        return rank_lower_bound_self;
      }
    }

    if (dfs_size == static_cast<std::size_t>(max_depth_)) {
      return rank_lower_bound_self;
    }

    int rank_lower_bound_child = std::numeric_limits<int>::max();
    for (int i = 0; i <= max_constraint_idx; ++i) {
      dfs_constraints->push_back(minimal_constraints_[i]);
      rank_lower_bound_child =
          std::min(rank_lower_bound_child,
                   Search(i, step_count, dfs_constraints, local_map, proof));
      dfs_constraints->pop_back();
      if (rank_lower_bound_child <= rank_lower_bound_self) {
        break;
      }
    }

    return std::max(rank_lower_bound_self, rank_lower_bound_child);
  }

  const Constraints<P, NA> &base_constraints_;
  const OrbitMap<Problem> &orbit_map_;
  int known_rank_lower_bound_ = 0;
  uint64_t step_limit_ = 0;
  size_t max_map_size_ = 0;
  int max_depth_ = 0;
  int target_rank_lower_bound_ = 0;
  std::vector<std::vector<uint32_t>> masks_;
  std::vector<Vec> minimal_constraints_;
};
