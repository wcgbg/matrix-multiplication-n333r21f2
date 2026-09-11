#pragma once

// 𝔽₂ primitives of the rank-one-span exclusion (see
// core/rank_lower_bound_rank_one_span.h for the method): wide bit rows and
// echelon bases, RREF subspace enumeration, the A-slice span in core
// coordinates with its enumeration cost, the engine-choice constants, and the
// v1 engine that sweeps every quotient subspace. Trusted: the verifier
// re-runs these to check a RankOneSpanProof.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "boost/unordered/unordered_flat_map.hpp"
#include "ng-log/logging.h"

#include "core/gf_vec.h"
#include "core/math_utils.h"
#include "core/rank_one_span_common.h" // RankOneSpanResult, saturating helpers
#include "core/tensor.h"

namespace rank_one_span_internal {

using rank_one_span_common::SatAdd;
using rank_one_span_common::SatMul;
using rank_one_span_common::SatPow2;

// Σ_{k=0..e} GaussianBinomial(dq, k)₂ — the number of subspaces of 𝔽₂^dq of
// dimension at most e — with saturating arithmetic (the values overflow uint64
// at modest dq). Requires e ≥ 0; e is capped at dq.
inline uint64_t SubspaceCount2(int dq, int e) {
  CHECK_GE(dq, 0);
  CHECK_GE(e, 0);
  e = std::min(e, dq);
  // gb[k] = GaussianBinomial(n, k) for the current n, via the q-Pascal
  // recurrence GB(n, k) = GB(n-1, k-1) + 2^k · GB(n-1, k).
  std::vector<uint64_t> gb(static_cast<size_t>(e) + 1, 0);
  gb[0] = 1;
  for (int n = 1; n <= dq; ++n) {
    for (int k = std::min(e, n); k >= 1; --k) {
      gb[k] = SatAdd(gb[k - 1], SatMul(SatPow2(k), gb[k]));
    }
  }
  uint64_t total = 0;
  for (int k = 0; k <= e; ++k) {
    total = SatAdd(total, gb[k]);
  }
  return total;
}

// Fixed-capacity 𝔽₂ row of up to NBits bits (BitVec caps at 64; the ambient
// core ρ_b·ρ_c can reach NB·NC = 81 for ⟨3,3,3⟩ and beyond for other
// problems). Plain word array; only the first `bits` positions are used at
// runtime.
template <std::size_t NBits> struct WideBits {
  static constexpr std::size_t kWords = (NBits + 63) / 64;
  std::array<uint64_t, kWords> w{};

  bool operator==(const WideBits &) const = default;

  void XorInPlace(const WideBits &o) {
    for (std::size_t i = 0; i < kWords; ++i) {
      w[i] ^= o.w[i];
    }
  }
  bool IsZero() const {
    for (uint64_t x : w) {
      if (x != 0) {
        return false;
      }
    }
    return true;
  }
  // Highest set bit index, or -1 if zero.
  int TopBit() const {
    for (int i = static_cast<int>(kWords) - 1; i >= 0; --i) {
      if (w[i] != 0) {
        return i * 64 + std::bit_width(w[i]) - 1;
      }
    }
    return -1;
  }
  bool GetBit(int i) const { return (w[i / 64] >> (i % 64)) & 1; }
  void SetBit(int i) { w[i / 64] |= uint64_t{1} << (i % 64); }
};

// Incremental 𝔽₂ linear basis over WideBits rows, kept in echelon order
// (strictly decreasing top bits). Reducing a vector top-down against such a
// basis zeroes it at every row's top bit and is linear, so residues are
// canonical coset representatives — no full Jordan pass needed.
template <std::size_t NBits> struct LinearBasis {
  // (top bit, row), sorted by top bit descending.
  std::vector<std::pair<int, WideBits<NBits>>> rows;

  int Dim() const { return static_cast<int>(rows.size()); }

  WideBits<NBits> Reduce(WideBits<NBits> v) const {
    for (const auto &[top, row] : rows) {
      if (v.GetBit(top)) {
        v.XorInPlace(row);
      }
    }
    return v;
  }

  // Insert v (reduced first); returns true iff the dimension grew.
  bool Insert(const WideBits<NBits> &v) {
    WideBits<NBits> r = Reduce(v);
    const int top = r.TopBit();
    if (top < 0) {
      return false;
    }
    auto it =
        std::lower_bound(rows.begin(), rows.end(), top,
                         [](const auto &row, int t) { return row.first > t; });
    rows.insert(it, {top, std::move(r)});
    return true;
  }
};

// Enumerate every k-dimensional subspace of 𝔽₂^dq exactly once, as RREF row
// sets (rows are dq-bit uint64 vectors with strictly decreasing pivots; every
// pivot column is zero in the other rows). Callback returns true to stop
// early; the function returns whether it was stopped.
// `first_pivot` >= 0 restricts the enumeration to subspaces whose RREF has
// that first (largest) pivot — a disjoint partition of the full enumeration
// over first_pivot in [k-1, dq-1], used to parallelize the family search.
template <class Cb>
bool ForEachSubspaceRREF(int dq, int k, const Cb &cb, int first_pivot = -1) {
  CHECK_LE(dq, 63);
  CHECK_GE(k, 1);
  if (k > dq) {
    return false;
  }
  if (first_pivot >= 0) {
    CHECK_LT(first_pivot, dq);
    CHECK_GE(first_pivot, k - 1);
  }
  std::vector<int> pivots(k);
  std::vector<uint64_t> rows(k);
  // Free slots: (row, bit position) pairs a given pivot choice leaves
  // unconstrained — positions below the row's own pivot that are not pivots.
  std::vector<std::pair<int, int>> slots;

  // Recurse over descending pivot choices; for each complete choice, sweep all
  // free-bit assignments.
  auto rec = [&](auto &&self, int row) -> bool {
    if (row == k) {
      slots.clear();
      for (int i = 0; i < k; ++i) {
        for (int pos = 0; pos < pivots[i]; ++pos) {
          if (std::find(pivots.begin(), pivots.end(), pos) == pivots.end()) {
            slots.push_back({i, pos});
          }
        }
      }
      CHECK_LT(slots.size(), 64u);
      for (uint64_t mask = 0; mask < (uint64_t{1} << slots.size()); ++mask) {
        for (int i = 0; i < k; ++i) {
          rows[i] = uint64_t{1} << pivots[i];
        }
        for (std::size_t s = 0; s < slots.size(); ++s) {
          if ((mask >> s) & 1) {
            rows[slots[s].first] |= uint64_t{1} << slots[s].second;
          }
        }
        if (cb(rows)) {
          return true;
        }
      }
      return false;
    }
    int hi = (row == 0 ? dq - 1 : pivots[row - 1] - 1);
    // Leave room for the remaining k - 1 - row pivots below.
    int lo = k - 1 - row;
    if (row == 0 && first_pivot >= 0) {
      hi = first_pivot;
      lo = first_pivot;
    }
    for (int p = hi; p >= lo; --p) {
      pivots[row] = p;
      if (self(self, row + 1)) {
        return true;
      }
    }
    return false;
  };
  return rec(rec, 0);
}

// The A-slice span of `tensor` in core coordinates: the B/C flattening-image
// bases (column-reversed RREF over 𝔽₂ with unique pivots), and V = the span of
// the A-slices expressed in the product of those bases. Because the bases are
// full RREF, the core coordinate of a slice is simply its value at the pivot
// positions: slice_core[s][t] = T[i][pb[s]][pc[t]], packed into a WideBits row
// at bit s·ρ_c + t.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
struct SliceSpanA {
  int rb = 0, rc = 0;     // core dims: B/C flattening ranks
  int rho = 0;            // dim V (the A-flattening rank)
  LinearBasis<NB * NC> v; // span of the A-slices, core coordinates
};

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
SliceSpanA<P, NA, NB, NC> BuildSliceSpanA(const Tensor<P, NA, NB, NC> &tensor) {
  static_assert(P == 2, "the family-search engine is F_2-only");
  const GF<P> zero = GF<P>::Zero();

  // B image: span of the fibers T[i][·][k]; C image: span of T[i][j][·].
  std::vector<GFVec<2, static_cast<int>(NB)>> b_rows;
  std::vector<GFVec<2, static_cast<int>(NC)>> c_rows;
  for (std::size_t i = 0; i < NA; ++i) {
    for (std::size_t k = 0; k < NC; ++k) {
      GFVec<2, static_cast<int>(NB)> f{};
      for (std::size_t j = 0; j < NB; ++j) {
        if (tensor[i][j][k] != zero) {
          f.Set(static_cast<int>(j), GF<P>::One());
        }
      }
      if (!f.IsZero()) {
        b_rows.push_back(f);
      }
    }
    for (std::size_t j = 0; j < NB; ++j) {
      GFVec<2, static_cast<int>(NC)> f{};
      for (std::size_t k = 0; k < NC; ++k) {
        if (tensor[i][j][k] != zero) {
          f.Set(static_cast<int>(k), GF<P>::One());
        }
      }
      if (!f.IsZero()) {
        c_rows.push_back(f);
      }
    }
  }
  SliceSpanA<P, NA, NB, NC> span;
  span.rb = GaussJordanEliminationF2<static_cast<int>(NB)>(&b_rows);
  span.rc = GaussJordanEliminationF2<static_cast<int>(NC)>(&c_rows);
  // Nonzero rows sit at the back (zero rows park at the front); each pivot is
  // that row's leading bit and its column is cleared in every other row.
  std::vector<int> pb(span.rb), pc(span.rc);
  for (int s = 0; s < span.rb; ++s) {
    pb[s] = b_rows[b_rows.size() - span.rb + s].LeadingNonzeroIdx();
  }
  for (int t = 0; t < span.rc; ++t) {
    pc[t] = c_rows[c_rows.size() - span.rc + t].LeadingNonzeroIdx();
  }
  for (std::size_t i = 0; i < NA; ++i) {
    WideBits<NB * NC> slice;
    for (int s = 0; s < span.rb; ++s) {
      for (int t = 0; t < span.rc; ++t) {
        if (tensor[i][pb[s]][pc[t]] != zero) {
          slice.SetBit(s * span.rc + t);
        }
      }
    }
    span.v.Insert(slice);
  }
  span.rho = span.v.Dim();
  return span;
}

// Enumeration cost for slicing along A at `target_rank`: number of nonzero
// core rank-one matrices plus the number of candidate subspaces, saturating.
// 0 when the exclusion is trivial (target_rank < ρ).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
uint64_t CostFromSpan(const SliceSpanA<P, NA, NB, NC> &span, int target_rank) {
  if (target_rank < span.rho) {
    return 0;
  }
  const uint64_t rank1_count =
      SatMul(SatPow2(span.rb) - 1, SatPow2(span.rc) - 1);
  const int dq = span.rb * span.rc - span.rho;
  return SatAdd(rank1_count, SubspaceCount2(dq, target_rank - span.rho));
}

// ---- The two decision engines --------------------------------------------
//
// Both decide the same predicate exactly — "does a rank-one-spanned U with
// V ⊆ U ⊆ core-ambient and dim U ≤ target exist?" — so which one runs affects
// only cost, never the verdict, and the proof needs no mode marker.
//
// The v1 engine (EnumerateAllSubspaces) sweeps every quotient subspace and is
// used whenever its Gaussian-binomial cost is at most
// kRankOneSpanV1CostThreshold — a constexpr, deliberately independent of the
// caller's budget, so certificates produced earlier keep verifying through the
// identical code path. Above the threshold the family search runs, with the
// caller's budget reinterpreted as a deterministic operation count.

inline constexpr uint64_t kRankOneSpanV1CostThreshold = 100'000'000;
// Family-search applicability guards (outside them: kOverBudget).
inline constexpr int kMaxFamilyK = 5;   // max quotient dims handled
inline constexpr int kMaxFamilyDq = 24; // occupancy arrays are 2^dq entries
inline constexpr int kMaxFamilySideDim = 10;

// Small fixed-capacity echelon basis over dq-bit keys (rows kept in strictly
// decreasing top-bit order, i.e. decreasing value; Reduce over that order
// yields canonical residues — same argument as LinearBasis). Trivially
// copyable on purpose: the family search copies bases in hot loops.
struct KeyBasis {
  std::array<uint64_t, 16> rows;
  int n = 0;

  int Dim() const { return n; }
  uint64_t Reduce(uint64_t v) const {
    for (int i = 0; i < n; ++i) {
      v = std::min(v, v ^ rows[i]);
    }
    return v;
  }
  bool Contains(uint64_t v) const { return Reduce(v) == 0; }
  bool Insert(uint64_t v) {
    v = Reduce(v);
    if (v == 0) {
      return false;
    }
    CHECK_LT(n, 16);
    int pos = n;
    while (pos > 0 && rows[pos - 1] < v) {
      rows[pos] = rows[pos - 1];
      --pos;
    }
    rows[pos] = v;
    ++n;
    return true;
  }
};

// The ambient positions that are no row's top bit, ascending: V's residues
// are supported exactly there, so they are the quotient coordinates.
template <std::size_t NBits>
std::vector<int> QuotientFreePositions(const LinearBasis<NBits> &v,
                                       int ambient_dim) {
  std::vector<int> free_pos;
  free_pos.reserve(ambient_dim - v.Dim());
  std::vector<bool> is_top(static_cast<std::size_t>(ambient_dim), false);
  for (const auto &row : v.rows) {
    is_top[row.first] = true;
  }
  for (int i = 0; i < ambient_dim; ++i) {
    if (!is_top[i]) {
      free_pos.push_back(i);
    }
  }
  return free_pos;
}

// The core rank-one matrix u·wᵀ as a row: bit s·rc + t is set iff u has bit
// s and w has bit t.
template <std::size_t NBits>
WideBits<NBits> RankOneRow(uint64_t u, uint64_t w, int rb, int rc) {
  WideBits<NBits> r;
  for (int s = 0; s < rb; ++s) {
    if ((u >> s) & 1) {
      for (int t = 0; t < rc; ++t) {
        if ((w >> t) & 1) {
          r.SetBit(s * rc + t);
        }
      }
    }
  }
  return r;
}

// The quotient key of a residue modulo V: its bits at the free positions
// (at most 63 of them; callers CHECK the width).
template <std::size_t NBits>
uint64_t KeyFromResidue(const WideBits<NBits> &resid,
                        const std::vector<int> &free_pos) {
  uint64_t key = 0;
  for (std::size_t i = 0; i < free_pos.size(); ++i) {
    if (resid.GetBit(free_pos[i])) {
      key |= uint64_t{1} << i;
    }
  }
  return key;
}

// The nonzero core rank-one matrices: those inside V feed the shared base
// span b0; with `keep_buckets` the rest are bucketed by their canonical
// quotient key.
template <std::size_t NBits> struct RankOneBuckets {
  LinearBasis<NBits> b0;
  boost::unordered_flat_map<uint64_t, std::vector<WideBits<NBits>>> buckets;
};

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
RankOneBuckets<NB * NC>
BucketCoreRankOnes(const SliceSpanA<P, NA, NB, NC> &span,
                   const std::vector<int> &free_pos, bool keep_buckets) {
  using Row = WideBits<NB * NC>;
  RankOneBuckets<NB * NC> rank_ones;
  for (uint64_t u = 1; u < (uint64_t{1} << span.rb); ++u) {
    for (uint64_t w = 1; w < (uint64_t{1} << span.rc); ++w) {
      const Row r = RankOneRow<NB * NC>(u, w, span.rb, span.rc);
      const Row resid = span.v.Reduce(r);
      if (resid.IsZero()) {
        rank_ones.b0.Insert(r);
      } else if (keep_buckets) {
        rank_ones.buckets[KeyFromResidue(resid, free_pos)].push_back(r);
      }
    }
  }
  return rank_ones;
}

// All 2^k keys of span{rows}, (*coset_keys)[0] = 0.
inline void ExpandCosetKeys(const std::vector<uint64_t> &rows,
                            std::vector<uint64_t> *coset_keys) {
  coset_keys->clear();
  coset_keys->push_back(0);
  for (uint64_t row : rows) {
    const std::size_t n = coset_keys->size();
    for (std::size_t i = 0; i < n; ++i) {
      coset_keys->push_back((*coset_keys)[i] ^ row);
    }
  }
}

// Whether b0 together with the rank-ones of the nonzero coset keys spans
// `want` dimensions: a count prefilter, then incremental insertion into
// *scratch.
template <std::size_t NBits>
bool IsRankOneSpannedCandidate(const std::vector<uint64_t> &coset_keys,
                               const RankOneBuckets<NBits> &rank_ones, int want,
                               LinearBasis<NBits> *scratch) {
  std::size_t count = 0;
  for (std::size_t i = 1; i < coset_keys.size(); ++i) {
    auto it = rank_ones.buckets.find(coset_keys[i]);
    if (it != rank_ones.buckets.end()) {
      count += it->second.size();
    }
  }
  if (rank_ones.b0.Dim() + static_cast<int>(count) < want) {
    return false;
  }
  scratch->rows.assign(rank_ones.b0.rows.begin(), rank_ones.b0.rows.end());
  for (std::size_t i = 1; i < coset_keys.size(); ++i) {
    auto it = rank_ones.buckets.find(coset_keys[i]);
    if (it == rank_ones.buckets.end()) {
      continue;
    }
    for (const WideBits<NBits> &r : it->second) {
      if (scratch->Insert(r) && scratch->Dim() == want) {
        return true;
      }
    }
  }
  return false;
}

// The v1 engine: enumerate every quotient subspace of dim ≤ target − ρ.
// (The caller has already applied the trivial target < ρ exclusion and the
// budget check.)
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
RankOneSpanResult EnumerateAllSubspaces(const SliceSpanA<P, NA, NB, NC> &span,
                                        int target_rank) {
  const int rb = span.rb, rc = span.rc, rho = span.rho;
  const int dq = rb * rc - rho;
  const int e = std::min(target_rank - rho, dq);

  const std::vector<int> free_pos = QuotientFreePositions(span.v, rb * rc);
  CHECK_EQ(static_cast<int>(free_pos.size()), dq);
  // With e = 0 the only candidate is U = V, so buckets are never consulted
  // and dq may exceed 63; with e ≥ 1 the budget check already bounds
  // 2^dq − 1 ≤ max_subspaces, but re-CHECK for the key width.
  if (e >= 1) {
    CHECK_LE(dq, 63);
  }
  const RankOneBuckets<NB * NC> rank_ones =
      BucketCoreRankOnes(span, free_pos, /*keep_buckets=*/e >= 1);

  // k = 0: U = V is rank-one spanned iff the in-V rank-ones span all of V.
  if (rank_ones.b0.Dim() == rho) {
    return RankOneSpanResult::kWitnessFound;
  }

  // k = 1..e: U = V + (lift of a k-dim key subspace W). U's rank-ones are the
  // buckets of W's 2^k − 1 nonzero keys plus the in-V ones; U is rank-one
  // spanned iff they span dim ρ + k. (A candidate whose rank-ones span only a
  // proper subspace U' ⊂ U is not a witness here, but U' ⊇ V is itself
  // enumerated at its own smaller k, so the sweep as a whole is complete.)
  std::vector<uint64_t> coset_keys;
  LinearBasis<NB * NC> scratch;
  for (int k = 1; k <= e; ++k) {
    const bool found =
        ForEachSubspaceRREF(dq, k, [&](const std::vector<uint64_t> &rows) {
          ExpandCosetKeys(rows, &coset_keys);
          return IsRankOneSpannedCandidate(coset_keys, rank_ones, rho + k,
                                           &scratch);
        });
    if (found) {
      return RankOneSpanResult::kWitnessFound;
    }
  }
  return RankOneSpanResult::kExcluded;
}

} // namespace rank_one_span_internal
