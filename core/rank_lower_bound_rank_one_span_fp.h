#pragma once

// Rank-one-span exclusion over a prime field 𝔽_P: the exhaustive "v1"
// engine of core/rank_lower_bound_rank_one_span.h without the 𝔽₂ bit packing
// and without the 𝔽₂-specific family search.
//
// Same predicate as the 𝔽₂ engine. Slice the tensor along A: the NA slices are
// NB×NC matrices whose span V (dim ρ, the A-flattening rank) sits inside the
// core B_img ⊗ C_img (rb·rc dims, the B- and C-flattening ranks). Then
//   rank(T) = min { dim U : V ⊆ U ⊆ core, U spanned by rank-≤1 matrices },
// so "no rank-one-spanned U ⊇ V of dim ≤ target" proves rank(T) ≥ target + 1.
// Any U ⊇ V is V + (a lift of a subspace W of the quotient core/V), and its
// rank-one elements are exactly the core rank-one matrices whose residue mod V
// lies in W. The engine buckets every core rank-one matrix u·wᵀ by the
// projective point of its residue, then sweeps every W of dim k ≤ target − ρ
// (RREF enumeration over 𝔽_P), gathers the buckets of W's points and tests
// whether they, with the rank-ones inside V, span ρ + k dimensions. A witness
// at some W means rank(T) ≤ ρ + k; no witness at any W means the exclusion.
// (A W whose rank-ones span only a proper subspace U' ⊂ U is not a witness at
// W, but U' ⊇ V is itself enumerated at its own smaller k, so the sweep is
// complete.)
//
// Written generically in P so that the P = 2 instantiation can be cross-checked
// against the trusted 𝔽₂ engine; the dispatchers in
// core/rank_lower_bound_rank_one_span.h route P ≠ 2 here.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#include "boost/unordered/unordered_flat_map.hpp"
#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/gf.h"
#include "core/gf_vec.h"
#include "core/math_utils.h"
#include "core/rank_one_span_common.h"
#include "core/tensor.h"

namespace rank_one_span_fp_internal {

using rank_one_span_common::SatAdd;
using rank_one_span_common::SatMul;
using rank_one_span_common::SatPow;

// Σ_{k=0..e} GaussianBinomial(dq, k)_P, saturating (q-Pascal recurrence
// GB(n, k) = GB(n-1, k-1) + P^k · GB(n-1, k)).
template <int P> uint64_t SubspaceCountP(int dq, int e) {
  CHECK_GE(dq, 0);
  CHECK_GE(e, 0);
  e = std::min(e, dq);
  std::vector<uint64_t> gb(static_cast<size_t>(e) + 1, 0);
  gb[0] = 1;
  for (int n = 1; n <= dq; ++n) {
    for (int k = std::min(e, n); k >= 1; --k) {
      gb[k] = SatAdd(gb[k - 1], SatMul(SatPow(P, k), gb[k]));
    }
  }
  uint64_t total = 0;
  for (int k = 0; k <= e; ++k) {
    total = SatAdd(total, gb[k]);
  }
  return total;
}

template <int P> uint8_t InvP(uint8_t a) {
  return GF<P>::Inverse(GF<P>{a}).value;
}

// A core vector of up to N digits (only the first `len` are used).
template <std::size_t N> using FpRow = std::array<uint8_t, N>;

// a -= c · b on the first len digits.
template <int P, std::size_t N>
void SubScaled(FpRow<N> &a, int c, const FpRow<N> &b, int len) {
  const int nc = (P - c % P) % P; // -c mod P
  if (nc == 0) {
    return;
  }
  for (int i = 0; i < len; ++i) {
    a[i] = static_cast<uint8_t>((a[i] + nc * b[i]) % P);
  }
}

template <std::size_t N> int TopNonzero(const FpRow<N> &r, int len) {
  for (int i = len - 1; i >= 0; --i) {
    if (r[i]) {
      return i;
    }
  }
  return -1;
}

// Fully reduced basis: each row's pivot is its highest nonzero digit, the
// pivot entry is 1 and the pivot column is zero in every other row. The
// residue of a vector (Reduce) therefore vanishes at every pivot position, so
// the quotient core/V is coordinatised by the non-pivot positions.
template <int P, std::size_t N> struct FpBasis {
  int len = 0;
  std::vector<std::pair<int, FpRow<N>>> rows; // (pivot, row)

  int Dim() const { return static_cast<int>(rows.size()); }

  FpRow<N> Reduce(FpRow<N> r) const {
    for (const auto &[piv, row] : rows) {
      if (r[piv]) {
        SubScaled<P, N>(r, r[piv], row, len);
      }
    }
    return r;
  }

  // Inserts r unless it lies in the span; returns whether the dim grew.
  bool Insert(FpRow<N> r) {
    r = Reduce(r);
    const int piv = TopNonzero(r, len);
    if (piv < 0) {
      return false;
    }
    const int inv = InvP<P>(r[piv]);
    for (int i = 0; i < len; ++i) {
      r[i] = static_cast<uint8_t>(r[i] * inv % P);
    }
    for (auto &[p, row] : rows) {
      if (row[piv]) {
        SubScaled<P, N>(row, row[piv], r, len);
      }
    }
    rows.emplace_back(piv, r);
    return true;
  }
};

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
struct SliceSpanFp {
  int rb = 0, rc = 0; // core dims: B/C flattening ranks
  int rho = 0;        // dim V (the A-flattening rank)
  FpBasis<P, NB * NC> v;
};

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
SliceSpanFp<P, NA, NB, NC>
BuildSliceSpanFp(const Tensor<P, NA, NB, NC> &tensor) {
  using F = GF<P>;
  const F zero = F::Zero();

  // B image: span of the fibers T[i][·][k]; C image: span of T[i][j][·].
  std::vector<GFVec<P, static_cast<int>(NB)>> b_rows;
  std::vector<GFVec<P, static_cast<int>(NC)>> c_rows;
  for (std::size_t i = 0; i < NA; ++i) {
    for (std::size_t k = 0; k < NC; ++k) {
      GFVec<P, static_cast<int>(NB)> f{};
      for (std::size_t j = 0; j < NB; ++j) {
        if (tensor[i][j][k] != zero) {
          f.Set(static_cast<int>(j), tensor[i][j][k]);
        }
      }
      if (!f.IsZero()) {
        b_rows.push_back(f);
      }
    }
    for (std::size_t j = 0; j < NB; ++j) {
      GFVec<P, static_cast<int>(NC)> f{};
      for (std::size_t k = 0; k < NC; ++k) {
        if (tensor[i][j][k] != zero) {
          f.Set(static_cast<int>(k), tensor[i][j][k]);
        }
      }
      if (!f.IsZero()) {
        c_rows.push_back(f);
      }
    }
  }
  SliceSpanFp<P, NA, NB, NC> span;
  if constexpr (P == 2) {
    span.rb = GaussJordanEliminationF2<static_cast<int>(NB)>(&b_rows);
    span.rc = GaussJordanEliminationF2<static_cast<int>(NC)>(&c_rows);
  } else {
    span.rb = GaussJordanEliminationFq<P, static_cast<int>(NB)>(&b_rows);
    span.rc = GaussJordanEliminationFq<P, static_cast<int>(NC)>(&c_rows);
  }
  // The elimination leaves the nonzero rows at the back in column-reversed
  // RREF with unit pivots and cleared pivot columns, so the coordinate of any
  // fiber on core basis row s is its entry at that row's pivot column.
  std::vector<int> pb(span.rb), pc(span.rc);
  for (int s = 0; s < span.rb; ++s) {
    pb[s] = b_rows[b_rows.size() - span.rb + s].LeadingNonzeroIdx();
  }
  for (int t = 0; t < span.rc; ++t) {
    pc[t] = c_rows[c_rows.size() - span.rc + t].LeadingNonzeroIdx();
  }
  span.v.len = span.rb * span.rc;
  for (std::size_t i = 0; i < NA; ++i) {
    FpRow<NB * NC> slice{};
    for (int s = 0; s < span.rb; ++s) {
      for (int t = 0; t < span.rc; ++t) {
        slice[s * span.rc + t] = tensor[i][pb[s]][pc[t]].value;
      }
    }
    span.v.Insert(slice);
  }
  span.rho = span.v.Dim();
  return span;
}

// Enumeration cost at `target_rank`: core rank-one matrices (projective u,
// all nonzero w) plus candidate quotient subspaces, saturating; 0 when the
// exclusion is trivial (target_rank < ρ).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
uint64_t CostFromSpanFp(const SliceSpanFp<P, NA, NB, NC> &span,
                        int target_rank) {
  if (target_rank < span.rho) {
    return 0;
  }
  const uint64_t u_count = (SatPow(P, span.rb) - 1) / (P - 1);
  const uint64_t rank1_count = SatMul(u_count, SatPow(P, span.rc) - 1);
  const int dq = span.rb * span.rc - span.rho;
  return SatAdd(rank1_count, SubspaceCountP<P>(dq, target_rank - span.rho));
}

// Base-P code of the projective normalisation (first nonzero digit 1) of the
// residue's digits at the free positions.
template <int P, std::size_t N>
uint64_t QuotientKey(const FpRow<N> &resid, const std::vector<int> &free_pos) {
  int scale = 0;
  uint64_t key = 0;
  for (int i = static_cast<int>(free_pos.size()) - 1; i >= 0; --i) {
    key = key * P + resid[free_pos[i]];
  }
  // Normalise: find the first nonzero digit (lowest free index).
  for (int i = 0; i < static_cast<int>(free_pos.size()); ++i) {
    if (resid[free_pos[i]]) {
      scale = InvP<P>(resid[free_pos[i]]);
      break;
    }
  }
  if (scale == 1 || scale == 0) {
    return key;
  }
  key = 0;
  for (int i = static_cast<int>(free_pos.size()) - 1; i >= 0; --i) {
    key = key * P + resid[free_pos[i]] * scale % P;
  }
  return key;
}

// The ambient positions that are no row's pivot, ascending: the quotient
// coordinates.
template <int P, std::size_t N>
std::vector<int> QuotientFreePositionsFp(const FpBasis<P, N> &v, int len) {
  std::vector<int> free_pos;
  std::vector<bool> is_piv(static_cast<std::size_t>(len), false);
  for (const auto &[piv, row] : v.rows) {
    is_piv[piv] = true;
  }
  for (int i = 0; i < len; ++i) {
    if (!is_piv[i]) {
      free_pos.push_back(i);
    }
  }
  return free_pos;
}

// Base-P digits of `code`, least significant first, in the first n entries.
template <int P, std::size_t N>
std::array<int, N> DecodeBaseP(uint64_t code, int n) {
  std::array<int, N> digits{};
  for (int s = 0; s < n; ++s) {
    digits[s] = static_cast<int>(code % P);
    code /= P;
  }
  return digits;
}

// A nonzero u is the representative of its projective point iff its lowest
// nonzero digit is 1.
template <std::size_t N>
bool IsProjectiveRepresentative(const std::array<int, N> &u, int n) {
  for (int s = 0; s < n; ++s) {
    if (u[s]) {
      return u[s] == 1;
    }
  }
  return false;
}

// The rank-one matrix u·wᵀ as a core row: entry s·rc + t = u[s]·w[t].
template <int P, std::size_t NB, std::size_t NC>
FpRow<NB * NC> OuterProductRow(const std::array<int, NB> &u,
                               const std::array<int, NC> &w, int rb, int rc) {
  FpRow<NB * NC> r{};
  for (int s = 0; s < rb; ++s) {
    if (u[s]) {
      for (int t = 0; t < rc; ++t) {
        r[s * rc + t] = static_cast<uint8_t>(u[s] * w[t] % P);
      }
    }
  }
  return r;
}

// The core rank-one matrices u·wᵀ (u projective, w any nonzero): those
// inside V feed the base span b0; with `keep_buckets` the rest are bucketed
// by their projective quotient key.
template <int P, std::size_t N> struct RankOneBucketsFp {
  FpBasis<P, N> b0;
  boost::unordered_flat_map<uint64_t, std::vector<FpRow<N>>> buckets;
};

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
RankOneBucketsFp<P, NB * NC>
BucketCoreRankOnesFp(const SliceSpanFp<P, NA, NB, NC> &span,
                     const std::vector<int> &free_pos, bool keep_buckets) {
  using Row = FpRow<NB * NC>;
  const int rb = span.rb, rc = span.rc, len = rb * rc;
  RankOneBucketsFp<P, NB * NC> rank_ones;
  rank_ones.b0.len = len;
  const uint64_t u_total = SatPow(P, rb), w_total = SatPow(P, rc);
  for (uint64_t ucode = 1; ucode < u_total; ++ucode) {
    const std::array<int, NB> u = DecodeBaseP<P, NB>(ucode, rb);
    if (!IsProjectiveRepresentative(u, rb)) {
      continue; // one representative per projective point
    }
    for (uint64_t wcode = 1; wcode < w_total; ++wcode) {
      const std::array<int, NC> w = DecodeBaseP<P, NC>(wcode, rc);
      const Row r = OuterProductRow<P, NB, NC>(u, w, rb, rc);
      const Row resid = span.v.Reduce(r);
      if (TopNonzero(resid, len) < 0) {
        rank_ones.b0.Insert(r);
      } else if (keep_buckets) {
        rank_ones.buckets[QuotientKey<P>(resid, free_pos)].push_back(r);
      }
    }
  }
  return rank_ones;
}

// The dq-bit masks with exactly k bits: the pivot sets of the k-dim RREF
// subspaces of 𝔽_P^dq. The masks are 32-bit, so dq < 32 (the cost estimate
// keeps every affordable quotient far below that; the key-width CHECK of the
// engine alone would admit dq up to 39 for P = 3).
inline std::vector<uint32_t> PivotSetsOfSize(int dq, int k) {
  CHECK_LT(dq, 32) << "quotient dimension too large for 32-bit pivot masks";
  std::vector<uint32_t> pivot_sets;
  for (uint32_t mask = 0; mask < (1u << dq); ++mask) {
    if (__builtin_popcount(mask) == k) {
      pivot_sets.push_back(mask);
    }
  }
  return pivot_sets;
}

// The shape of the k-row RREF matrices with the given pivot set: pivot
// columns ascending, and the (row, col) entries that hold free digits (right
// of the row's pivot, in no pivot column).
struct RrefShapeFp {
  std::vector<int> piv;
  std::vector<std::pair<int, int>> free_entries;
};

inline RrefShapeFp RrefShapeFromPivots(uint32_t pivots, int dq, int k) {
  RrefShapeFp shape;
  for (int c = 0; c < dq; ++c) {
    if (pivots >> c & 1) {
      shape.piv.push_back(c);
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int c = shape.piv[i] + 1; c < dq; ++c) {
      if (!(pivots >> c & 1)) {
        shape.free_entries.push_back({i, c});
      }
    }
  }
  return shape;
}

// Base-P key of Σ_i coef[i]·rows[i], digit dq − 1 most significant.
template <int P>
uint64_t KeyOfCombinationFp(const std::vector<int> &coef,
                            const std::vector<std::vector<int>> &rows,
                            int dq) {
  uint64_t key = 0;
  for (int c = dq - 1; c >= 0; --c) {
    int acc = 0;
    for (std::size_t i = 0; i < coef.size(); ++i) {
      acc += coef[i] * rows[i][c];
    }
    key = key * P + acc % P;
  }
  return key;
}

// Appends to *hit the buckets of W's projective points (Σ c_i·row_i with the
// first nonzero c_i equal to 1) and returns their total size. *coef is
// scratch of size k.
template <int P, std::size_t N>
std::size_t GatherBucketsFp(const std::vector<std::vector<int>> &rows, int dq,
                            const RankOneBucketsFp<P, N> &rank_ones,
                            std::vector<int> *coef,
                            std::vector<const std::vector<FpRow<N>> *> *hit) {
  const int k = static_cast<int>(rows.size());
  std::size_t count = 0;
  for (int first = 0; first < k; ++first) {
    // coef[first] = 1, coef[<first] = 0, coef[>first] free.
    const int rest = k - first - 1;
    uint64_t combos = 1;
    for (int i = 0; i < rest; ++i) {
      combos *= P;
    }
    for (uint64_t cc = 0; cc < combos; ++cc) {
      std::fill(coef->begin(), coef->end(), 0);
      (*coef)[first] = 1;
      uint64_t t = cc;
      for (int i = first + 1; i < k; ++i) {
        (*coef)[i] = static_cast<int>(t % P);
        t /= P;
      }
      auto it = rank_ones.buckets.find(KeyOfCombinationFp<P>(*coef, rows, dq));
      if (it != rank_ones.buckets.end()) {
        count += it->second.size();
        hit->push_back(&it->second);
      }
    }
  }
  return count;
}

// Whether b0 together with the rows of the hit buckets spans `want`
// dimensions (incremental insertion into *scratch).
template <int P, std::size_t N>
bool SpansDimensionFp(const RankOneBucketsFp<P, N> &rank_ones,
                      const std::vector<const std::vector<FpRow<N>> *> &hit,
                      int want, FpBasis<P, N> *scratch) {
  scratch->rows.assign(rank_ones.b0.rows.begin(), rank_ones.b0.rows.end());
  for (const auto *bucket : hit) {
    for (const FpRow<N> &r : *bucket) {
      if (scratch->Insert(r) && scratch->Dim() == want) {
        return true;
      }
    }
  }
  return false;
}

// Every k-dim RREF subspace W with the given pivot set (an odometer over the
// free digits, least significant first): count the rank-ones on W's
// projective points, then test the span. True iff a witness was found.
template <int P, std::size_t N>
bool SweepPivotSetFp(uint32_t pivots, int k, int dq, int rho,
                     const RankOneBucketsFp<P, N> &rank_ones) {
  const RrefShapeFp shape = RrefShapeFromPivots(pivots, dq, k);
  const int nfree = static_cast<int>(shape.free_entries.size());
  std::vector<int> digit(nfree, 0);
  std::vector<std::vector<int>> rows(k, std::vector<int>(dq, 0));
  std::vector<int> coef(k);
  std::vector<const std::vector<FpRow<N>> *> hit;
  FpBasis<P, N> scratch;
  scratch.len = rank_ones.b0.len;
  uint64_t total = 1;
  for (int f = 0; f < nfree; ++f) {
    total *= P;
  }
  for (uint64_t n = 0; n < total; ++n) {
    for (int i = 0; i < k; ++i) {
      std::fill(rows[i].begin(), rows[i].end(), 0);
      rows[i][shape.piv[i]] = 1;
    }
    for (int f = 0; f < nfree; ++f) {
      rows[shape.free_entries[f].first][shape.free_entries[f].second] =
          digit[f];
    }
    hit.clear();
    const std::size_t count =
        GatherBucketsFp<P, N>(rows, dq, rank_ones, &coef, &hit);
    if (rank_ones.b0.Dim() + static_cast<int>(count) >= rho + k &&
        SpansDimensionFp(rank_ones, hit, rho + k, &scratch)) {
      return true;
    }
    for (int f = 0; f < nfree; ++f) {
      if (++digit[f] < P) {
        break;
      }
      digit[f] = 0;
    }
  }
  return false;
}

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
RankOneSpanResult
EnumerateAllSubspacesFp(const SliceSpanFp<P, NA, NB, NC> &span,
                        int target_rank, bool parallel) {
  const int rb = span.rb, rc = span.rc, rho = span.rho;
  const int len = rb * rc;
  const int dq = len - rho;
  const int e = std::min(target_rank - rho, dq);

  const std::vector<int> free_pos = QuotientFreePositionsFp(span.v, len);
  CHECK_EQ(static_cast<int>(free_pos.size()), dq);
  if (e >= 1) {
    CHECK_LT(SatPow(P, dq), uint64_t{1} << 62) << "quotient key too wide";
  }
  const RankOneBucketsFp<P, NB * NC> rank_ones =
      BucketCoreRankOnesFp(span, free_pos, /*keep_buckets=*/e >= 1);

  // k = 0: U = V is rank-one spanned iff the in-V rank-ones span all of V.
  if (rank_ones.b0.Dim() == rho) {
    return RankOneSpanResult::kWitnessFound;
  }

  // k = 1..e: W = a k-dim subspace of 𝔽_P^dq in RREF (pivot columns
  // ascending, unit pivots, zeros in the other rows' pivot columns, free
  // digits elsewhere); its projective points are Σ c_i·row_i with the first
  // nonzero c_i equal to 1. One task per pivot set; a running task finishes
  // its pivot set even after another task found a witness.
  std::atomic<bool> found{false};
  for (int k = 1; k <= e && !found.load(); ++k) {
    const std::vector<uint32_t> pivot_sets = PivotSetsOfSize(dq, k);
    auto sweep = [&](uint32_t pivots) {
      if (found.load(std::memory_order_relaxed)) {
        return;
      }
      if (SweepPivotSetFp<P, NB * NC>(pivots, k, dq, rho, rank_ones)) {
        found.store(true, std::memory_order_relaxed);
      }
    };
    if (parallel) {
      tbb::parallel_for(size_t{0}, pivot_sets.size(),
                        [&](size_t i) { sweep(pivot_sets[i]); });
    } else {
      for (uint32_t pivots : pivot_sets) {
        sweep(pivots);
        if (found.load()) {
          break;
        }
      }
    }
  }
  return found.load() ? RankOneSpanResult::kWitnessFound
                      : RankOneSpanResult::kExcluded;
}

} // namespace rank_one_span_fp_internal

// Decide whether any rank-one-spanned U with V ⊆ U ⊆ B_img ⊗ C_img has
// dim U ≤ target_rank (V = span of the A-slices). kExcluded proves
// rank(tensor) ≥ target_rank + 1; kOverBudget (cost above `max_subspaces`)
// makes no claim. Deterministic verdict; `parallel` only spreads the sweep.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
RankOneSpanResult
RankOneSpanExcludeFpA(const Tensor<P, NA, NB, NC> &tensor, int target_rank,
                      uint64_t max_subspaces, bool parallel = false) {
  using namespace rank_one_span_fp_internal;
  const auto span = BuildSliceSpanFp<P, NA, NB, NC>(tensor);
  if (target_rank < span.rho) {
    return RankOneSpanResult::kExcluded;
  }
  if (CostFromSpanFp(span, target_rank) > max_subspaces) {
    return RankOneSpanResult::kOverBudget;
  }
  return EnumerateAllSubspacesFp<P, NA, NB, NC>(span, target_rank, parallel);
}

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
uint64_t RankOneSpanCostFpA(const Tensor<P, NA, NB, NC> &tensor,
                            int target_rank) {
  using namespace rank_one_span_fp_internal;
  return CostFromSpanFp(BuildSliceSpanFp<P, NA, NB, NC>(tensor), target_rank);
}
