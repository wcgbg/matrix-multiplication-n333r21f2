#pragma once

// Rank-one-span exclusion lower bound (𝔽₂ only).
//
// Classical characterization: for T ∈ 𝔽^a ⊗ 𝔽^b ⊗ 𝔽^c, let V be the span of
// the a A-slices, each read as a matrix in 𝔽^{b×c}. Then
//
//   rank(T) = min{ dim U : V ⊆ U ⊆ 𝔽^{b×c}, U spanned by matrices of rank ≤ 1
//   }.
//
// (≤: the slices of Σ u_i ⊗ v_i ⊗ w_i lie in span{v_i w_iᵀ}. ≥: pick a basis
// of rank-one matrices R_1..R_d of U, express slice i as Σ_s λ_{is} R_s, and
// Σ_s (Σ_i λ_{is} e_i) ⊗ v_s ⊗ w_s reproduces T with d terms.) Moreover U may
// be assumed inside B_img ⊗ C_img, the product of the flattening images:
// projecting any U onto it preserves containment of V, being rank-one spanned,
// and can only shrink the dimension.
//
// So "rank(T) ≤ r" is decided exactly by enumerating every subspace U with
// V ⊆ U ⊆ B_img ⊗ C_img and dim U ≤ r — one per subspace of the quotient
// (dimension dq = ρ_b·ρ_c − ρ, where ρ = dim V) of dimension ≤ e = r − ρ,
// i.e. Σ_{k≤e} GaussianBinomial(dq, k)₂ candidates — and checking whether any
// is spanned by its rank-one elements. If none is, rank(T) ≥ r + 1 is proved.
//
// This is only affordable when e is small (the target rank barely exceeds a
// flattening rank) — exactly the highly constrained orbits where the
// backtracking DFS saturates. Cost is estimated up front (rank-one count plus
// subspace count, saturating) and compared against a caller budget, so callers
// can safely attempt the technique on every orbit.
//
// Bucketing: every nonzero rank-one matrix u·wᵀ of the core (there are
// (2^ρ_b − 1)(2^ρ_c − 1)) is reduced modulo V to a canonical dq-bit quotient
// key. A candidate U = V + W contains exactly the rank-ones whose key lies in
// the corresponding k-dim key subspace, so its rank-one set is the union of 2^k
// buckets and almost all candidates are rejected by a constant-time count.
//
// Trusted: the verifier re-runs this exclusion to check a RankOneSpanProof
// (recompute-style, like flatten and forced product; no sidecar data).
//
// The 𝔽₂ primitives and the v1 engine live in core/rank_one_span_f2.h, the
// family search (v2 engine) in core/rank_one_span_family_search.h, and the
// exhaustive engine for odd primes in core/rank_lower_bound_rank_one_span_fp.h;
// this header holds the public entry points.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "core/rank_lower_bound_flatten.h"          // CyclicTranspose
#include "core/rank_lower_bound_rank_one_span_fp.h" // the P ≠ 2 engine
#include "core/rank_one_span_common.h"              // RankOneSpanResult
#include "core/rank_one_span_f2.h"
#include "core/rank_one_span_family_search.h"
#include "core/tensor.h"

// Decide whether any rank-one-spanned subspace U with V ⊆ U ⊆ B_img ⊗ C_img
// has dim U ≤ target_rank, where V is the span of the A-slices of `tensor`
// (see the header comment). kExcluded proves rank(tensor) ≥ target_rank + 1.
//
// `max_subspaces` is the operation budget: the v1 engine charges its
// enumeration size against it up front, the v2 family search counts operations
// as it goes. Engine choice depends only on the tensor and target (see
// kRankOneSpanV1CostThreshold), so a proof produced under any sufficient
// budget re-verifies under any other sufficient budget through the identical
// path. kOverBudget makes no claim.
//
// `parallel` lets the family search use TBB internally (prover-side only —
// the verdict is unaffected, but op counts and the exact over-budget cutoff
// become timing dependent, so the verifier keeps the default false).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
RankOneSpanResult RankOneSpanExcludeA(const Tensor<P, NA, NB, NC> &tensor,
                                      int target_rank, uint64_t max_subspaces,
                                      bool parallel = false) {
  if constexpr (P != 2) {
    // Prime fields other than 𝔽₂: the exhaustive engine only (no family
    // search), see core/rank_lower_bound_rank_one_span_fp.h.
    return RankOneSpanExcludeFpA<P, NA, NB, NC>(tensor, target_rank,
                                                max_subspaces, parallel);
  } else {
    using namespace rank_one_span_internal;

    const SliceSpanA<P, NA, NB, NC> span =
        BuildSliceSpanA<P, NA, NB, NC>(tensor);
    // Any U ⊇ V has dim ≥ ρ, so a target below ρ is the (trivial) flattening
    // bound. This also covers target_rank < 0.
    if (target_rank < span.rho) {
      return RankOneSpanResult::kExcluded;
    }
    const uint64_t v1_cost = CostFromSpan(span, target_rank);
    if (v1_cost <= kRankOneSpanV1CostThreshold) {
      if (v1_cost > max_subspaces) {
        return RankOneSpanResult::kOverBudget;
      }
      return EnumerateAllSubspaces<P, NA, NB, NC>(span, target_rank);
    }
    uint64_t ops = 0;
    const RankOneSpanResult result = RankOneSpanFamilySearch<P, NA, NB, NC>(
        span, target_rank, max_subspaces, &ops, parallel);
    if (result != RankOneSpanResult::kOverBudget) {
      return result;
    }
    if (v1_cost <= max_subspaces) {
      return EnumerateAllSubspaces<P, NA, NB, NC>(span, target_rank);
    }
    return RankOneSpanResult::kOverBudget;
  }
}

// Up-front enumeration cost of RankOneSpanExcludeA (saturating); lets callers
// budget before running, and the prover pick the cheapest slicing axis.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
uint64_t RankOneSpanCostA(const Tensor<P, NA, NB, NC> &tensor,
                          int target_rank) {
  if constexpr (P != 2) {
    return RankOneSpanCostFpA<P, NA, NB, NC>(tensor, target_rank);
  } else {
    return rank_one_span_internal::CostFromSpan(
        rank_one_span_internal::BuildSliceSpanA<P, NA, NB, NC>(tensor),
        target_rank);
  }
}

// Prover wrapper: try the three cyclic positions in ascending expected cost
// until one decides. Returns (result, axis) with axis the number of
// CyclicTranspose applications before slicing along the first axis — the same
// convention as ForcedProductProof.projection_type, and what
// RankOneSpanProof.slice_axis records. Any single axis is a complete decision
// procedure, so the first decided axis is the answer; an axis whose engines
// are over budget just falls through to the next.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
std::pair<RankOneSpanResult, int>
RankOneSpanExclude(const Tensor<P, NA, NB, NC> &tensor, int target_rank,
                   uint64_t max_subspaces, bool parallel = false) {
  using namespace rank_one_span_internal;
  const Tensor<P, NB, NC, NA> t1 = CyclicTranspose<P, NA, NB, NC>(tensor);
  const Tensor<P, NC, NA, NB> t2 = CyclicTranspose<P, NB, NC, NA>(t1);
  // Order axes by (v1 cost, quotient dim, axis): the v1-affordable ones come
  // first, then the ones with the smallest family-search quotient.
  struct AxisInfo {
    uint64_t v1_cost;
    int dq;
    int axis;
  };
  std::array<AxisInfo, 3> order;
  if constexpr (P == 2) {
    const auto s0 = BuildSliceSpanA<P, NA, NB, NC>(tensor);
    const auto s1 = BuildSliceSpanA<P, NB, NC, NA>(t1);
    const auto s2 = BuildSliceSpanA<P, NC, NA, NB>(t2);
    order[0] = {CostFromSpan(s0, target_rank), s0.rb * s0.rc - s0.rho, 0};
    order[1] = {CostFromSpan(s1, target_rank), s1.rb * s1.rc - s1.rho, 1};
    order[2] = {CostFromSpan(s2, target_rank), s2.rb * s2.rc - s2.rho, 2};
  } else {
    using namespace rank_one_span_fp_internal;
    const auto s0 = BuildSliceSpanFp<P, NA, NB, NC>(tensor);
    const auto s1 = BuildSliceSpanFp<P, NB, NC, NA>(t1);
    const auto s2 = BuildSliceSpanFp<P, NC, NA, NB>(t2);
    order[0] = {CostFromSpanFp(s0, target_rank), s0.rb * s0.rc - s0.rho, 0};
    order[1] = {CostFromSpanFp(s1, target_rank), s1.rb * s1.rc - s1.rho, 1};
    order[2] = {CostFromSpanFp(s2, target_rank), s2.rb * s2.rc - s2.rho, 2};
  }
  std::sort(order.begin(), order.end(),
            [](const AxisInfo &a, const AxisInfo &b) {
              return std::tie(a.v1_cost, a.dq, a.axis) <
                     std::tie(b.v1_cost, b.dq, b.axis);
            });
  for (const AxisInfo &info : order) {
    RankOneSpanResult result;
    if (info.axis == 0) {
      result = RankOneSpanExcludeA<P, NA, NB, NC>(tensor, target_rank,
                                                  max_subspaces, parallel);
    } else if (info.axis == 1) {
      result = RankOneSpanExcludeA<P, NB, NC, NA>(t1, target_rank,
                                                  max_subspaces, parallel);
    } else {
      result = RankOneSpanExcludeA<P, NC, NA, NB>(t2, target_rank,
                                                  max_subspaces, parallel);
    }
    if (result != RankOneSpanResult::kOverBudget) {
      return {result, info.axis};
    }
  }
  return {RankOneSpanResult::kOverBudget, 0};
}
