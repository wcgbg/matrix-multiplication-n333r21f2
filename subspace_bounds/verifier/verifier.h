#pragma once

// Certificate verifier. Port of the matrix-mult
// proof_verifier/rank_lower_bound_verifier.h, generalized over a Problem.
//
// VerifyRankLowerBound mirrors the DP driver (rank_lower_bound_computer.h): it
// sweeps subspace dimensions from NA down to 0, and at each dimension (1)
// checks every orbit's proof against a map holding only the already-verified,
// strictly-larger dimensions, then (2) inserts that dimension's claimed bounds.
// By induction the final empty-constraint orbit's bound is sound. The five
// proof types are each re-checked independently:
//   - flatten/forced product: recompute the technique on the constrained
//   tensor;
//   - rank-one-span: re-run the exhaustive exclusion on the recorded slicing
//     axis at rank_lower_bound - 1 (core/rank_lower_bound_rank_one_span.h);
//   - degenerate: map the one-larger subspace to its canonical representative
//   via
//     the recorded witness and look up its bound;
//   - backtracking: replay the orbit's DFS trace, looked up by index in the
//     single per-certificate archive (backtracking_verifier.h).
//
// Unlike the prover (which uses the square-root OrbitMap), the verifier keeps a
// plain canonical→rank map — one entry per orbit, no Store-image expansion. The
// certificate records the full (query_elem, store_elem) witness per step, so
// the verifier maps a query straight to the orbit representative
// (CanonicalFromWitness below), the same shape as the matrix verifier's
// constraints_to_rank_lower_bound.

#include <format>
#include <string>
#include <vector>

#include "boost/unordered/unordered_flat_map.hpp"
#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/backtracking_proof.h"
#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/rank_lower_bound_flatten.h"
#include "core/rank_lower_bound_forced_product.h"
#include "core/rank_lower_bound_rank_one_span.h"
#include "core/tensor.h"
#include "subspace_bounds/verifier/backtracking_verifier.h"

// RankMap and CanonicalFromWitness are defined in backtracking_verifier.h (the
// lower-level header, included above).

// FlattenMatrixProof: recompute the flattening bound and check it justifies the
// claim.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void VerifyFlattenProof(const Tensor<P, NA, NB, NC> &tensor,
                        int rank_lower_bound) {
  const int computed =
      RankLowerBoundFlatten<P, NA, NB, NC>(tensor, rank_lower_bound);
  CHECK_GE(computed, rank_lower_bound);
}

// ForcedProductProof: recompute the forced-product bound on the cyclic position
// recorded in projection_type. Mirrors RankLowerBoundForcedProduct in the
// computer.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void VerifyForcedProductProof(
    const Tensor<P, NA, NB, NC> &tensor, int rank_lower_bound,
    const pb::ForcedProductProof &forced_product_proof) {
  const uint32_t proj = forced_product_proof.projection_type();
  int computed = 0;
  if (proj == 0) {
    computed = RankLowerBoundForcedProductA<P, NA, NB, NC>(tensor);
  } else if (proj == 1) {
    Tensor<P, NB, NC, NA> t1 = CyclicTranspose<P, NA, NB, NC>(tensor);
    computed = RankLowerBoundForcedProductA<P, NB, NC, NA>(t1);
  } else if (proj == 2) {
    Tensor<P, NB, NC, NA> t1 = CyclicTranspose<P, NA, NB, NC>(tensor);
    Tensor<P, NC, NA, NB> t2 = CyclicTranspose<P, NB, NC, NA>(t1);
    computed = RankLowerBoundForcedProductA<P, NC, NA, NB>(t2);
  } else {
    LOG(FATAL) << "Invalid projection_type: " << proj;
  }
  CHECK_GE(computed, rank_lower_bound);
}

// RankOneSpanProof: re-run the rank-one-span exclusion on the recorded axis at
// target = rank_lower_bound - 1. A single exclusion at that target covers all
// ranks ≤ target (every candidate subspace of smaller dimension is enumerated
// too), so no climb replay is needed. The budget cap fails closed: a
// certificate whose recompute would be absurdly expensive is rejected
// (kOverBudget trips the CHECK), never accepted unverified. Invariant: this
// constant must dominate any prover budget used to produce a shipped
// certificate — the engines are deterministic, so a proof produced within
// budget B re-verifies iff this constant is ≥ the operations actually spent.
//
// The re-run below uses the parallel engine: on the ACCEPT path every worker
// completes its full share (no witness, no over-budget), so the summed op
// count — and hence acceptance — is independent of scheduling; the early-exit
// paths (witness found, budget tripped) only ever REJECT, where timing
// dependence is harmless.
inline constexpr uint64_t kRankOneSpanVerifierBudget = 40'000'000'000'000;

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void VerifyRankOneSpanProof(const Tensor<P, NA, NB, NC> &tensor,
                            int rank_lower_bound,
                            const pb::RankOneSpanProof &rank_one_span_proof) {
  const uint32_t axis = rank_one_span_proof.slice_axis();
  const int target = rank_lower_bound - 1;
  RankOneSpanResult result = RankOneSpanResult::kOverBudget;
  if (axis == 0) {
    result = RankOneSpanExcludeA<P, NA, NB, NC>(
        tensor, target, kRankOneSpanVerifierBudget, /*parallel=*/true);
  } else if (axis == 1) {
    Tensor<P, NB, NC, NA> t1 = CyclicTranspose<P, NA, NB, NC>(tensor);
    result = RankOneSpanExcludeA<P, NB, NC, NA>(
        t1, target, kRankOneSpanVerifierBudget, /*parallel=*/true);
  } else if (axis == 2) {
    Tensor<P, NB, NC, NA> t1 = CyclicTranspose<P, NA, NB, NC>(tensor);
    Tensor<P, NC, NA, NB> t2 = CyclicTranspose<P, NB, NC, NA>(t1);
    result = RankOneSpanExcludeA<P, NC, NA, NB>(
        t2, target, kRankOneSpanVerifierBudget, /*parallel=*/true);
  } else {
    LOG(FATAL) << "Invalid slice_axis: " << axis;
  }
  CHECK(result == RankOneSpanResult::kExcluded)
      << "rank-one-span exclusion did not hold at target " << target
      << " on axis " << axis
      << (result == RankOneSpanResult::kOverBudget ? " (over budget)" : "");
}

// DegenerateProof: rebuild the one-larger subspace (base + extra_constraint),
// confirm it stays independent, map it to its canonical representative via the
// witness, and look up its bound. Constraining can only lower rank, so that
// bound is a valid bound here.
template <class Problem>
void VerifyDegenerateProof(
    const Constraints<Problem::kP, Problem::kNA> &constraints,
    int rank_lower_bound, const pb::DegenerateProof &degenerate_proof,
    const typename Problem::SymmetryGroup &group, const RankMap<Problem> &map) {
  constexpr int NA = Problem::kNA;
  constexpr int P = Problem::kP;
  using QueryElem = typename Problem::SymmetryGroup::QuerySet::Elem;
  using StoreElem = typename Problem::SymmetryGroup::StoreSet::Elem;

  Constraints<P, NA> extended = constraints;
  extended.push_back(
      DecodeGFVec<P, NA>(degenerate_proof.extra_constraint()));
  CHECK((IsLinearIndependentRREF<P, NA>(extended)));
  CHECK_EQ(extended.size(), constraints.size() + 1);

  const auto query_elem =
      static_cast<QueryElem>(degenerate_proof.transformation().query_elem());
  const auto store_elem =
      static_cast<StoreElem>(degenerate_proof.transformation().store_elem());
  const Constraints<P, NA> canonical =
      CanonicalFromWitness<Problem>(group, extended, query_elem, store_elem);
  const auto it = map.find(canonical);
  CHECK(it != map.end()) << "degenerate: canonical orbit not found in map";
  CHECK_GE(it->second, rank_lower_bound);
}

// BacktrackingProof: pull the orbit's trace from the loaded archive, check its
// length, and replay it.
template <class Problem>
void VerifyBacktrackingProof(
    const Constraints<Problem::kP, Problem::kNA> &constraints,
    int rank_lower_bound, const pb::RankLowerBoundProof &proof_proto,
    int orbit_index, const typename Problem::SymmetryGroup &group,
    const RankMap<Problem> &map, const BacktrackingProofArchive &archive) {
  const BacktrackingProof proof = archive.Get(orbit_index);
  CHECK_EQ(proof_proto.backtracking_proof().proof_size(), proof.Size());
  RankLowerBoundBacktrackingVerifier<Problem>::Verify(
      constraints, rank_lower_bound, proof, group, map);
}

// Verify a single orbit's proof against the (larger-dimension) map.
template <class Problem>
void VerifyOne(const pb::ConstrainedTensor &rt,
               const typename Problem::SymmetryGroup &group,
               const RankMap<Problem> &map,
               const BacktrackingProofArchive &archive) {
  constexpr int NA = Problem::kNA;
  constexpr int NB = Problem::kNB;
  constexpr int NC = Problem::kNC;
  constexpr int P = Problem::kP;

  const Constraints<P, NA> constraints =
      ConstraintsFromBytes<P, NA>(rt.constraints());

  // A claimed positive bound must carry a proof; a 0 bound is vacuously true.
  const int rank_lower_bound =
      rt.has_rank_lower_bound() ? rt.rank_lower_bound() : 0;
  const auto &proof = rt.rank_lower_bound_proof();
  if (proof.proof_case() == pb::RankLowerBoundProof::PROOF_NOT_SET) {
    CHECK_LE(rank_lower_bound, 0) << "Orbit " << rt.index() << " claims bound "
                                  << rank_lower_bound << " with no proof";
    return;
  }

  // Constraints must already be in canonical (column-reversed) RREF.
  {
    Constraints<P, NA> rref = constraints;
    GaussJordanRREF<P, NA>(&rref);
    CHECK(rref == constraints)
        << "Orbit " << rt.index() << " constraints are not in RREF";
  }

  if (proof.has_flatten_matrix_proof()) {
    const Tensor<P, NA, NB, NC> tensor =
        ApplyConstraintsToTensor<P, NA, NB, NC>(constraints,
                                                   Problem::MakeTensor());
    VerifyFlattenProof<P, NA, NB, NC>(tensor, rank_lower_bound);
  } else if (proof.has_forced_product_proof()) {
    const Tensor<P, NA, NB, NC> tensor =
        ApplyConstraintsToTensor<P, NA, NB, NC>(constraints,
                                                   Problem::MakeTensor());
    VerifyForcedProductProof<P, NA, NB, NC>(tensor, rank_lower_bound,
                                               proof.forced_product_proof());
  } else if (proof.has_rank_one_span_proof()) {
    const Tensor<P, NA, NB, NC> tensor =
        ApplyConstraintsToTensor<P, NA, NB, NC>(constraints,
                                                   Problem::MakeTensor());
    VerifyRankOneSpanProof<P, NA, NB, NC>(tensor, rank_lower_bound,
                                             proof.rank_one_span_proof());
  } else if (proof.has_degenerate_proof()) {
    VerifyDegenerateProof<Problem>(constraints, rank_lower_bound,
                                   proof.degenerate_proof(), group, map);
  } else if (proof.has_backtracking_proof()) {
    VerifyBacktrackingProof<Problem>(constraints, rank_lower_bound, proof,
                                     static_cast<int>(rt.index()), group, map,
                                     archive);
  } else {
    LOG(FATAL) << "Orbit " << rt.index() << ": unknown proof type";
  }
}

// Verify every orbit in a certificate. Sweeps dimensions NA→0; at each,
// verifies all orbits in parallel (consulting only larger, already-inserted
// dimensions), then inserts the dimension's claimed bounds into the plain
// canonical→rank map.
template <class Problem>
int VerifyRankLowerBound(const pb::Certificate &certificate,
                         const std::string &archive_path) {
  constexpr int NA = Problem::kNA;
  constexpr int P = Problem::kP;

  CHECK_EQ(certificate.characteristic(), P);
  CHECK_EQ(certificate.extension_degree(), 1);
  CHECK_EQ(certificate.na(), Problem::kNA);
  CHECK_EQ(certificate.nb(), Problem::kNB);
  CHECK_EQ(certificate.nc(), Problem::kNC);
  if (!certificate.problem_name().empty()) {
    CHECK_EQ(certificate.problem_name(), Problem::Name())
        << "Certificate problem_name does not match the compiled-in problem";
  }
  CHECK_GT(certificate.constrained_tensors_size(), 0);

  // Load the backtracking-trace archive once (read-only during the parallel
  // sweeps below). Only required if some orbit actually has a backtracking
  // proof.
  bool needs_archive = false;
  for (const pb::ConstrainedTensor &rt : certificate.constrained_tensors()) {
    if (rt.rank_lower_bound_proof().has_backtracking_proof()) {
      needs_archive = true;
      break;
    }
  }
  BacktrackingProofArchive archive;
  if (needs_archive) {
    CHECK(!archive_path.empty())
        << "Certificate has backtracking proofs but no archive path was given";
    archive = BacktrackingProofArchive::Load(archive_path);
    CHECK_EQ(archive.Size(),
             static_cast<size_t>(certificate.constrained_tensors_size()));
  }

  const typename Problem::SymmetryGroup group;
  RankMap<Problem> map;

  size_t total_count = 0;
  for (int dim = NA; dim >= 0; --dim) {
    std::vector<const pb::ConstrainedTensor *> rts;
    for (int i = 0; i < certificate.constrained_tensors_size(); ++i) {
      const pb::ConstrainedTensor &rt = certificate.constrained_tensors(i);
      const int rt_dim =
          static_cast<int>(rt.constraints().size() / sizeof(GFVec<P, NA>));
      if (rt_dim == dim) {
        rts.push_back(&rt);
      }
    }
    LOG(INFO) << "Verifying dim=" << dim << ", count=" << rts.size();

    // Phase 1: verify every orbit at this dimension (read-only on the map).
    tbb::parallel_for(size_t(0), rts.size(), [&](size_t i) {
      VerifyOne<Problem>(*rts[i], group, map, archive);
    });

    // Phase 2: insert this dimension's claimed bounds so smaller dimensions can
    // look them up. Done after phase 1 so no insert races a read.
    for (const pb::ConstrainedTensor *rt : rts) {
      const int bound = rt->has_rank_lower_bound() ? rt->rank_lower_bound() : 0;
      const Constraints<P, NA> constraints =
          ConstraintsFromBytes<P, NA>(rt->constraints());
      CHECK(map.emplace(constraints, bound).second)
          << "duplicate orbit representative at index " << rt->index();
    }
    total_count += rts.size();
  }
  CHECK_EQ(total_count,
           static_cast<size_t>(certificate.constrained_tensors_size()));

  const pb::ConstrainedTensor &last = certificate.constrained_tensors(
      certificate.constrained_tensors_size() - 1);
  CHECK(last.constraints().empty())
      << "Last orbit must be the unconstrained tensor (empty constraints)";
  const int result = last.has_rank_lower_bound() ? last.rank_lower_bound() : 0;
  LOG(INFO) << std::format("Verified. Rank lower bound for {} is {}.",
                           certificate.problem_name(), result);
  return result;
}
