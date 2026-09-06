#pragma once

// Top-level dynamic-programming driver.
//
// Reads a Certificate (a list of orbit representatives with their current rank
// lower bounds) and iteratively improves each orbit's lower bound using the
// five techniques, sweeping subspace dimensions from most-constrained (NA, the
// full dual space) down to least-constrained (0, the empty subspace). Because a
// dimension's degenerate / backtracking lookups only ever consult orbits of
// strictly larger dimension — already processed and inserted into the OrbitMap
// — one top-down pass per dimension settles every orbit.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ng-log/logging.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "core/backtracking_proof.h"
#include "core/bit_vec.h"
#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/proto_io.h"
#include "core/rank_lower_bound_flatten.h"
#include "core/rank_lower_bound_forced_product.h"
#include "core/rank_lower_bound_rank_one_span.h"
#include "core/tensor.h"
#include "subspace_bounds/search/orbit_map.h"
#include "subspace_bounds/search/rank_lower_bound_backtracking.h"
#include "subspace_bounds/search/rank_lower_bound_degenerate.h"

struct ProcessOptions {
  bool basic_method = true; // Flatten + Forced Product
  bool degenerate_method = true;
  uint64_t backtracking_step_limit = std::numeric_limits<uint64_t>::max();
  size_t backtracking_max_map_size = 10'000'000;
  int dim_min = 0;
  int dim_max = std::numeric_limits<int>::max();
  int forced_product_max_iterations_log2 = 24;
  // Rank-one-span exclusion budget (𝔽₂ only): skip the technique on an orbit
  // when its up-front cost estimate exceeds this; 0 disables it. Off by
  // default so existing golden certificates stay byte-identical.
  uint64_t rank1span_max_subspaces = 0;
  // Archive regeneration: for every orbit whose recorded proof is a
  // backtracking proof and whose archive trace is missing or does not match
  // proof_size, re-run the search at the recorded bound (target = lb, never
  // lb + 1) and store the new trace. No other technique runs and no bound is
  // searched for; only proof_size (and, if the search happens to prove more,
  // the bound) can change. Restores an archive that fell out of sync with
  // its certificate.
  bool regenerate_backtracking_proofs = false;
  // Restrict processing to the orbits with these `index` values (empty = every
  // orbit of the dimensions in [dim_min, dim_max]). Orbits not listed keep
  // their bounds and proofs and only pre-seed the OrbitMap. Used for
  // per-orbit reruns and ablations (rank_lower_bound_main --recompute_orbits).
  std::vector<int> only_orbits;
};

// Three-position Forced Product: run RankLowerBoundForcedProductA on the tensor
// and its two cyclic transposes, recording which projection won. Mirrors
// rank_search/rank_lower_bound_computer.h's RankLowerBoundForcedProduct.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
std::pair<int, pb::ForcedProductProof>
RankLowerBoundForcedProduct(const Tensor<P, NA, NB, NC> &tensor,
                            int known_lower_bound, int max_iterations_log2) {
  int rank0 = RankLowerBoundForcedProductA<P, NA, NB, NC>(
      tensor, known_lower_bound, max_iterations_log2);
  known_lower_bound = std::max(known_lower_bound, rank0);
  Tensor<P, NB, NC, NA> tensor1 = CyclicTranspose<P, NA, NB, NC>(tensor);
  int rank1 = RankLowerBoundForcedProductA<P, NB, NC, NA>(
      tensor1, known_lower_bound, max_iterations_log2);
  known_lower_bound = std::max(known_lower_bound, rank1);
  Tensor<P, NC, NA, NB> tensor2 = CyclicTranspose<P, NB, NC, NA>(tensor1);
  int rank2 = RankLowerBoundForcedProductA<P, NC, NA, NB>(
      tensor2, known_lower_bound, max_iterations_log2);
  known_lower_bound = std::max(known_lower_bound, rank2);

  pb::ForcedProductProof forced_product_proof;
  if (rank0 == known_lower_bound) {
    forced_product_proof.set_projection_type(0);
  } else if (rank1 == known_lower_bound) {
    forced_product_proof.set_projection_type(1);
  } else if (rank2 == known_lower_bound) {
    forced_product_proof.set_projection_type(2);
  } else {
    return {0, forced_product_proof};
  }
  return {known_lower_bound, forced_product_proof};
}

// Number of constraints (subspace dimension) encoded in a ConstrainedTensor.
template <class Problem> int NumConstraints(const pb::ConstrainedTensor &rt) {
  return static_cast<int>(
      rt.constraints().size() /
      sizeof(GFVec<Problem::kP, Problem::kNA>));
}

namespace rank_lower_bound_computer_internal {

template <class Problem>
using ProblemTensor =
    Tensor<Problem::kP, Problem::kNA, Problem::kNB, Problem::kNC>;
template <class Problem>
using ProblemConstraints = Constraints<Problem::kP, Problem::kNA>;

// The best bound found so far for one orbit and the proof that established
// it. A technique replaces the proof only on a strictly better bound (a proof
// at an equal bound is never swapped). `blob` is the backtracking trace; it is
// non-empty iff `proof` is a backtracking proof.
struct OrbitBound {
  int rank_lower_bound = -1; // -1: no bound yet
  pb::RankLowerBoundProof proof;
  BacktrackingProof blob;
};

// Archive regeneration (ProcessOptions::regenerate_backtracking_proofs): a
// single search at the recorded bound; the trace it returns replaces the
// missing/mismatched one. Nothing else runs. On failure the recorded bound
// comes back with no proof, so the write-back skips the orbit.
template <class Problem>
std::tuple<int, pb::RankLowerBoundProof, BacktrackingProof>
RegenerateBacktrackingTrace(const pb::ConstrainedTensor &rt,
                            const ProblemConstraints<Problem> &constraints,
                            const OrbitMap<Problem> &orbit_map,
                            const ProcessOptions &options,
                            int rank_lower_bound) {
  CHECK(rt.rank_lower_bound_proof().has_backtracking_proof());
  auto [rank, backtracking_proof, blob] =
      RankLowerBoundBacktracking<Problem>::Search(
          constraints, orbit_map, rank_lower_bound - 1,
          options.backtracking_step_limit, options.backtracking_max_map_size);
  if (rank < rank_lower_bound) {
    LOG(ERROR) << "could not regenerate the backtracking trace of orbit index="
               << rt.index() << " at bound " << rank_lower_bound
               << " within the step limit; its archive entry stays stale";
    return {rank_lower_bound, pb::RankLowerBoundProof{}, BacktrackingProof{}};
  }
  pb::RankLowerBoundProof proof;
  *proof.mutable_backtracking_proof() = std::move(backtracking_proof);
  return {rank, std::move(proof), std::move(blob)};
}

template <class Problem>
void TryFlatten(const pb::ConstrainedTensor &rt,
                const ProblemTensor<Problem> &tensor, OrbitBound *best) {
  const int flatten_rank = RankLowerBoundFlatten(tensor);
  if (flatten_rank > best->rank_lower_bound) {
    best->rank_lower_bound = flatten_rank;
    *best->proof.mutable_flatten_matrix_proof() = {};
    VLOG(1) << "index=" << rt.index() << " flatten -> "
            << best->rank_lower_bound;
  }
}

template <class Problem>
void TryDegenerate(const pb::ConstrainedTensor &rt,
                   const OrbitMap<Problem> &orbit_map,
                   const ProblemConstraints<Problem> &constraints,
                   OrbitBound *best) {
  auto [degenerate_rank, degenerate_proof] =
      RankLowerBoundDegenerate<Problem>(orbit_map, constraints);
  if (degenerate_rank > best->rank_lower_bound) {
    best->rank_lower_bound = degenerate_rank;
    *best->proof.mutable_degenerate_proof() = std::move(degenerate_proof);
    VLOG(1) << "index=" << rt.index() << " degenerate -> "
            << best->rank_lower_bound;
  }
}

template <class Problem>
void TryForcedProduct(const pb::ConstrainedTensor &rt,
                      const ProblemTensor<Problem> &tensor,
                      const ProcessOptions &options, OrbitBound *best) {
  auto [forced_product_rank, forced_product_proof] =
      RankLowerBoundForcedProduct(tensor, best->rank_lower_bound,
                                  options.forced_product_max_iterations_log2);
  if (forced_product_rank > best->rank_lower_bound) {
    best->rank_lower_bound = forced_product_rank;
    *best->proof.mutable_forced_product_proof() =
        std::move(forced_product_proof);
    VLOG(1) << "index=" << rt.index() << " forced product (axis "
            << best->proof.forced_product_proof().projection_type()
            << ") -> " << best->rank_lower_bound;
  }
}

// Rank-one-span exclusion (prime fields; the family search is 𝔽₂-only),
// climbing one rank at a time: excluding rank ≤ lb proves lb + 1. Never
// attempts a target ≥ the known upper bound: exclusion there must fail, and
// the write-back CHECKs lb ≤ ub.
template <class Problem>
void ClimbRankOneSpan(const pb::ConstrainedTensor &rt,
                      const ProblemTensor<Problem> &tensor,
                      const ProcessOptions &options, OrbitBound *best) {
  const int upper = rt.has_rank_upper_bound()
                        ? rt.rank_upper_bound()
                        : std::numeric_limits<int>::max();
  while (best->rank_lower_bound < upper) {
    auto [rank1span_result, rank1span_axis] =
        RankOneSpanExclude(tensor, best->rank_lower_bound,
                           options.rank1span_max_subspaces, /*parallel=*/true);
    if (rank1span_result != RankOneSpanResult::kExcluded) {
      if (rank1span_result == RankOneSpanResult::kWitnessFound &&
          rt.has_rank_upper_bound() &&
          best->rank_lower_bound < rt.rank_upper_bound()) {
        // The exclusion found an actual rank-(<= current lb) witness, so
        // the recorded upper bound is not tight. Report only; improving
        // the upper bound is the flip-graph stage's job.
        LOG(WARNING) << "rank-one-span found a rank<=" << best->rank_lower_bound
                     << " witness for orbit index=" << rt.index()
                     << "; rank_upper_bound=" << rt.rank_upper_bound()
                     << " is improvable (not recorded)";
      } else if (rank1span_result == RankOneSpanResult::kOverBudget) {
        LOG(INFO) << "rank-one-span over budget for orbit index="
                  << rt.index() << " at target " << best->rank_lower_bound
                  << ": no claim";
      }
      break;
    }
    ++best->rank_lower_bound;
    best->proof.mutable_rank_one_span_proof()->set_slice_axis(
        static_cast<uint32_t>(rank1span_axis));
    VLOG(1) << "index=" << rt.index() << " rank-one-span (axis "
            << rank1span_axis << ") -> " << best->rank_lower_bound;
  }
}

// Backtracking DFS, repeated while it keeps improving the bound; the winning
// trace rides along in best->blob.
template <class Problem>
void ClimbBacktracking(const pb::ConstrainedTensor &rt,
                       const ProblemConstraints<Problem> &constraints,
                       const OrbitMap<Problem> &orbit_map,
                       const ProcessOptions &options, OrbitBound *best) {
  while (true) {
    auto [backtracking_rank, backtracking_proof, backtracking_blob] =
        RankLowerBoundBacktracking<Problem>::Search(
            constraints, orbit_map, best->rank_lower_bound,
            options.backtracking_step_limit,
            options.backtracking_max_map_size);
    if (backtracking_rank <= best->rank_lower_bound) {
      break;
    }
    best->rank_lower_bound = backtracking_rank;
    *best->proof.mutable_backtracking_proof() = std::move(backtracking_proof);
    best->blob = std::move(backtracking_blob);
    VLOG(1) << "index=" << rt.index() << " backtracking -> "
            << best->rank_lower_bound;
  }
}

} // namespace rank_lower_bound_computer_internal

// Process a single constrained-tensor orbit, returning (new_lb, proof, trace).
// `trace` is the winning backtracking DFS trace when a backtracking proof won
// (the caller stores it in the per-certificate archive), and empty otherwise.
// Technique order: flatten, degenerate, forced product, rank-one-span,
// backtracking. Flatten and forced product only run on orbits without a
// recorded bound (a re-run keeps what an earlier pass found); the others are
// cheap under their budgets and self-contained, so they also run on re-runs.
template <class Problem>
std::tuple<int, pb::RankLowerBoundProof, BacktrackingProof>
ProcessOrbit(const pb::ConstrainedTensor &rt,
             const OrbitMap<Problem> &orbit_map,
             const ProcessOptions &options, bool regenerate = false) {
  using namespace rank_lower_bound_computer_internal;
  OrbitBound best;
  best.rank_lower_bound =
      rt.has_rank_lower_bound() ? rt.rank_lower_bound() : -1;
  const ProblemConstraints<Problem> constraints =
      ConstraintsFromBytes<Problem::kP, Problem::kNA>(rt.constraints());
  if (regenerate) {
    return RegenerateBacktrackingTrace<Problem>(rt, constraints, orbit_map,
                                                options, best.rank_lower_bound);
  }
  const ProblemTensor<Problem> tensor =
      ApplyConstraintsToTensor<Problem::kP, Problem::kNA, Problem::kNB,
                               Problem::kNC>(constraints,
                                             Problem::MakeTensor());
  const bool fresh = options.basic_method && !rt.has_rank_lower_bound();
  if (fresh) {
    TryFlatten<Problem>(rt, tensor, &best);
  }
  if (options.degenerate_method) {
    TryDegenerate<Problem>(rt, orbit_map, constraints, &best);
  }
  if (fresh) {
    TryForcedProduct<Problem>(rt, tensor, options, &best);
  }
  if (options.rank1span_max_subspaces > 0) {
    ClimbRankOneSpan<Problem>(rt, tensor, options, &best);
  }
  if (options.backtracking_step_limit > 0) {
    ClimbBacktracking<Problem>(rt, constraints, orbit_map, options, &best);
  }
  return {best.rank_lower_bound, std::move(best.proof), std::move(best.blob)};
}

namespace rank_lower_bound_computer_internal {

using OrbitResult = std::tuple<int, pb::RankLowerBoundProof, BacktrackingProof>;

// The certificate's orbits with `dim` constraints (restricted to
// options.only_orbits when that is non-empty), in a random order.
template <class Problem>
std::vector<pb::ConstrainedTensor *>
OrbitsAtDim(int dim, const ProcessOptions &options,
            pb::Certificate *certificate, std::mt19937_64 *rng) {
  std::vector<pb::ConstrainedTensor *> rts;
  for (int i = 0; i < certificate->constrained_tensors_size(); ++i) {
    pb::ConstrainedTensor *rt = certificate->mutable_constrained_tensors(i);
    if (NumConstraints<Problem>(*rt) != dim) {
      continue;
    }
    if (!options.only_orbits.empty() &&
        std::find(options.only_orbits.begin(), options.only_orbits.end(),
                  static_cast<int>(rt->index())) == options.only_orbits.end()) {
      continue;
    }
    rts.push_back(rt);
  }
  std::shuffle(rts.begin(), rts.end(), *rng);
  return rts;
}

// Regeneration mode: which orbits have a backtracking trace that is absent
// from the archive or inconsistent with the certificate (decided serially;
// the archive is not read inside the parallel phase).
inline std::vector<bool>
StaleBacktrackingTraces(const std::vector<pb::ConstrainedTensor *> &rts,
                        const BacktrackingProofArchive &archive) {
  std::vector<bool> regenerate(rts.size(), false);
  for (size_t i = 0; i < rts.size(); ++i) {
    const pb::ConstrainedTensor &rt = *rts[i];
    if (!rt.rank_lower_bound_proof().has_backtracking_proof()) {
      continue;
    }
    const size_t recorded =
        rt.rank_lower_bound_proof().backtracking_proof().proof_size();
    regenerate[i] =
        archive.Get(static_cast<int>(rt.index())).Size() != recorded;
  }
  return regenerate;
}

// Phase 1: the new bound of every orbit, computed in parallel with the map
// read-only. Each task writes only its own slot, so the backtracking blob
// rides along with no shared mutation. In regeneration mode the orbits whose
// trace is not stale get their recorded bound with no proof (the write-back
// skips them).
template <class Problem>
std::vector<OrbitResult>
ComputeOrbitBounds(const std::vector<pb::ConstrainedTensor *> &rts,
                   const std::vector<bool> &regenerate,
                   const OrbitMap<Problem> &orbit_map,
                   const ProcessOptions &options) {
  std::vector<OrbitResult> results(rts.size());
  std::atomic<int> progress = 0;
  tbb::parallel_for(0, static_cast<int>(rts.size()), [&](int idx) {
    if (options.regenerate_backtracking_proofs && !regenerate[idx]) {
      results[idx] = {rts[idx]->rank_lower_bound(), {}, {}};
      progress.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    auto [rank, rank_proof, blob] = ProcessOrbit<Problem>(
        *rts[idx], orbit_map, options, options.regenerate_backtracking_proofs);
    if (rank_proof.proof_case() != pb::RankLowerBoundProof::PROOF_NOT_SET) {
      if (options.regenerate_backtracking_proofs) {
        LOG(INFO) << "Regenerated backtracking trace for index="
                  << rts[idx]->index() << " at bound " << rank
                  << " (proof_size="
                  << rank_proof.backtracking_proof().proof_size() << ")";
      } else {
        LOG(INFO) << "Better LB for index=" << rts[idx]->index() << ": "
                  << rts[idx]->rank_lower_bound() << "->" << rank;
      }
    }
    results[idx] = {rank, std::move(rank_proof), std::move(blob)};

    int local_progress = progress.fetch_add(1, std::memory_order_relaxed) + 1;
    double progress_percentage = local_progress * 100.0 / rts.size();
    LOG_EVERY_T(INFO, 10) << std::format(
        "ProcessOrbitsAtDim P1: {} / {} = {:.2f}%", local_progress, rts.size(),
        progress_percentage);
  });
  return results;
}

// Phase 2: every orbit's (final) bound goes into the map so smaller
// dimensions can look it up. Done after phase 1 so no Set races a Get.
template <class Problem>
void PublishBounds(const std::vector<pb::ConstrainedTensor *> &rts,
                   const std::vector<OrbitResult> &results,
                   OrbitMap<Problem> *orbit_map) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, static_cast<int>(rts.size())),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i != range.end(); ++i) {
          orbit_map->Set(ConstraintsFromBytes<Problem::kP, Problem::kNA>(
                             rts[i]->constraints()),
                         std::get<0>(results[i]));
        }
      });
}

// Phase 3: the improved bounds and proofs go back into the certificate and
// the in-memory archive (serial; the archive is never touched in a
// parallel_for). Returns true iff any orbit improved.
inline bool WriteBackProofs(const std::vector<pb::ConstrainedTensor *> &rts,
                            std::vector<OrbitResult> *results,
                            BacktrackingProofArchive *archive) {
  bool has_update = false;
  for (int i = 0; i < static_cast<int>(rts.size()); ++i) {
    auto &[new_rank, new_proof, new_blob] = (*results)[i];
    if (new_proof.proof_case() == pb::RankLowerBoundProof::PROOF_NOT_SET) {
      continue;
    }
    pb::ConstrainedTensor *rt = rts[i];
    CHECK_LE(rt->rank_lower_bound(), new_rank);
    has_update = true;
    if (new_proof.has_backtracking_proof()) {
      archive->Set(static_cast<int>(rt->index()), std::move(new_blob));
    } else {
      // A non-backtracking proof won; drop any stale blob for this orbit.
      archive->Clear(static_cast<int>(rt->index()));
    }
    rt->set_rank_lower_bound(new_rank);
    *rt->mutable_rank_lower_bound_proof() = std::move(new_proof);
    if (rt->has_rank_upper_bound()) {
      CHECK_LE(rt->rank_lower_bound(), rt->rank_upper_bound()) << rt->index();
    }
  }
  return has_update;
}

} // namespace rank_lower_bound_computer_internal

// Process all orbits at a single subspace dimension, then insert their
// (updated) bounds into the OrbitMap so the next-smaller dimension can consult
// them. Returns true iff any orbit's lower bound improved. Mirrors
// ProcessOneRankLowerBound in the matrix-mult code.
template <class Problem>
bool ProcessOrbitsAtDim(int dim, const ProcessOptions &options,
                        BacktrackingProofArchive *archive,
                        pb::Certificate *certificate,
                        OrbitMap<Problem> *orbit_map, std::mt19937_64 *rng) {
  using namespace rank_lower_bound_computer_internal;
  const auto iteration_start = std::chrono::steady_clock::now();

  const std::vector<pb::ConstrainedTensor *> rts =
      OrbitsAtDim<Problem>(dim, options, certificate, rng);
  LOG(INFO) << "Processing dim=" << dim << ", count=" << rts.size();

  const std::vector<bool> regenerate =
      options.regenerate_backtracking_proofs
          ? StaleBacktrackingTraces(rts, *archive)
          : std::vector<bool>(rts.size(), false);
  std::vector<OrbitResult> results =
      ComputeOrbitBounds<Problem>(rts, regenerate, *orbit_map, options);

  LOG(INFO) << "ProcessOrbitsAtDim P2...";
  PublishBounds<Problem>(rts, results, orbit_map);

  LOG(INFO) << "ProcessOrbitsAtDim P3...";
  const bool has_update = WriteBackProofs(rts, &results, archive);

  const auto iteration_end = std::chrono::steady_clock::now();
  LOG(INFO)
      << "ProcessOrbitsAtDim done. duration=" << std::fixed
      << std::setprecision(1)
      << std::chrono::duration<double>(iteration_end - iteration_start).count();
  return has_update;
}

// Run the full DP: pre-seed the OrbitMap with any bounds already in the
// certificate, then sweep dimensions from NA down to 0, processing each and
// checkpointing the certificate to `output_path` after every dimension.
template <class Problem>
void ProcessOrbits(const ProcessOptions &options,
                   const std::string &output_path,
                   pb::Certificate *certificate) {
  constexpr int NA = Problem::kNA;
  constexpr int P = Problem::kP;

  const typename Problem::SymmetryGroup group;
  OrbitMap<Problem> orbit_map(&group);
  std::mt19937_64 rng;

  // Pre-seed: orbits with a known bound (e.g. from a prior run, or dimensions
  // excluded by dim_min/dim_max) must be in the map for the boundary
  // dimension's degenerate / backtracking lookups to find them.
  tbb::parallel_for(
      tbb::blocked_range<int>(0, certificate->constrained_tensors_size()),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i != range.end(); ++i) {
          const pb::ConstrainedTensor &rt = certificate->constrained_tensors(i);
          if (rt.has_rank_lower_bound()) {
            orbit_map.Set(ConstraintsFromBytes<P, NA>(rt.constraints()),
                          rt.rank_lower_bound());
          }
        }
      });

  // Backtracking traces live in one archive next to the certificate output.
  // With no output path there is nowhere to anchor it, so it is skipped. For
  // partial re-runs (dim_min/dim_max), load any existing archive so orbits
  // whose backtracking proofs are not reprocessed keep their traces.
  const std::string archive_path =
      output_path.empty() ? std::string{}
                          : GetBacktrackingProofArchivePath(output_path);
  BacktrackingProofArchive archive;
  if (!archive_path.empty() && std::filesystem::exists(archive_path)) {
    archive = BacktrackingProofArchive::Load(archive_path);
    CHECK_EQ(archive.Size(),
             static_cast<size_t>(certificate->constrained_tensors_size()));
  } else {
    archive.Resize(certificate->constrained_tensors_size());
  }

  for (int dim = NA; dim >= 0; --dim) {
    if (dim < options.dim_min || dim > options.dim_max) {
      continue;
    }
    ProcessOrbitsAtDim<Problem>(dim, options, &archive, certificate, &orbit_map,
                                &rng);
    if (!output_path.empty()) {
      WriteProtoToFile(*certificate, output_path);
      archive.Save(archive_path);
    }
  }
}
