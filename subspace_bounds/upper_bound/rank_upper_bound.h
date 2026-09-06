#pragma once

// Tensor rank *upper* bound over F_2 via the flip-graph local search.
//
// RankUpperBound takes an arbitrary 3-tensor over F_2 (core/tensor.h) and runs
// the randomized flip/plus search (see flip_scheme.h), returning the smallest
// rank it found together with the matching decomposition string. The returned
// decomposition is verified to reconstruct the input tensor.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "ng-log/logging.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "core/gf.h"
#include "core/tensor.h"
#include "subspace_bounds/upper_bound/flip_scheme.h"

namespace upper_bound {

// Step / path budget (the search has no wall-clock notion). A "path" is one
// independent restart from the trivial decomposition; within a path the search
// takes up to `path_limit` flip/plus steps.
struct UpperBoundOptions {
  uint64_t seed = std::mt19937_64::default_seed;
  int64_t path_limit = 1000;
  int max_steps_at_a_rank = 100;
  int64_t num_paths = 10;
};

template <std::size_t NA, std::size_t NB, std::size_t NC>
struct UpperBoundResult {
  int rank = 0;
  std::string scheme;
};

namespace internal {

// Reconstruct the tensor a decomposition represents (XOR of outer products) and
// CHECK it equals `tensor`.
template <std::size_t NA, std::size_t NB, std::size_t NC>
void VerifyReconstruction(const FlipScheme &scheme,
                          const Tensor<2, NA, NB, NC> &tensor) {
  Tensor<2, NA, NB, NC> rebuilt = {};
  for (const RankOneTensor &abc : scheme.RankOneTensors()) {
    for (std::size_t i = 0; i < NA; ++i) {
      if (!((abc[0] >> i) & Vec(1))) {
        continue;
      }
      for (std::size_t j = 0; j < NB; ++j) {
        if (!((abc[1] >> j) & Vec(1))) {
          continue;
        }
        for (std::size_t k = 0; k < NC; ++k) {
          if ((abc[2] >> k) & Vec(1)) {
            rebuilt[i][j][k] += GF<2>::One();
          }
        }
      }
    }
  }
  CHECK(rebuilt == tensor) << "decomposition does not reconstruct the tensor";
}

// One search path, mirroring f2/scheme_2024_main.cc's Search loop. Whenever a
// smaller rank is reached, copies the current scheme into *best.
inline void SearchPath(FlipScheme scheme, int64_t path_limit,
                       int max_steps_at_a_rank, std::mt19937_64 *rng,
                       FlipScheme *best) {
  int step_at_a_rank = 0;
  int rank = scheme.Rank();
  for (int64_t step = 0; step < path_limit; ++step, ++step_at_a_rank) {
    int old_rank = rank;
    rank = scheme.Rank();
    if (rank != old_rank) {
      step_at_a_rank = 0;
    }
    if (rank < best->Rank()) {
      *best = scheme;
    }
    if (step_at_a_rank < max_steps_at_a_rank) {
      if (!scheme.Step(rng)) {
        scheme.Plus(rng);
        step_at_a_rank = 0;
      }
    } else {
      scheme.Plus(rng);
      step_at_a_rank = 0;
    }
  }
}

} // namespace internal

template <std::size_t NA, std::size_t NB, std::size_t NC>
UpperBoundResult<NA, NB, NC>
RankUpperBound(const Tensor<2, NA, NB, NC> &tensor,
               const UpperBoundOptions &opts = {}) {
  static_assert(NA > 0 && NB > 0 && NC > 0, "tensor dimensions must be > 0");
  static_assert(NA < 64 && NB < 64 && NC < 64,
                "each mode must fit in a uint64_t bitmask");

  // Trivial decomposition: one e_i (X) e_j (X) e_k term per nonzero entry.
  std::vector<RankOneTensor> terms;
  for (std::size_t i = 0; i < NA; ++i) {
    for (std::size_t j = 0; j < NB; ++j) {
      for (std::size_t k = 0; k < NC; ++k) {
        if (tensor[i][j][k] != GF<2>::Zero()) {
          terms.push_back({Vec(1) << i, Vec(1) << j, Vec(1) << k});
        }
      }
    }
  }

  FlipScheme base(
      {static_cast<int>(NA), static_cast<int>(NB), static_cast<int>(NC)},
      terms);

  int64_t num_paths = std::max<int64_t>(opts.num_paths, 1);

  // Run the independent restarts in parallel. Each path gets its own RNG
  // (seeded deterministically from the base seed and path index). A worker
  // keeps a local best across the paths in its range, then merges it into the
  // shared `best` under a mutex when its range is done.
  FlipScheme best = base;
  std::mutex best_mutex;
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, num_paths),
      [&](const tbb::blocked_range<int64_t> &range) {
        FlipScheme local_best = base;
        for (int64_t path_idx = range.begin(); path_idx < range.end();
             ++path_idx) {
          std::mt19937_64 rng(opts.seed + static_cast<uint64_t>(path_idx));
          internal::SearchPath(base, opts.path_limit, opts.max_steps_at_a_rank,
                               &rng, &local_best);
        }
        std::lock_guard<std::mutex> lock(best_mutex);
        if (local_best.Rank() < best.Rank()) {
          best = local_best;
        }
      });

  internal::VerifyReconstruction(best, tensor);

  return {best.Rank(), best.ToString()};
}

} // namespace upper_bound
