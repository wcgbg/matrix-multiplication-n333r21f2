#pragma once

// Forced Product lower bound (Hopcroft–Kerr Lemma 2).
//
// Lemma 2: let F = {f_0, …, f_{n−1}} be expressions where f_0, …, f_{k−1} are
// independent and each is a single product. If F is computable with p
// multiplications, then there is an algorithm for F with p multiplications in
// which k of them are exactly f_0, …, f_{k−1}.
//
// Applied to the A-mode: each a-coordinate i contributes the bc-slice T[i] (an
// NB × NC matrix). The rank-1 slices are "single products" and, once a maximal
// independent set of r1 of them is fixed as forced products, the remaining
// rank must cover the higher-rank slices plus every 𝔽_q-combination of the
// forced products folded into them. We search all q^(r2p·r1) combinations and
// take the minimum flattening lower bound, then add back the r1 forced rows.
//
// SOUNDNESS NOTE: the lemma gives rank(T) = r1 + min over ALL combinations c of
// rank(T_c). We lower-bound each rank(T_c) by a flatten bound and take the min,
// so the min MUST range over every combination. Each coefficient ranges over
// all P elements of 𝔽_P; skipping combinations could report a minimum larger
// than the true rank and give an unsound lower bound.
//
// RankLowerBoundForcedProductA handles one slicing axis. The three-position
// wrapper records the winning projection in
// subspace_bounds/search/rank_lower_bound_computer.h.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <vector>

#include "ng-log/logging.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"

#include "core/dynamic_matrix.h"
#include "core/gf.h"
#include "core/rank_lower_bound_flatten.h"
#include "core/tensor.h"

namespace forced_product_internal {

// Scrambler: a large prime coprime to num_iterations gives a pseudo-random
// visit order so an early break is likely to hit a low-rank witness fast.
inline constexpr uint64_t kScramblePrime = 73074167;

// The NB × NC rank of every a-slice.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
std::vector<int> SliceRanks(const Tensor<P, NA, NB, NC> &tensor) {
  std::vector<int> ranks(NA, 0);
  for (std::size_t i = 0; i < NA; ++i) {
    DynamicMatrix<P> bc_matrix(tensor[i]);
    ranks[i] = bc_matrix.Rank();
  }
  return ranks;
}

// The slices partitioned: a maximal independent set of rank-1 slices becomes
// the forced-product rows (r1_bc_collection, r1_bc_rows of them, each an
// NB·NC vector); everything else (rank ≥ 2, plus the rank-1 slices that were
// dependent on already-chosen ones) goes into r2p, packed at the front
// (r2p_size0 slices). Zero slices are dropped.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
struct ForcedProductPartition {
  Tensor<P, NA, NB, NC> r2p = {};
  int r2p_size0 = 0;
  DynamicMatrix<P> r1_bc_collection{0, static_cast<int>(NB * NC)};
  int r1_bc_rows = 0;
};

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
ForcedProductPartition<P, NA, NB, NC>
PartitionSlices(const Tensor<P, NA, NB, NC> &tensor,
                const std::vector<int> &ranks) {
  ForcedProductPartition<P, NA, NB, NC> part;
  for (std::size_t i = 0; i < NA; ++i) {
    if (ranks[i] == 0) {
      // drop: zero slice contributes nothing
    } else if (ranks[i] == 1) {
      part.r1_bc_collection.ResizeRows(part.r1_bc_rows + 1);
      for (std::size_t j = 0; j < NB; ++j) {
        for (std::size_t k = 0; k < NC; ++k) {
          part.r1_bc_collection(part.r1_bc_rows, static_cast<int>(j * NC + k)) =
              tensor[i][j][k];
        }
      }
      if (part.r1_bc_collection.Rank() == part.r1_bc_rows + 1) {
        ++part.r1_bc_rows;
      } else {
        part.r1_bc_collection.ResizeRows(part.r1_bc_rows);
        part.r2p[part.r2p_size0] = tensor[i];
        ++part.r2p_size0;
      }
    } else {
      part.r2p[part.r2p_size0] = tensor[i];
      ++part.r2p_size0;
    }
  }
  return part;
}

// kQ^(r2p_size0 · r1_bc_rows), or nullopt (after a WARNING) when it exceeds
// 2^max_iterations_log2. The exponent can be O(NA^2), so the product can be
// astronomically larger than uint64: it is never handed to IntPow, which
// treats overflow as fatal.
inline std::optional<uint64_t>
NumIterationsWithinBudget(int kQ, int r2p_size0, int r1_bc_rows,
                          int max_iterations_log2) {
  CHECK_GE(max_iterations_log2, 0);
  CHECK_LE(max_iterations_log2, 64);
  const int bit_width = r2p_size0 * r1_bc_rows;
  uint64_t num_iterations = 1;
  for (int i = 0; i < bit_width; ++i) {
    // With max_iterations_log2 = 64 the budget check below never fires, so
    // guard the product itself: kQ^bit_width must fit in 64 bits.
    CHECK_LE(num_iterations,
             std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(kQ))
        << "Forced Product: " << kQ << "^(" << r2p_size0 << "*" << r1_bc_rows
        << ") does not fit in 64 bits";
    num_iterations *= static_cast<uint64_t>(kQ);
    if (max_iterations_log2 < 64 &&
        num_iterations > (uint64_t{1} << max_iterations_log2)) {
      LOG(WARNING) << "Cancel Forced Product. num_iterations=" << kQ << "^("
                   << r2p_size0 << "*" << r1_bc_rows << ") exceeds 2^"
                   << max_iterations_log2;
      return std::nullopt;
    }
  }
  if (num_iterations > (uint64_t{1} << 24)) {
    LOG(INFO) << "Forced Product. num_iterations=" << kQ << "^(" << r2p_size0
              << "*" << r1_bc_rows << ")=" << num_iterations;
  }
  return num_iterations;
}

// r2p with combination `scrambled` of the forced products folded in: digit
// (i · r1_bc_rows + r1_idx) of `scrambled` (a bit over F_2, a base-q digit
// otherwise) is the coefficient of forced-product row r1_idx added to r2p
// slice i.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
Tensor<P, NA, NB, NC>
FoldForcedProducts(const Tensor<P, NA, NB, NC> &r2p, const GF<P> *r1_bc_data,
                   int r1_bc_rows, int bit_width, uint64_t scrambled) {
  using Field = GF<P>;
  Tensor<P, NA, NB, NC> tensor_t = r2p;
  if constexpr (P == 2) {
    // F_2 fast path: each coefficient is a bit, contribution is XOR.
    // Over GF<2,1>, `+=` is XOR.
    for (int bit_idx = 0; bit_idx < bit_width; ++bit_idx) {
      if (((scrambled >> bit_idx) & 1) == 0) {
        continue;
      }
      int r1_idx = bit_idx % r1_bc_rows; // forced-product row
      int i = bit_idx / r1_bc_rows;      // r2p slice
      for (std::size_t j = 0; j < NB; ++j) {
        for (std::size_t k = 0; k < NC; ++k) {
          tensor_t[i][j][k] += r1_bc_data[r1_idx * (NB * NC) + j * NC + k];
        }
      }
    }
  } else {
    // General 𝔽_q path: read scrambled as a base-q numeral; each digit
    // is a field index in [0, q) (the GF<P> `value` IS the index), so
    // the coefficient ranges over every element of 𝔽_q. Contribution is
    // digit · r1_bc_data over 𝔽_q.
    constexpr int kQ = Field::kQ;
    uint64_t v = scrambled;
    for (int bit_idx = 0; bit_idx < bit_width; ++bit_idx) {
      const Field digit{static_cast<uint8_t>(v % kQ)};
      v /= kQ;
      if (digit == Field::Zero()) {
        continue;
      }
      int r1_idx = bit_idx % r1_bc_rows;
      int i = bit_idx / r1_bc_rows;
      for (std::size_t j = 0; j < NB; ++j) {
        for (std::size_t k = 0; k < NC; ++k) {
          const Field r1_val = r1_bc_data[r1_idx * (NB * NC) + j * NC + k];
          tensor_t[i][j][k] = tensor_t[i][j][k] + digit * r1_val;
        }
      }
    }
  }
  return tensor_t;
}

} // namespace forced_product_internal

// Returns a rank lower bound from the Forced Product technique on the A-mode,
// or 0 when the technique does not apply (no rank-1 slices, or the search space
// is too large to enumerate). Never returns more than max(known_lower_bound,…);
// the caller maxes the three cyclic positions.
//
// Each "coefficient position" can take any value in the full field 𝔽_q, so the
// enumeration covers q^bit_width combinations (q = P). The slice-update is
// `tensor_t[i][j][k] += digit · r1_bc[...]` over 𝔽_q, where `digit` ranges over
// every field element. The field F_2 is special-cased to a
// single-bit XOR fast path; odd P uses the general field path.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
int RankLowerBoundForcedProductA(const Tensor<P, NA, NB, NC> &tensor,
                                 int known_lower_bound = 0,
                                 int max_iterations_log2 = 64) {
  using namespace forced_product_internal;
  using Field = GF<P>;
  const std::vector<int> ranks = SliceRanks(tensor);
  if (std::count(ranks.begin(), ranks.end(), 1) == 0) {
    return 0;
  }
  const ForcedProductPartition<P, NA, NB, NC> part =
      PartitionSlices(tensor, ranks);

  constexpr int kQ = Field::kQ; // coefficients range over all of F_P.
  const int bit_width = part.r2p_size0 * part.r1_bc_rows;
  const std::optional<uint64_t> budgeted = NumIterationsWithinBudget(
      kQ, part.r2p_size0, part.r1_bc_rows, max_iterations_log2);
  if (!budgeted) {
    return 0;
  }
  const uint64_t num_iterations = *budgeted;

  const auto start_time = std::chrono::steady_clock::now();

  const Field *r1_bc_data = part.r1_bc_collection.data();
  CHECK_NOTNULL(r1_bc_data);

  std::atomic<bool> early_break{false};
  std::atomic<uint64_t> progress{0};
  int rank_lower_bound = tbb::parallel_reduce(
      tbb::blocked_range<uint64_t>(0, num_iterations),
      std::numeric_limits<int>::max(),
      [&](const tbb::blocked_range<uint64_t> &range, int init) {
        int local_min = init;
        for (uint64_t t = range.begin(); t != range.end(); ++t) {
          uint64_t local_progress =
              progress.fetch_add(1, std::memory_order_relaxed);
          if (local_progress > 0 && local_progress % (uint64_t(1) << 24) == 0) {
            LOG(INFO) << std::format(
                "Progress: {}/{} = {:.2f}%", local_progress, num_iterations,
                static_cast<double>(local_progress) / num_iterations * 100.0);
          }
          if (early_break) {
            break;
          }
          // Scrambled visit order.
          const uint64_t scrambled = (t * kScramblePrime) % num_iterations;
          const Tensor<P, NA, NB, NC> tensor_t = FoldForcedProducts(
              part.r2p, r1_bc_data, part.r1_bc_rows, bit_width, scrambled);
          int remaining = RankLowerBoundFlatten<P, NA, NB, NC>(
              tensor_t, local_min - part.r1_bc_rows);
          local_min = std::min(local_min, part.r1_bc_rows + remaining);
          if (local_min <= known_lower_bound) {
            early_break = true;
          }
        }
        return local_min;
      },
      [](int a, int b) { return std::min(a, b); });

  const auto end_time = std::chrono::steady_clock::now();
  const auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time)
          .count();
  if (duration_ms > 1000) {
    LOG(INFO) << std::format("RankLowerBoundForcedProductA. rank={} dur={:.2f}",
                             rank_lower_bound, duration_ms / 1000.0);
  }

  // An early break means we proved the minimum can't beat known_lower_bound, so
  // this position yields no improvement.
  if (early_break.load()) {
    return known_lower_bound;
  }
  return std::max(known_lower_bound, rank_lower_bound);
}
