#pragma once

// Shared vocabulary of the rank-one-span exclusion engines
// (core/rank_lower_bound_rank_one_span.h for 𝔽₂, core/rank_lower_bound_rank_one_span_fp.h
// for other prime fields): the verdict type and saturating counters.

#include <cstdint>
#include <limits>

enum class RankOneSpanResult {
  kExcluded,     // no rank-one-spanned U with V ⊆ U, dim U ≤ target exists:
                 // proves rank(T) ≥ target + 1
  kWitnessFound, // some U is rank-one spanned: rank(T) ≤ dim U ≤ target
  kOverBudget    // enumeration would exceed the caller's budget; no claim
};

namespace rank_one_span_common {

inline uint64_t SatAdd(uint64_t a, uint64_t b) {
  return a > std::numeric_limits<uint64_t>::max() - b
             ? std::numeric_limits<uint64_t>::max()
             : a + b;
}

inline uint64_t SatMul(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  return a > std::numeric_limits<uint64_t>::max() / b
             ? std::numeric_limits<uint64_t>::max()
             : a * b;
}

// 2^k, saturating.
inline uint64_t SatPow2(int k) {
  return k >= 64 ? std::numeric_limits<uint64_t>::max() : uint64_t{1} << k;
}

// q^k, saturating.
inline uint64_t SatPow(uint64_t q, int k) {
  uint64_t r = 1;
  for (int i = 0; i < k; ++i) {
    r = SatMul(r, q);
  }
  return r;
}

} // namespace rank_one_span_common
