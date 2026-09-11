#pragma once

// Shared family membership and line-root enumeration for the named cases.
// Template definitions included by core/rank_one_span_family_search.h.

#include "core/rank_one_span_family_search.h"

namespace rank_one_span_internal {

// The occupied keys of multiplicity >= 2 (the only candidates for a point
// shared by two families).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
std::vector<uint64_t> FamilySearch<P, NA, NB, NC>::CollectMulti2() const {
  std::vector<uint64_t> multi2;
  for (uint64_t q : occupied_) {
    if (cmult_[q] >= 2) {
      multi2.push_back(q);
    }
  }
  return multi2;
}

// Per family, its lines {x, a, x^a} as the pairs (x, a) with x < a < x^a.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
std::vector<std::vector<typename FamilySearch<P, NA, NB, NC>::Line>>
FamilySearch<P, NA, NB, NC>::CollectFamilyLines() const {
  std::vector<std::vector<Line>> lines(static_cast<std::size_t>(num_families_) +
                                       1);
  for (int f = 1; f <= num_families_; ++f) {
    for (uint64_t x : families_[f].elems) {
      if (x == 0) {
        continue;
      }
      for (uint64_t a : families_[f].elems) {
        if (a > x && (a ^ x) > a) {
          lines[f].push_back({x, a});
        }
      }
    }
  }
  return lines;
}

// Every line through two multiplicity >= 2 keys, once (via its two smallest
// points), with the number of families whose image contains it and the
// first two such families: the root collection of the gamma, delta and C10
// cases. Charged as those collections were (1 per pair before the dedupe,
// 2 per family per surviving pair) on a private context, flushed at the
// end.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
template <class Visit>
void FamilySearch<P, NA, NB, NC>::ForEachMulti2Line(const Visit &visit) const {
  SearchCtx ctx = MakeCtx();
  for (std::size_t i = 0; i < multi2_.size(); ++i) {
    for (std::size_t j = i + 1; j < multi2_.size(); ++j) {
      Charge(ctx, 1);
      const uint64_t a = multi2_[i], b = multi2_[j];
      if ((a ^ b) < b) {
        continue;
      }
      int common = 0, f1 = 0, f2 = 0;
      for (int f = 1; f <= num_families_; ++f) {
        if (families_[f].image.Contains(a) && families_[f].image.Contains(b)) {
          ++common;
          (f1 == 0 ? f1 : f2) = f;
        }
      }
      Charge(ctx, static_cast<uint64_t>(num_families_) * 2);
      visit(a, b, common, f1, f2);
    }
  }
  Flush(ctx);
}

// The families whose image contains q (no charge; callers charge).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
int FamilySearch<P, NA, NB, NC>::FamiliesOf(uint64_t q, int *out) const {
  int cnt = 0;
  for (int f = 1; f <= num_families_; ++f) {
    if (families_[f].image.Contains(q)) {
      out[cnt++] = f;
    }
  }
  return cnt;
}

} // namespace rank_one_span_internal
