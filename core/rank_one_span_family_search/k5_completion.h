#pragma once

// Shared completions and three-subspace enumeration for cases C3..C10.
// Template definitions included by core/rank_one_span_family_search.h.

#include "core/rank_one_span_family_search.h"

namespace rank_one_span_internal {

// Prefilter for a 4-dim base X: Σ_f min(d_f(X)+1, 3, dim I_f) ≥ need is
// necessary for any all-d≤3 witness X + ⟨g⟩ (growth lemma). Uses the
// point-count cap for d_f(X); a large sentinel when fmask_ is absent.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
int FamilySearch<P, NA, NB, NC>::FamCap4(SearchCtx &ctx,
                                         const uint64_t *qb4) const {
  if (fmask_.empty()) {
    return 1 << 20;
  }
  ExpandPoints(qb4, 4, ctx.pts.data());
  uint64_t acc = 0;
  for (int i = 1; i < 16; ++i) {
    acc += spread_[fmask_[ctx.pts[i]]];
  }
  Charge(ctx, 16);
  int cap = 0;
  for (int f = 1; f <= num_families_; ++f) {
    const int nf = static_cast<int>((acc >> (8 * (f - 1))) & 0xFF);
    const int bf = nf >= 7 ? 3 : (nf >= 3 ? 2 : (nf >= 1 ? 1 : 0));
    cap += std::min({bf + 1, 3, families_[f].image.Dim()});
  }
  return cap;
}

// Complete a 4-dim independent base with one occupied generator under
// the all-d≤3 caps (growth filter as in the k = 4 cases, cap 3).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
bool FamilySearch<P, NA, NB, NC>::Complete4Cap3(SearchCtx &ctx,
                                                const uint64_t *qb4) const {
  if (FamCap4(ctx, qb4) < need_) {
    return Stopped();
  }
  const GrowthPlan plan = PrepareGrowth(
      ctx, qb4, 4, 3, static_cast<uint64_t>(num_families_) * (odim_ + 4));
  if (plan.gneed > plan.ngrow) {
    return Stopped();
  }
  KeyBasis xb;
  for (int i = 0; i < 4; ++i) {
    xb.Insert(qb4[i]);
  }
  for (uint64_t g : occupied_) {
    if (Stopped()) {
      return true;
    }
    Charge(ctx, 2);
    if (xb.Contains(g)) {
      continue;
    }
    if (!GeneratorGrows(ctx, g, plan)) {
      continue;
    }
    for (int i = 0; i < 4; ++i) {
      ctx.qb[i] = qb4[i];
    }
    ctx.qb[4] = g;
    if (Submit(ctx, ctx.qb.data(), 5)) {
      return true;
    }
  }
  return Stopped();
}

// Complete a 3-dim independent base with two occupied generators
// (visits ordered g1 then delegates; harmless duplicates).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
bool FamilySearch<P, NA, NB, NC>::Complete3TwoGens(SearchCtx &ctx,
                                                   const uint64_t *qb3) const {
  KeyBasis base;
  for (int i = 0; i < 3; ++i) {
    base.Insert(qb3[i]);
  }
  if (base.Dim() != 3) {
    return Stopped();
  }
  for (uint64_t g1 : occupied_) {
    if (Stopped()) {
      return true;
    }
    Charge(ctx, 2);
    if (base.Contains(g1)) {
      continue;
    }
    const uint64_t qb4[4] = {qb3[0], qb3[1], qb3[2], g1};
    if (Complete4Cap3(ctx, qb4)) {
      return true;
    }
  }
  return Stopped();
}

// Complete a line with three occupied generators. This is used only
// for the degenerate bouquet in which every T2 family cuts the witness
// in the same line (C10).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
bool FamilySearch<P, NA, NB, NC>::Complete2ThreeGens(
    SearchCtx &ctx, const uint64_t *qb2) const {
  KeyBasis base;
  base.Insert(qb2[0]);
  base.Insert(qb2[1]);
  if (base.Dim() != 2) {
    return Stopped();
  }
  for (uint64_t g1 : occupied_) {
    if (Stopped()) {
      return true;
    }
    Charge(ctx, 2);
    if (base.Contains(g1)) {
      continue;
    }
    const uint64_t qb3[3] = {qb2[0], qb2[1], g1};
    if (Complete3TwoGens(ctx, qb3)) {
      return true;
    }
  }
  return Stopped();
}

// Enumerate the 3-subspaces of I_f through p as {p, d1, d2}: 2-subspaces
// of I_f / ⟨p⟩ via a reduced basis. cb returns true to stop.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
template <class Cb>
bool FamilySearch<P, NA, NB, NC>::For3SubsThrough(int f, uint64_t p,
                                                  const Cb &cb) const {
  const Family &fam = families_[f];
  KeyBasis quo;
  for (int i = 0; i < fam.image.Dim(); ++i) {
    quo.Insert(std::min(fam.image.rows[i], fam.image.rows[i] ^ p));
  }
  if (quo.Dim() != fam.image.Dim() - 1 || quo.Dim() < 2) {
    return false; // p not in I_f, or image too small
  }
  return ForEachSubspaceRREF(quo.Dim(), 2,
                             [&](const std::vector<uint64_t> &rows) {
                               uint64_t d1 = 0, d2 = 0;
                               for (int j = 0; j < quo.Dim(); ++j) {
                                 if ((rows[0] >> j) & 1) {
                                   d1 ^= quo.rows[j];
                                 }
                                 if ((rows[1] >> j) & 1) {
                                   d2 ^= quo.rows[j];
                                 }
                               }
                               return cb(d1, d2);
                             });
}

} // namespace rank_one_span_internal
