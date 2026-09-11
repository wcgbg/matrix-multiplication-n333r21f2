#pragma once

// Occupied singles/pairs and the paper's dmax = k and dmax = k-1 cases.
// Template definitions included by core/rank_one_span_family_search.h.

#include "core/rank_one_span_family_search.h"

namespace rank_one_span_internal {

// ---- k = 1 and k = 2: all singles / pairs of occupied keys ----

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseK1() const {
  RunCase(occupied_.size(), [&](SearchCtx &ctx, std::size_t idx) {
    const uint64_t q = occupied_[idx];
    Charge(ctx, 1);
    if (cmult_[q] < need_) {
      return;
    }
    ctx.qb[0] = q;
    Submit(ctx, ctx.qb.data(), 1);
  });
}

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseK2() const {
  RunCase(occupied_.size(), [&](SearchCtx &ctx, std::size_t i) {
    const uint64_t q1 = occupied_[i];
    for (std::size_t j = i + 1; j < occupied_.size(); ++j) {
      if (Stopped()) {
        return;
      }
      Charge(ctx, 1);
      const uint64_t q2 = occupied_[j];
      if (cmult_[q1] + cmult_[q2] + cmult_[q1 ^ q2] < need_) {
        continue;
      }
      ctx.qb[0] = q1;
      ctx.qb[1] = q2;
      if (Submit(ctx, ctx.qb.data(), 2)) {
        return;
      }
    }
  });
}

// ---- dmax = k: U-bar inside one family image ----

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseDmaxK() const {
  const auto tasks = PivotTasks(k_);
  RunCase(tasks.size(), [&](SearchCtx &ctx, std::size_t ti) {
    const auto [f, p0] = tasks[ti];
    const Family &fam = families_[f];
    ForEachSubspaceRREF(
        fam.image.Dim(), k_,
        [&](const std::vector<uint64_t> &rows) {
          if (Stopped()) {
            return true;
          }
          Charge(ctx, 1);
          RowsToKeys(fam, rows, k_, ctx.qb.data());
          return Submit(ctx, ctx.qb.data(), k_);
        },
        p0);
  });
}

// ---- dmax = k-1: a (k-1)-subspace of one image plus one occupied
// generator ----

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseDmaxKm1() const {
  const auto tasks = PivotTasks(k_ - 1);
  RunCase(tasks.size(), [&](SearchCtx &ctx, std::size_t ti) {
    const auto [f, p0] = tasks[ti];
    const Family &fam = families_[f];
    ForEachSubspaceRREF(
        fam.image.Dim(), k_ - 1,
        [&](const std::vector<uint64_t> &rows) {
          if (Stopped()) {
            return true;
          }
          Charge(ctx, 1);
          uint64_t sb[kMaxFamilyK];
          RowsToKeys(fam, rows, k_ - 1, sb);
          const int snpts = 1 << (k_ - 1);
          uint64_t spts[1 << (kMaxFamilyK - 1)];
          ExpandPoints(sb, k_ - 1, spts);
          int64_t base = 0;
          for (int i = 1; i < snpts; ++i) {
            base += cmult_[spts[i]];
          }
          KeyBasis sbasis;
          for (int i = 0; i < k_ - 1; ++i) {
            sbasis.Insert(sb[i]);
          }
          // S-level analysis: witnesses here have dmax = k − 1, so
          // d_f(Ū) ≤ min(d_f(S) + [g grows f], k − 1, dim I_f) with the
          // growth lemma (3): skip S when full growth cannot reach the
          // counting bound; require each g to grow ≥ gneed families.
          const GrowthPlan plan = PrepareGrowth(
              ctx, sb, k_ - 1, k_ - 1,
              static_cast<uint64_t>(num_families_) * (odim_ + k_));
          if (plan.gneed > plan.ngrow) {
            return false; // full growth cannot reach `need`
          }
          for (uint64_t g : occupied_) {
            if (Stopped()) {
              return true;
            }
            Charge(ctx, 2);
            if (sbasis.Contains(g)) {
              continue;
            }
            if (!GeneratorGrows(ctx, g, plan)) {
              continue;
            }
            int64_t cnt = base;
            for (int i = 0; i < snpts; ++i) {
              cnt += cmult_[spts[i] ^ g];
            }
            Charge(ctx, snpts);
            if (cnt < need_) {
              continue;
            }
            for (int i = 0; i < k_ - 1; ++i) {
              ctx.qb[i] = sb[i];
            }
            ctx.qb[k_ - 1] = g;
            if (Submit(ctx, ctx.qb.data(), k_)) {
              return true;
            }
          }
          return false;
        },
        p0);
  });
}

} // namespace rank_one_span_internal
