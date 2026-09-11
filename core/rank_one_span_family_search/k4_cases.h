#pragma once

// Dimension-four remainder: gamma, alpha, beta, delta (paper/main.tex).
// Template definitions included by core/rank_one_span_family_search.h.

#include "core/rank_one_span_family_search.h"

namespace rank_one_span_internal {

// (γ) roots: lines shared by ≥ 3 families (a, b ∈ I_f implies
// t = a ⊕ b ∈ I_f, so the pair decides membership). Collected serially
// (cheap), completed in parallel with two generators.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseGamma() const {
  std::vector<std::pair<uint64_t, uint64_t>> gamma_roots;
  ForEachMulti2Line([&](uint64_t a, uint64_t b, int common, int, int) {
    if (common >= 3) {
      gamma_roots.push_back({a, b});
    }
  });
  RunCase(gamma_roots.size(), [&](SearchCtx &ctx, std::size_t ri) {
    const auto [a, b] = gamma_roots[ri];
    KeyBasis line;
    line.Insert(a);
    line.Insert(b);
    for (std::size_t gi = 0; gi < occupied_.size(); ++gi) {
      if (Stopped()) {
        return;
      }
      const uint64_t g1 = occupied_[gi];
      if (line.Contains(g1)) {
        continue;
      }
      KeyBasis ext = line;
      ext.Insert(g1);
      for (std::size_t gj = gi + 1; gj < occupied_.size(); ++gj) {
        if (Stopped()) {
          return;
        }
        Charge(ctx, 1);
        const uint64_t g2 = occupied_[gj];
        if (ext.Contains(g2)) {
          continue;
        }
        ctx.qb[0] = a;
        ctx.qb[1] = b;
        ctx.qb[2] = g1;
        ctx.qb[3] = g2;
        if (Submit(ctx, ctx.qb.data(), 4)) {
          return;
        }
      }
    }
  });
}

// (α) two distinct intersecting lines through a multi-point, plus one
// generator.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseAlpha() const {
  RunCase(multi2_.size(), [&](SearchCtx &ctx, std::size_t pi) {
    const uint64_t p = multi2_[pi];
    std::array<int, std::size_t{1} << kMaxFamilySideDim> pf;
    const int npf = FamiliesOf(p, pf.data());
    Charge(ctx, num_families_);
    for (int i = 0; i < npf; ++i) {
      for (int j = i + 1; j < npf; ++j) {
        const Family &f1 = families_[pf[i]], &f2 = families_[pf[j]];
        for (uint64_t a : f1.elems) {
          if (Stopped()) {
            return;
          }
          if (a == 0 || a == p || (a ^ p) < a) {
            continue; // line {p, a, a^p} once
          }
          for (uint64_t b : f2.elems) {
            if (Stopped()) {
              return;
            }
            Charge(ctx, 1);
            if (b == 0 || b == p || (b ^ p) < b) {
              continue;
            }
            if (b == a || b == (a ^ p)) {
              continue; // X must have dim 3
            }
            // Per-X analysis (sound for the all-d_f≤2 shape): d_f(Ū) ≤
            // min(d_f(X) + [g grows f], 2, dim I_f), growth lemma (3).
            const uint64_t base[3] = {p, a, b};
            const GrowthPlan plan = PrepareGrowth(
                ctx, base, 3, 2, static_cast<uint64_t>(num_families_) * odim_);
            if (plan.gneed > plan.ngrow) {
              continue;
            }
            KeyBasis xb;
            xb.Insert(p);
            xb.Insert(a);
            xb.Insert(b);
            for (uint64_t g : occupied_) {
              if (Stopped()) {
                return;
              }
              Charge(ctx, 2);
              if (xb.Contains(g)) {
                continue;
              }
              if (!GeneratorGrows(ctx, g, plan)) {
                continue;
              }
              ctx.qb[0] = p;
              ctx.qb[1] = a;
              ctx.qb[2] = b;
              ctx.qb[3] = g;
              if (Submit(ctx, ctx.qb.data(), 4)) {
                return;
              }
            }
          }
        }
      }
    }
  });
}

// (β) two disjoint lines from two distinct families, rooted at an
// occupied ordered pair (x, y) with x ⊕ y occupied: base {x, a, y} plus
// one generator b from the second family's image, with the growth filter.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseBeta() const {
  RunCase(occupied_.size(), [&](SearchCtx &ctx, std::size_t xi) {
    const uint64_t x = occupied_[xi];
    std::array<int, std::size_t{1} << kMaxFamilySideDim> pfx, pfy;
    int npfx = 0;
    bool npfx_done = false;
    for (uint64_t y : occupied_) {
      if (Stopped()) {
        return;
      }
      Charge(ctx, 1);
      if (y == x || !Occ(x ^ y)) {
        continue;
      }
      if (!npfx_done) {
        npfx = FamiliesOf(x, pfx.data());
        Charge(ctx, num_families_);
        npfx_done = true;
      }
      const int npfy = FamiliesOf(y, pfy.data());
      Charge(ctx, num_families_);
      for (int i = 0; i < npfx; ++i) {
        for (int j = 0; j < npfy; ++j) {
          if (pfx[i] == pfy[j]) {
            continue;
          }
          const Family &f1 = families_[pfx[i]], &f2 = families_[pfy[j]];
          for (uint64_t a : f1.elems) {
            if (Stopped()) {
              return;
            }
            if (a == 0 || a == x || (a ^ x) < a) {
              continue;
            }
            KeyBasis xb;
            xb.Insert(x);
            xb.Insert(a);
            if (xb.Contains(y)) {
              continue; // need dim 4 overall
            }
            // Growth filter on the base {x, a, y} (all-d_f≤2 shape):
            // d_f(Ū) ≤ min(d_f(base) + [b grows f], 2, dim I_f).
            const uint64_t base[3] = {x, a, y};
            const GrowthPlan plan = PrepareGrowth(
                ctx, base, 3, 2, static_cast<uint64_t>(num_families_) * odim_);
            if (plan.gneed > plan.ngrow) {
              continue;
            }
            KeyBasis xyb = xb;
            xyb.Insert(y);
            for (uint64_t b : f2.elems) {
              if (Stopped()) {
                return;
              }
              Charge(ctx, 2);
              if (b == 0 || b == y || (b ^ y) < b) {
                continue;
              }
              if (xyb.Contains(b)) {
                continue; // lines not disjoint / dim < 4
              }
              if (!GeneratorGrows(ctx, b, plan)) {
                continue;
              }
              ctx.qb[0] = x;
              ctx.qb[1] = a;
              ctx.qb[2] = y;
              ctx.qb[3] = b;
              if (Submit(ctx, ctx.qb.data(), 4)) {
                return;
              }
            }
          }
        }
      }
    }
  });
}

// (δ) exactly two distinct lines among the ≥ 3 family intersections, so
// one line D lies in exactly two images (three or more is γ) and the
// other, D', belongs to a third family and is disjoint from D (a common
// point is α): Ū = D ⊕ D'. Roots: lines with exactly two families.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseDelta() const {
  {
    struct DeltaRoot {
      uint64_t a, b;
      int f1, f2;
    };
    std::vector<DeltaRoot> delta_roots;
    ForEachMulti2Line([&](uint64_t a, uint64_t b, int common, int f1, int f2) {
      if (common == 2) {
        delta_roots.push_back({a, b, f1, f2});
      }
    });
    RunCase(delta_roots.size(), [&](SearchCtx &ctx, std::size_t ri) {
      const DeltaRoot t = delta_roots[ri];
      KeyBasis line;
      line.Insert(t.a);
      line.Insert(t.b);
      for (int f3 = 1; f3 <= num_families_; ++f3) {
        if (f3 == t.f1 || f3 == t.f2) {
          continue;
        }
        for (const auto &l2 : flines_[f3]) {
          if (Stopped()) {
            return;
          }
          Charge(ctx, 1);
          KeyBasis x = line;
          if (!x.Insert(l2[0]) || !x.Insert(l2[1])) {
            continue; // not disjoint
          }
          ctx.qb[0] = t.a;
          ctx.qb[1] = t.b;
          ctx.qb[2] = l2[0];
          ctx.qb[3] = l2[1];
          if (Submit(ctx, ctx.qb.data(), 4)) {
            return;
          }
        }
      }
    });
  }
}

} // namespace rank_one_span_internal
