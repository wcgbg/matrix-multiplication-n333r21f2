#pragma once

// Dimension-five three-space cases: C3, C3', C4, C5, C9 (paper/main.tex).
// Template definitions included by core/rank_one_span_family_search.h.

#include "core/rank_one_span_family_search.h"

namespace rank_one_span_internal {

// (C3) two 3-subspaces through a shared multi-point, distinct families.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC3() const {
  {
    struct C3Task {
      uint64_t p;
      int f1, f2;
    };
    std::vector<C3Task> tasks;
    std::array<int, std::size_t{1} << kMaxFamilySideDim> pf;
    for (uint64_t p : multi2_) {
      const int npf = FamiliesOf(p, pf.data());
      for (int i = 0; i < npf; ++i) {
        for (int j = i + 1; j < npf; ++j) {
          tasks.push_back({p, pf[i], pf[j]});
        }
      }
    }
    RunCase(tasks.size(), [&](SearchCtx &ctx, std::size_t ti) {
      const C3Task t = tasks[ti];
      For3SubsThrough(t.f1, t.p, [&](uint64_t a1, uint64_t a2) {
        if (Stopped()) {
          return true;
        }
        Charge(ctx, 1);
        return For3SubsThrough(t.f2, t.p, [&](uint64_t b1, uint64_t b2) {
          if (Stopped()) {
            return true;
          }
          Charge(ctx, 6);
          KeyBasis x;
          x.Insert(t.p);
          x.Insert(a1);
          x.Insert(a2);
          x.Insert(b1);
          x.Insert(b2);
          if (x.Dim() == 5) {
            for (int i = 0; i < 5; ++i) {
              ctx.qb[i] = x.rows[i];
            }
            return Submit(ctx, ctx.qb.data(), 5);
          }
          if (x.Dim() == 4) {
            uint64_t qb4[4] = {x.rows[0], x.rows[1], x.rows[2], x.rows[3]};
            return Complete4Cap3(ctx, qb4);
          }
          return false; // dim 3 (equal): C3' covers
        });
      });
    });
  }
}

// (C3') a 3-space lying in two images, plus two generators.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC3Prime() const {
  {
    std::vector<std::array<uint64_t, 3>> roots;
    for (int f1 = 1; f1 <= num_families_; ++f1) {
      for (int f2 = f1 + 1; f2 <= num_families_; ++f2) {
        KeyBasis inter;
        for (uint64_t q : families_[f1].elems) {
          if (q && families_[f2].image.Contains(q)) {
            inter.Insert(q);
          }
        }
        if (inter.Dim() < 3) {
          continue;
        }
        ForEachSubspaceRREF(inter.Dim(), 3,
                            [&](const std::vector<uint64_t> &rows) {
                              std::array<uint64_t, 3> d{};
                              for (int i = 0; i < 3; ++i) {
                                for (int j = 0; j < inter.Dim(); ++j) {
                                  if ((rows[i] >> j) & 1) {
                                    d[i] ^= inter.rows[j];
                                  }
                                }
                              }
                              roots.push_back(d);
                              return false;
                            });
      }
    }
    RunCase(roots.size(), [&](SearchCtx &ctx, std::size_t ri) {
      Complete3TwoGens(ctx, roots[ri].data());
    });
  }
}

// (C4) 3-subspace through a multi-point z plus a line of another family
// through z, plus a generator.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC4() const {
  {
    struct C4Task {
      uint64_t z;
      int fd, fl;
    };
    std::vector<C4Task> tasks;
    std::array<int, std::size_t{1} << kMaxFamilySideDim> pf;
    for (uint64_t z : multi2_) {
      const int npf = FamiliesOf(z, pf.data());
      for (int i = 0; i < npf; ++i) {
        for (int j = 0; j < npf; ++j) {
          if (i != j) {
            tasks.push_back({z, pf[i], pf[j]});
          }
        }
      }
    }
    RunCase(tasks.size(), [&](SearchCtx &ctx, std::size_t ti) {
      const C4Task t = tasks[ti];
      For3SubsThrough(t.fd, t.z, [&](uint64_t a1, uint64_t a2) {
        if (Stopped()) {
          return true;
        }
        Charge(ctx, 1);
        KeyBasis dspan;
        dspan.Insert(t.z);
        dspan.Insert(a1);
        dspan.Insert(a2);
        for (uint64_t a : families_[t.fl].elems) {
          if (Stopped()) {
            return true;
          }
          if (a == 0 || a == t.z || (a ^ t.z) < a) {
            continue;
          }
          Charge(ctx, 1);
          if (dspan.Contains(a)) {
            continue;
          }
          const uint64_t qb4[4] = {t.z, a1, a2, a};
          if (Complete4Cap3(ctx, qb4)) {
            return true;
          }
        }
        return false;
      });
    });
  }
}

// (C5) occupied cross-family triangle (z, l, z⊕l) with a 3-subspace
// through z; base D + ⟨l⟩, plus a generator.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC5() const {
  RunCase(occupied_.size(), [&](SearchCtx &ctx, std::size_t zi) {
    const uint64_t z = occupied_[zi];
    std::array<int, std::size_t{1} << kMaxFamilySideDim> zf;
    int nzf = -1;
    for (uint64_t l : occupied_) {
      if (Stopped()) {
        return;
      }
      Charge(ctx, 1);
      const uint64_t lp = z ^ l;
      if (l == z || !Occ(lp)) {
        continue;
      }
      if (nzf < 0) {
        nzf = FamiliesOf(z, zf.data());
        Charge(ctx, num_families_);
      }
      for (int zi2 = 0; zi2 < nzf; ++zi2) {
        const int fz = zf[zi2];
        // need fl ∋ l, fl ≠ fz, and some f3 ∋ z⊕l outside {fz, fl}
        bool ok = false;
        if (!fmask_.empty()) {
          const uint8_t ml = fmask_[l], mp = fmask_[lp];
          const uint8_t bz = static_cast<uint8_t>(1) << (fz - 1);
          for (int fl = 1; fl <= num_families_ && !ok; ++fl) {
            const uint8_t bl = static_cast<uint8_t>(1) << (fl - 1);
            if (fl != fz && (ml & bl) && (mp & ~(bz | bl))) {
              ok = true;
            }
          }
        } else {
          ok = true; // no cheap check; the exact filters below decide
        }
        if (!ok) {
          continue;
        }
        const bool stop = For3SubsThrough(fz, z, [&](uint64_t a1, uint64_t a2) {
          if (Stopped()) {
            return true;
          }
          Charge(ctx, 2);
          KeyBasis dspan;
          dspan.Insert(z);
          dspan.Insert(a1);
          dspan.Insert(a2);
          if (dspan.Contains(l)) {
            return false;
          }
          const uint64_t qb4[4] = {z, a1, a2, l};
          return Complete4Cap3(ctx, qb4);
        });
        if (stop) {
          return;
        }
      }
    }
  });
}

// (C9) |T3| = 1 with every other family's line inside the 3-dim
// intersection D: Ū = D + two occupied generators. Roots: 3-subspaces D
// of an image that carry full lines (≥ 3 points) of at least
// need − F − 2 other families (the counting bound for this shape).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC9() const {
  {
    const int t9 = std::max(need_ - num_families_ - 2, 1);
    const auto tasks = PivotTasks(3);
    RunCase(tasks.size(), [&](SearchCtx &ctx, std::size_t ti) {
      const auto [f, p0] = tasks[ti];
      const Family &fam = families_[f];
      ForEachSubspaceRREF(
          fam.image.Dim(), 3,
          [&](const std::vector<uint64_t> &rows) {
            if (Stopped()) {
              return true;
            }
            Charge(ctx, 8);
            uint64_t d3[3];
            RowsToKeys(fam, rows, 3, d3);
            uint64_t pts7[7];
            for (int mask = 1; mask < 8; ++mask) {
              uint64_t p = 0;
              for (int i = 0; i < 3; ++i) {
                if ((mask >> i) & 1) {
                  p ^= d3[i];
                }
              }
              pts7[mask - 1] = p;
            }
            int others = 0;
            if (!fmask_.empty()) {
              uint64_t acc = 0;
              for (const uint64_t q : pts7) {
                acc += spread_[fmask_[q]];
              }
              for (int g = 1; g <= num_families_; ++g) {
                if (g != f &&
                    static_cast<int>((acc >> (8 * (g - 1))) & 0xFF) >= 3) {
                  ++others;
                }
              }
            } else {
              for (int g = 1; g <= num_families_; ++g) {
                if (g == f) {
                  continue;
                }
                int cnt = 0;
                for (const uint64_t q : pts7) {
                  cnt += families_[g].image.Contains(q) ? 1 : 0;
                }
                if (cnt >= 3) {
                  ++others;
                }
              }
              Charge(ctx, static_cast<uint64_t>(num_families_) * 7);
            }
            if (others < t9) {
              return false;
            }
            return Complete3TwoGens(ctx, d3);
          },
          p0);
    });
  }
}

} // namespace rank_one_span_internal
