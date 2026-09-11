#pragma once

// Dimension-five line arrangements: C6, C7, C8, C10 (paper/main.tex).
// Template definitions included by core/rank_one_span_family_search.h.

#include "core/rank_one_span_family_search.h"

namespace rank_one_span_internal {

// (C6) disjoint line pair from two distinct families, plus a generator.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC6() const {
  {
    struct C6Task {
      int f1, f2;
      uint32_t l1;
    };
    std::vector<C6Task> tasks;
    for (int f1 = 1; f1 <= num_families_; ++f1) {
      for (int f2 = f1 + 1; f2 <= num_families_; ++f2) {
        for (uint32_t i = 0; i < flines_[f1].size(); ++i) {
          tasks.push_back({f1, f2, i});
        }
      }
    }
    RunCase(tasks.size(), [&](SearchCtx &ctx, std::size_t ti) {
      const C6Task t = tasks[ti];
      const uint64_t x = flines_[t.f1][t.l1][0], a = flines_[t.f1][t.l1][1];
      const uint64_t xa = x ^ a;
      for (const auto &l2 : flines_[t.f2]) {
        if (Stopped()) {
          return;
        }
        Charge(ctx, 1);
        const uint64_t y = l2[0], b = l2[1];
        if (y == x || y == a || y == xa) {
          continue;
        }
        if (b == x || b == a || b == xa || b == (x ^ y) || b == (a ^ y) ||
            b == (xa ^ y)) {
          continue;
        }
        const uint64_t qb4[4] = {x, a, y, b};
        if (Complete4Cap3(ctx, qb4)) {
          return;
        }
      }
    });
  }
}

// (C7) a 3-space W carrying lines of ≥ max(need − F, 3) families
// (roots from bouquet spans), plus two generators.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC7() const {
  {
    const int t7 = std::max(need_ - num_families_, 3);
    std::vector<std::array<uint64_t, 3>> wroots;
    {
      SearchCtx ctx = MakeCtx();
      std::array<int, std::size_t{1} << kMaxFamilySideDim> pf;
      for (uint64_t p : multi2_) {
        const int npf = FamiliesOf(p, pf.data());
        for (int i = 0; i < npf; ++i) {
          for (int j = i + 1; j < npf; ++j) {
            const Family &f1 = families_[pf[i]], &f2 = families_[pf[j]];
            for (uint64_t a : f1.elems) {
              if (a == 0 || a == p || (a ^ p) < a) {
                continue;
              }
              for (uint64_t b : f2.elems) {
                Charge(ctx, 1);
                if (b == 0 || b == p || (b ^ p) < b) {
                  continue;
                }
                if (b == a || b == (a ^ p)) {
                  continue;
                }
                if (!fmask_.empty()) {
                  // count families with a full line inside W (≥ 3 points)
                  const uint64_t pts7[7] = {p,     a,     b,        p ^ a,
                                            p ^ b, a ^ b, p ^ a ^ b};
                  uint64_t acc = 0;
                  for (const uint64_t q : pts7) {
                    acc += spread_[fmask_[q]];
                  }
                  Charge(ctx, 7);
                  int nl = 0;
                  for (int f = 1; f <= num_families_; ++f) {
                    if (static_cast<int>((acc >> (8 * (f - 1))) & 0xFF) >= 3) {
                      ++nl;
                    }
                  }
                  if (nl < t7) {
                    continue;
                  }
                }
                wroots.push_back({p, a, b});
              }
            }
          }
        }
      }
      Flush(ctx);
    }
    RunCase(wroots.size(), [&](SearchCtx &ctx, std::size_t ri) {
      Complete3TwoGens(ctx, wroots[ri].data());
    });
  }
}

// (C8) all lines through one key q of multiplicity ≥ max(need − F, 3):
// three independent line directions from distinct families + generator.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC8() const {
  {
    const int t8 = std::max(need_ - num_families_, 3);
    std::vector<uint64_t> qroots;
    for (uint64_t q : occupied_) {
      if (cmult_[q] >= t8) {
        qroots.push_back(q);
      }
    }
    RunCase(qroots.size(), [&](SearchCtx &ctx, std::size_t qi) {
      const uint64_t q = qroots[qi];
      std::array<int, std::size_t{1} << kMaxFamilySideDim> pf;
      const int npf = FamiliesOf(q, pf.data());
      Charge(ctx, num_families_);
      for (int i = 0; i < npf; ++i) {
        for (int j = i + 1; j < npf; ++j) {
          for (int m = j + 1; m < npf; ++m) {
            const Family &f1 = families_[pf[i]], &f2 = families_[pf[j]],
                         &f3 = families_[pf[m]];
            for (uint64_t a : f1.elems) {
              if (a == 0 || a == q || (a ^ q) < a) {
                continue;
              }
              for (uint64_t b : f2.elems) {
                if (b == 0 || b == q || (b ^ q) < b) {
                  continue;
                }
                for (uint64_t c : f3.elems) {
                  if (Stopped()) {
                    return;
                  }
                  Charge(ctx, 1);
                  if (c == 0 || c == q || (c ^ q) < c) {
                    continue;
                  }
                  KeyBasis x;
                  x.Insert(q);
                  x.Insert(a);
                  x.Insert(b);
                  x.Insert(c);
                  if (x.Dim() != 4) {
                    continue;
                  }
                  const uint64_t qb4[4] = {q, a, b, c};
                  if (Complete4Cap3(ctx, qb4)) {
                    return;
                  }
                }
              }
            }
          }
        }
      }
    });
  }
}

// (C10) all T2 intersections are one line repeated in at least
// need − F families: the line plus three occupied generators.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
void FamilySearch<P, NA, NB, NC>::CaseC10() const {
  {
    const int t10 = std::max(need_ - num_families_, 4);
    std::vector<Line> roots;
    ForEachMulti2Line([&](uint64_t a, uint64_t b, int common, int, int) {
      if (common >= t10) {
        roots.push_back({a, b});
      }
    });
    RunCase(roots.size(), [&](SearchCtx &ctx, std::size_t ri) {
      Complete2ThreeGens(ctx, roots[ri].data());
    });
  }
}

} // namespace rank_one_span_internal
