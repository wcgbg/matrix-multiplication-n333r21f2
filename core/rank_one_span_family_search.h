#pragma once

// The v2 engine of the rank-one-span exclusion over 𝔽₂: the family-structure
// search (class FamilySearch below), for quotients beyond the reach of the v1
// enumeration in core/rank_one_span_f2.h. See
// core/rank_lower_bound_rank_one_span.h for the method and the public API.
// Trusted: the verifier re-runs this search to check a RankOneSpanProof.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "boost/unordered/unordered_flat_map.hpp"
#include "ng-log/logging.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "core/rank_one_span_common.h"
#include "core/rank_one_span_f2.h"

namespace rank_one_span_internal {

// The v2 engine: family-structure search, for quotients far beyond the v1
// enumeration's reach. Complete decision procedure for quotient dims
// k ≤ min(e, kMaxFamilyK); returns kOverBudget when its op counter trips the
// budget or when the runtime coverage arithmetic below cannot establish
// completeness (never an unsound verdict).
//
// Structure exploited: fix the *family side* as the smaller of the two core
// factors (dim fdim, F = 2^fdim − 1 nonzero "family" vectors f). For fixed f,
// the map φ_f(o) = key(rank-one built from f and o) is linear in o, so
//   - the occupied keys are ∪_f (I_f ∖ 0) with I_f = im φ_f a subspace,
//   - span(b0) ⊇ r(f, ker φ_f) for every f.
// For a witness U = V + lift(Ū) (dim ρ + k), with d_f := dim(I_f ∩ Ū):
//   (1) U is the full preimage of Ū, so U's rank-ones are exactly
//       { r(f, o) : φ_f(o) ∈ Ū }, and Ū is spanned by its occupied keys
//       (else they span a smaller space than U).
//   (2) Sharp counting: modulo span(b0), family f contributes at most d_f
//       dimensions (r(f,·) is injective and linear, and r(f, ker φ_f) lies in
//       span(b0)), hence Σ_f d_f ≥ R_k := ρ + k − d0. The cheap point-count
//       corollary: Σ_{q ∈ Ū∖0} cmult(q) ≥ Σ_f d_f, with
//       cmult(q) = #{f : q ∈ I_f}.
//   (3) Growth lemma (used by every "base + one generator" completion): for
//       Ū = span(B) + ⟨g⟩, d_f(Ū) exceeds dim(I_f ∩ span(B)) only if
//       g ∈ span(B) + I_f (a new x ∈ I_f ∩ Ū is σ ⊕ g with σ ∈ span(B)), and
//       by at most 1. So when the case shape caps every d_f, the generator
//       must "grow" a computable number of families — an early-exit
//       membership scan against the joined bases rejects almost every g.
// Enumeration by dmax = max_f d_f (each case emits a superset of the
// witnesses in that shape; overlaps and duplicates are harmless):
//   - k ≤ 2: all singles / pairs of occupied keys (complete outright by (1)).
//   - dmax = k: Ū is a k-dim subspace of one image → ForEachSubspaceRREF
//     inside each I_f.
//   - dmax = k−1: a (k−1)-subspace of some I_f plus one occupied generator
//     outside it (one exists by (1)), with the growth filter (3).
//   - k = 3 remainder (all d_f ≤ 1): Σ_f d_f ≤ F; impossible iff F < R_3 —
//     checked at runtime, else kOverBudget.
//   - k = 4 remainder (all d_f ≤ 2): L := #{f : d_f = 2} ≥ R_4 − F, and we
//     require R_4 − F ≥ 3 (else kOverBudget), so at least three families
//     have a 2-dim intersection ("line") with Ū. Case split on the set of
//     these lines (distinct lines, with multiplicities):
//       (γ) some line lies in ≥ 3 images: Ū = line + two occupied generators.
//       (α) else two DISTINCT lines (from distinct families) meet in a point
//           p, which has cmult ≥ 2: Ū = (line through p in I_f1) + (line
//           through p in I_f2) + one occupied generator, with the growth
//           filter (3) under the d_f ≤ 2 cap.
//       (β) else all distinct lines are pairwise disjoint; if there are three
//           distinct ones D_1, D_2, D_3: any p3 ∈ D_3∖0 splits uniquely as
//           x ⊕ y with x ∈ D_1∖0, y ∈ D_2∖0 (disjointness makes both parts
//           nonzero), so (x, y, x⊕y) is an occupied triple with distinct
//           families for x and y, and Ū = D_1 ⊕ D_2 = (line through x in
//           I_f1) + ⟨y⟩ + one more point of D_2 — i.e. base {x, a, y} plus
//           one generator b from I_f2, again with the growth filter (3).
//       (δ) else exactly two distinct lines D, D' (disjoint), one of which,
//           say D, is the intersection of two families (L ≥ 3 and no line in
//           ≥ 3 images): Ū = D ⊕ D' with D' a line of a third family.
//     These four enumerations together cover every all-d_f≤2 witness.
//   - k = 5 remainder (all d_f ≤ 3): see the case comment in the code (C3,
//     C3', C4, C5, C6, C7, C8, C9, C10), with the coverage argument spelled
//     out there; it requires R_5 − F ≥ 4 (else kOverBudget).
//
// Parallelism: with `parallel` set, each case's root loop runs under
// tbb::parallel_for with per-worker scratch. The verdict is order-independent
// (an exclusion means the exhaustive sweep found nothing; any witness is a
// witness). The op count is exact on the accept path (every worker completes
// its share, so the sum does not depend on scheduling) and only the early
// exits (witness found, budget tripped) are timing dependent; that is why
// the verifier can run the parallel engine (see
// subspace_bounds/verifier/verifier.h).
//
// Structure: the constructor builds the read-only tables (family images and
// element lists, occupancy, multiplicities, b0 and the key -> rank-one
// buckets); Run sweeps k = 1..e through RunK, which applies the coverage
// guards and runs that k's case methods through RunSteps, in the order of
// the completeness argument above; every candidate reaches CheckCandidate
// through Submit. Workers count ops on a SearchCtx and Flush them to the
// shared total; the charge schedule is pinned by FamilySearchGoldenTest.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
class FamilySearch {
public:
  using Row = WideBits<NB * NC>;
  using Span = SliceSpanA<P, NA, NB, NC>;
  using Line = std::array<uint64_t, 2>; // {x, a}: the line {x, a, x^a}

  // Preconditions (CHECKed): target_rank >= span.rho and the applicability
  // guards of RankOneSpanFamilySearch (e <= kMaxFamilyK, dq <= kMaxFamilyDq,
  // min(rb, rc) <= kMaxFamilySideDim; they bound the table sizes below).
  // Builds the read-only search tables and charges their cost to setup_ops_.
  FamilySearch(const Span &span, int target_rank, uint64_t op_budget,
               bool parallel)
      : span_(span), target_rank_(target_rank), op_budget_(op_budget),
        parallel_(parallel), rb_(span.rb), rc_(span.rc), rho_(span.rho),
        dq_(rb_ * rc_ - rho_), e_(std::min(target_rank_ - rho_, dq_)),
        family_on_b_(rb_ <= rc_), fdim_(family_on_b_ ? rb_ : rc_),
        odim_(family_on_b_ ? rc_ : rb_), num_families_((1 << fdim_) - 1),
        qsize_(std::size_t{1} << dq_) {
    CHECK_GE(target_rank_, rho_);
    CHECK_LE(e_, kMaxFamilyK);
    CHECK_LE(dq_, kMaxFamilyDq);
    CHECK_LE(std::min(rb_, rc_), kMaxFamilySideDim);
    ComputeFreePos();
    BuildFamilies();
    BuildBuckets();
  }

  // Sweeps k = 1..e (after the setup-budget and k = 0 checks); *ops_out
  // receives the op count.
  RankOneSpanResult Run(uint64_t *ops_out) {
    if (setup_ops_ > op_budget_) {
      *ops_out = setup_ops_;
      return RankOneSpanResult::kOverBudget;
    }

    // k = 0: U = V.
    if (d0_ == rho_) {
      *ops_out = setup_ops_;
      return RankOneSpanResult::kWitnessFound;
    }
    ops_total_.store(setup_ops_, std::memory_order_relaxed);

    for (int k = 1; k <= e_ && !Stopped(); ++k) {
      RunK(k);
    }
    return Finish(ops_out);
  }

private:
  // Per-family image basis and element list; global occupancy structures.
  // Everything built by the constructor is read-only during the case sweeps.
  struct Family {
    KeyBasis image;
    std::vector<uint64_t> elems; // all image elements including 0, ascending
  };

  // Per-worker mutable scratch; everything a case touches concurrently lives
  // here (the setup structures are read-only from then on).
  struct SearchCtx {
    std::array<uint64_t, std::size_t{1} << kMaxFamilyK> pts;
    std::array<uint64_t, kMaxFamilyK> qb;
    LinearBasis<NB * NC> scratch;
    std::vector<KeyBasis> jbases;
    std::vector<int> grow_list;
    uint64_t local_ops = 0;
  };

  // Workers buffer ops locally and flush every kFlushInterval (and at chunk
  // end); the over-budget verdict never overwrites a witness verdict.
  static constexpr uint64_t kFlushInterval = uint64_t{1} << 20;

  // A case method with its name, for RunSteps.
  struct Step {
    const char *name;
    void (FamilySearch::*fn)() const;
  };

  // ---- Driver ----

  // The verdict and the op total once every sweep has ended.
  RankOneSpanResult Finish(uint64_t *ops_out) const {
    const int final_verdict = verdict_.load(std::memory_order_relaxed);
    *ops_out = ops_total_.load(std::memory_order_relaxed);
    if (final_verdict == 1) {
      return RankOneSpanResult::kWitnessFound;
    }
    if (final_verdict == -1) {
      return RankOneSpanResult::kOverBudget;
    }
    return RankOneSpanResult::kExcluded;
  }

  // Quotient dimension k: singles / pairs for k <= 2; otherwise the coverage
  // guards, the dmax = k and dmax = k-1 cases, then the k = 4 or k = 5
  // remainder cases.
  void RunK(int k) {
    k_ = k;
    need_ = rho_ + k - d0_;
    if (k == 1) {
      CaseK1();
      return;
    }
    if (k == 2) {
      CaseK2();
      return;
    }
    // Coverage arithmetic for the shapes the cases below do not reach. The
    // verdict is still 0 here (the k loop's condition just passed and no
    // worker is alive) and nothing is charged before Finish(), so marking the
    // search over budget reports exactly the current op total.
    if (k == 3 && num_families_ >= need_) {
      MarkOverBudget(); // all-d_f<=1 not excludable
      return;
    }
    if (k == 4 && need_ - num_families_ < 3) {
      MarkOverBudget(); // cannot force L >= 3
      return;
    }
    ops_k_start_ = ops_total_.load(std::memory_order_relaxed);
    if (!RunSteps({{"dmax=k", &FamilySearch::CaseDmaxK},
                   {"dmax=k-1", &FamilySearch::CaseDmaxKm1}})) {
      return;
    }
    if (k == 5) {
      RunK5();
    } else if (k == 4) {
      RunK4();
    }
  }

  // Runs the case methods in order, stopping at the first verdict; returns
  // false iff the sweep stopped.
  bool RunSteps(std::initializer_list<Step> steps) const {
    for (const Step &step : steps) {
      (this->*step.fn)();
      if (Stopped()) {
        return false;
      }
      LOG(INFO) << "family search k=" << k_ << " " << step.name
                << " case done, ops="
                << ops_total_.load(std::memory_order_relaxed) - ops_k_start_;
    }
    return true;
  }

  // ---- k = 1 and k = 2: all singles / pairs of occupied keys ----

  void CaseK1() const {
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

  void CaseK2() const {
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

  void CaseDmaxK() const {
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

  void CaseDmaxKm1() const {
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

  // ---- k = 5 remainder, all d_f <= 3 (cases C3..C10) ----

  // Completeness (see also the k = 4 header comment; d_f ≤ 3 here because
  // the dmax = 5 and dmax = 4 cases ran above; Σ_f d_f ≥ need requires
  // need ≥ F + 4, checked below):
  //  |T3| ≥ 2 (two families with d = 3): two 3-dim intersections in a
  //    5-space share a point (3+3 > 5) → C3 anchored at a multi-point;
  //    equal intersections lie in an image intersection → C3'.
  //  |T3| = 1 (D the 3-dim intersection, family f_D): the counting gives
  //    |T2| ≥ 2 (2|T3| + |T2| ≥ 4 from need ≥ F + 4), and more precisely
  //    |T2| ≥ need − F − 2. Each line of a T2 family meets D in a point,
  //    is disjoint from D, or lies inside D. Some line meets D in exactly
  //    a (multi-)point z → C4 (D through z, the line through z, one
  //    generator). Some line is disjoint from D → C6 (that line and any
  //    line of f_D inside D form a disjoint pair from distinct families;
  //    one generator). Every line lies inside D → C9 (D carries full
  //    lines of ≥ need − F − 2 other families; D plus two generators).
  //    C5 (an occupied cross-family triangle (z, l, z⊕l) with z ∈ D, base
  //    D + ⟨l⟩ and one generator) is kept as a fast path but is not
  //    needed for coverage.
  //  |T3| = 0: |T2| ≥ need − F ≥ 4 lines; a disjoint pair → C6; otherwise
  //    the distinct lines pairwise intersect, and then they either all
  //    pass through one common point or all lie in one 3-space (two
  //    distinct intersecting 2-spaces span exactly a 3-space; a further
  //    line meeting both is either through their common point or inside
  //    the span, and a line through the point outside the span cannot
  //    meet a line inside the span that avoids the point). Concurrent
  //    lines with three independent directions from distinct families
  //    → C8 (the common point has multiplicity ≥ |T2| ≥ need − F, one
  //    generator); at least two distinct lines inside a 3-space W (which
  //    then carries the lines of ≥ |T2| ≥ need − F families) → C7
  //    (W plus two generators); one repeated line → C10 (the line plus
  //    three generators).
  void RunK5() {
    if (need_ < num_families_ + 4) {
      MarkOverBudget(); // coverage not establishable (see RunK)
      return;
    }
    PrepareK5();
    RunSteps({{"C3", &FamilySearch::CaseC3},
              {"C3'", &FamilySearch::CaseC3Prime},
              {"C4", &FamilySearch::CaseC4},
              {"C5", &FamilySearch::CaseC5},
              {"C6", &FamilySearch::CaseC6},
              {"C7", &FamilySearch::CaseC7},
              {"C8", &FamilySearch::CaseC8},
              {"C10", &FamilySearch::CaseC10},
              {"C9", &FamilySearch::CaseC9}});
  }

  // The k = 5 tables: the multiplicity >= 2 keys, the per-key family bitmask
  // with its byte-lane spread LUT, and the per-family line lists.
  void PrepareK5() {
    multi2_ = CollectMulti2();
    // Per-key family bitmask with a byte-lane spread LUT (built when the
    // family count fits 8 bits): gives per-family point counts of a small
    // point set in a few adds, hence the sound cap
    // d_f(S) ≤ ⌊log2(#(I_f ∩ S ∖ 0) + 1)⌋ used by the FamCap4 prefilter.
    fmask_.clear();
    spread_.fill(0);
    if (num_families_ <= 8) {
      fmask_.assign(qsize_, 0);
      for (int f = 1; f <= num_families_; ++f) {
        for (uint64_t q : families_[f].elems) {
          if (q) {
            fmask_[q] |= static_cast<uint8_t>(1) << (f - 1);
          }
        }
      }
      for (int v = 0; v < 256; ++v) {
        for (int b = 0; b < 8; ++b) {
          if ((v >> b) & 1) {
            spread_[v] |= uint64_t{1} << (8 * b);
          }
        }
      }
    }
    flines_ = CollectFamilyLines(); // for C6/C7/C8
  }

  // Prefilter for a 4-dim base X: Σ_f min(d_f(X)+1, 3, dim I_f) ≥ need is
  // necessary for any all-d≤3 witness X + ⟨g⟩ (growth lemma). Uses the
  // point-count cap for d_f(X); a large sentinel when fmask_ is absent.
  int FamCap4(SearchCtx &ctx, const uint64_t *qb4) const {
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
  bool Complete4Cap3(SearchCtx &ctx, const uint64_t *qb4) const {
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
  bool Complete3TwoGens(SearchCtx &ctx, const uint64_t *qb3) const {
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
  bool Complete2ThreeGens(SearchCtx &ctx, const uint64_t *qb2) const {
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
  template <class Cb>
  bool For3SubsThrough(int f, uint64_t p, const Cb &cb) const {
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

  // The occupied keys of multiplicity >= 2 (the only candidates for a point
  // shared by two families).
  std::vector<uint64_t> CollectMulti2() const {
    std::vector<uint64_t> multi2;
    for (uint64_t q : occupied_) {
      if (cmult_[q] >= 2) {
        multi2.push_back(q);
      }
    }
    return multi2;
  }

  // Per family, its lines {x, a, x^a} as the pairs (x, a) with x < a < x^a.
  std::vector<std::vector<Line>> CollectFamilyLines() const {
    std::vector<std::vector<Line>> lines(
        static_cast<std::size_t>(num_families_) + 1);
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
  template <class Visit> void ForEachMulti2Line(const Visit &visit) const {
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
          if (families_[f].image.Contains(a) &&
              families_[f].image.Contains(b)) {
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
  int FamiliesOf(uint64_t q, int *out) const {
    int cnt = 0;
    for (int f = 1; f <= num_families_; ++f) {
      if (families_[f].image.Contains(q)) {
        out[cnt++] = f;
      }
    }
    return cnt;
  }

  // (C3) two 3-subspaces through a shared multi-point, distinct families.
  void CaseC3() const {
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
  void CaseC3Prime() const {
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
  void CaseC4() const {
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
  void CaseC5() const {
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
          const bool stop =
              For3SubsThrough(fz, z, [&](uint64_t a1, uint64_t a2) {
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

  // (C6) disjoint line pair from two distinct families, plus a generator.
  void CaseC6() const {
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
  void CaseC7() const {
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
                      if (static_cast<int>((acc >> (8 * (f - 1))) & 0xFF) >=
                          3) {
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
  void CaseC8() const {
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
  void CaseC10() const {
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

  // (C9) |T3| = 1 with every other family's line inside the 3-dim
  // intersection D: Ū = D + two occupied generators. Roots: 3-subspaces D
  // of an image that carry full lines (≥ 3 points) of at least
  // need − F − 2 other families (the counting bound for this shape).
  void CaseC9() const {
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

  // ---- k = 4 remainder, all d_f <= 2 (cases gamma, alpha, beta, delta) ----

  void RunK4() {
    PrepareK4();
    RunSteps({{"gamma", &FamilySearch::CaseGamma},
              {"alpha", &FamilySearch::CaseAlpha},
              {"beta", &FamilySearch::CaseBeta},
              {"delta", &FamilySearch::CaseDelta}});
  }

  // The keys of multiplicity >= 2 (the only candidates for a line shared by
  // two families).
  void PrepareK4() {
    multi2_ = CollectMulti2();
    flines_ = CollectFamilyLines(); // for delta
  }

  // (γ) roots: lines shared by ≥ 3 families (a, b ∈ I_f implies
  // t = a ⊕ b ∈ I_f, so the pair decides membership). Collected serially
  // (cheap), completed in parallel with two generators.
  void CaseGamma() const {
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
  void CaseAlpha() const {
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
              const GrowthPlan plan =
                  PrepareGrowth(ctx, base, 3, 2,
                                static_cast<uint64_t>(num_families_) * odim_);
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
  void CaseBeta() const {
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
              const GrowthPlan plan =
                  PrepareGrowth(ctx, base, 3, 2,
                                static_cast<uint64_t>(num_families_) * odim_);
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
  void CaseDelta() const {
    {
      struct DeltaRoot {
        uint64_t a, b;
        int f1, f2;
      };
      std::vector<DeltaRoot> delta_roots;
      ForEachMulti2Line(
          [&](uint64_t a, uint64_t b, int common, int f1, int f2) {
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

  // ---- Setup (constructor only) ----

  // Quotient coordinates, as in the v1 engine.
  void ComputeFreePos() {
    free_pos_ = QuotientFreePositions(span_.v, rb_ * rc_);
    CHECK_EQ(static_cast<int>(free_pos_.size()), dq_);
  }

  // The rank-one matrix of family vector f and other-side vector o.
  Row MakeRow(uint64_t f, uint64_t o) const {
    const uint64_t u = family_on_b_ ? f : o;
    const uint64_t w = family_on_b_ ? o : f;
    return RankOneRow<NB * NC>(u, w, rb_, rc_);
  }

  uint64_t KeyOf(const Row &r) const {
    return KeyFromResidue(span_.v.Reduce(r), free_pos_);
  }

  void BuildFamilies() {
    families_.resize(static_cast<std::size_t>(num_families_) + 1);
    occ_words_.assign(qsize_ / 64 + 1, 0);
    cmult_.assign(qsize_, 0);
    for (int f = 1; f <= num_families_; ++f) {
      Family &fam = families_[f];
      for (int j = 0; j < odim_; ++j) {
        fam.image.Insert(KeyOf(MakeRow(f, uint64_t{1} << j)));
      }
      fam.elems.push_back(0);
      for (int i = 0; i < fam.image.Dim(); ++i) {
        const std::size_t sz = fam.elems.size();
        for (std::size_t x = 0; x < sz; ++x) {
          fam.elems.push_back(fam.elems[x] ^ fam.image.rows[i]);
        }
      }
      std::sort(fam.elems.begin(), fam.elems.end());
      setup_ops_ += static_cast<uint64_t>(odim_) * odim_ + fam.elems.size();
      for (uint64_t q : fam.elems) {
        if (q != 0) {
          occ_words_[q >> 6] |= uint64_t{1} << (q & 63);
          if (cmult_[q] < 255) {
            ++cmult_[q];
          }
        }
      }
    }
    for (uint64_t q = 0; q < qsize_; q += 64) {
      if (occ_words_[q >> 6] == 0) {
        continue;
      }
      for (uint64_t b = 0; b < 64 && q + b < qsize_; ++b) {
        if (Occ(q + b)) {
          occupied_.push_back(q + b);
        }
      }
    }
    setup_ops_ += qsize_ / 64 + occupied_.size();
  }

  // b0 and the key -> rank-one buckets, from the full sweep (family-major).
  void BuildBuckets() {
    for (int f = 1; f <= num_families_; ++f) {
      for (uint64_t o = 1; o < (uint64_t{1} << odim_); ++o) {
        const Row r = MakeRow(f, o);
        const Row resid = span_.v.Reduce(r);
        if (resid.IsZero()) {
          b0_.Insert(r);
        } else {
          buckets_[KeyOf(r)].push_back(r);
        }
      }
      setup_ops_ += uint64_t{1} << odim_;
    }
    d0_ = b0_.Dim();
  }

  // ---- Op accounting and the candidate pipeline ----

  bool Occ(uint64_t q) const { return (occ_words_[q >> 6] >> (q & 63)) & 1; }

  SearchCtx MakeCtx() const {
    SearchCtx ctx;
    ctx.jbases.resize(static_cast<std::size_t>(num_families_) + 1);
    ctx.grow_list.resize(static_cast<std::size_t>(num_families_) + 1);
    return ctx;
  }

  void Flush(SearchCtx &ctx) const {
    if (ctx.local_ops == 0) {
      return;
    }
    const uint64_t total =
        ops_total_.fetch_add(ctx.local_ops, std::memory_order_relaxed) +
        ctx.local_ops;
    ctx.local_ops = 0;
    if (total > op_budget_) {
      MarkOverBudget();
    }
  }

  // Sets the over-budget verdict unless a witness verdict is already set.
  void MarkOverBudget() const {
    int expected = 0;
    verdict_.compare_exchange_strong(expected, -1, std::memory_order_relaxed);
  }

  void Charge(SearchCtx &ctx, uint64_t n) const {
    ctx.local_ops += n;
    if (ctx.local_ops >= kFlushInterval) {
      Flush(ctx);
    }
  }

  bool Stopped() const { return verdict_.load(std::memory_order_relaxed) != 0; }

  // Candidate pipeline: point-count filter -> exact per-family dims -> exact
  // span check. `qb` must hold k independent keys (dim U-bar = k). Returns
  // true iff the candidate is a witness.
  bool CheckCandidate(SearchCtx &ctx, const uint64_t *qb, int k) const {
    const int npts = 1 << k;
    ExpandPoints(qb, k, ctx.pts.data());
    const int need = rho_ + k - d0_;
    int64_t cnt = 0;
    for (int i = 1; i < npts; ++i) {
      cnt += cmult_[ctx.pts[i]];
    }
    Charge(ctx, npts);
    if (cnt < need) {
      return false;
    }
    int dim_sum = 0;
    for (int f = 1; f <= num_families_; ++f) {
      KeyBasis kb = families_[f].image;
      int grown = 0;
      for (int i = 0; i < k; ++i) {
        grown += kb.Insert(qb[i]) ? 1 : 0;
      }
      dim_sum += k - grown; // = dim(I_f ∩ Ū) since the qb are independent
    }
    Charge(ctx, static_cast<uint64_t>(num_families_) * (odim_ + k));
    if (dim_sum < need) {
      return false;
    }
    ctx.scratch.rows.assign(b0_.rows.begin(), b0_.rows.end());
    const int want = rho_ + k;
    uint64_t work = 1;
    for (int i = 1; i < npts; ++i) {
      if (!Occ(ctx.pts[i])) {
        continue;
      }
      const auto it = buckets_.find(ctx.pts[i]);
      if (it == buckets_.end()) {
        continue;
      }
      for (const Row &r : it->second) {
        ++work;
        if (ctx.scratch.Insert(r) && ctx.scratch.Dim() == want) {
          Charge(ctx, work);
          return true;
        }
      }
    }
    Charge(ctx, work);
    return false;
  }

  // Returns true when the caller should stop enumerating (witness found by
  // this candidate, or the shared verdict is already set).
  bool Submit(SearchCtx &ctx, const uint64_t *qb, int k) const {
    if (CheckCandidate(ctx, qb, k)) {
      verdict_.store(1, std::memory_order_relaxed);
      return true;
    }
    return Stopped();
  }

  // Growth filter, from the growth lemma (3) of the class comment: for a
  // base X of `nbase` independent keys in a witness shape that caps every d_f
  // at `cap`, d_f(U-bar) <= min(d_f(X) + [g grows f], cap, dim I_f).
  // PrepareGrowth joins every image with X in ctx.jbases, sums the capped
  // base contributions, lists in ctx.grow_list the families a generator could
  // still grow, and charges the caller-supplied `charge_amount`. The golden
  // tests pin this operation schedule because it affects budget stops. The plan
  // records how many families a generator must grow (gneed) and how many can
  // grow at all (ngrow); a base with gneed > ngrow cannot be completed.
  struct GrowthPlan {
    int gneed;
    int ngrow;
  };

  GrowthPlan PrepareGrowth(SearchCtx &ctx, const uint64_t *base, int nbase,
                           int cap, uint64_t charge_amount) const {
    int contrib0 = 0;
    int ngrow = 0;
    for (int f = 1; f <= num_families_; ++f) {
      KeyBasis &jb = ctx.jbases[f];
      jb = families_[f].image;
      int extended = 0;
      for (int i = 0; i < nbase; ++i) {
        extended += jb.Insert(base[i]) ? 1 : 0;
      }
      const int d = nbase - extended;
      contrib0 += std::min(d, cap);
      if (d < cap && d < families_[f].image.Dim()) {
        ctx.grow_list[ngrow++] = f;
      }
    }
    Charge(ctx, charge_amount);
    return {need_ - contrib0, ngrow};
  }

  // Whether generator g grows at least plan.gneed of the growable families
  // (early-exit scan over ctx.grow_list; charges the families examined).
  bool GeneratorGrows(SearchCtx &ctx, uint64_t g,
                      const GrowthPlan &plan) const {
    if (plan.gneed < 1) {
      return true;
    }
    int grown = 0, remaining = plan.ngrow;
    bool possible = true;
    for (int gi = 0; gi < plan.ngrow; ++gi) {
      if (grown + remaining < plan.gneed) {
        possible = false;
        break;
      }
      --remaining;
      if (ctx.jbases[ctx.grow_list[gi]].Contains(g)) {
        if (++grown >= plan.gneed) {
          break;
        }
      }
    }
    Charge(ctx, static_cast<uint64_t>(plan.ngrow - remaining));
    return possible && grown >= plan.gneed;
  }

  // pts[0..2^k) = every XOR combination of the k keys qb (pts[0] = 0).
  static void ExpandPoints(const uint64_t *qb, int k, uint64_t *pts) {
    pts[0] = 0;
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < (1 << i); ++j) {
        pts[(1 << i) + j] = pts[j] ^ qb[i];
      }
    }
  }

  // Map ForEachSubspaceRREF rows (coordinates over a family's image basis,
  // bit j <-> image.rows[j]) to actual keys; RREF rows are independent, so the
  // keys are too.
  static void RowsToKeys(const Family &fam, const std::vector<uint64_t> &rows,
                         int cnt, uint64_t *qb) {
    for (int i = 0; i < cnt; ++i) {
      uint64_t key = 0;
      for (int j = 0; j < fam.image.Dim(); ++j) {
        if ((rows[i] >> j) & 1) {
          key ^= fam.image.rows[j];
        }
      }
      qb[i] = key;
    }
  }

  // Case driver: runs `worker(ctx, i)` for i in [0, n) -- serially with a
  // single context (deterministic; the verifier's mode), or chunked under
  // tbb::parallel_for with one context per chunk.
  template <class Worker>
  void RunCase(std::size_t n, const Worker &worker) const {
    if (parallel_ && n > 1) {
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n),
                        [&](const tbb::blocked_range<std::size_t> &range) {
                          SearchCtx ctx = MakeCtx();
                          for (std::size_t i = range.begin();
                               i != range.end() && !Stopped(); ++i) {
                            worker(ctx, i);
                          }
                          Flush(ctx);
                        });
    } else {
      SearchCtx ctx = MakeCtx();
      for (std::size_t i = 0; i < n && !Stopped(); ++i) {
        worker(ctx, i);
      }
      Flush(ctx);
    }
  }

  // Task lists for the image-subspace cases: one task per (family, first
  // pivot) partition of ForEachSubspaceRREF.
  std::vector<std::pair<int, int>> PivotTasks(int kk) const {
    std::vector<std::pair<int, int>> tasks;
    for (int f = 1; f <= num_families_; ++f) {
      const int m = families_[f].image.Dim();
      for (int p0 = m - 1; p0 >= kk - 1; --p0) {
        tasks.push_back({f, p0});
      }
    }
    return tasks;
  }

  // ---- Inputs and derived scalars (declaration order = init order) ----
  const Span &span_;
  const int target_rank_;
  const uint64_t op_budget_;
  const bool parallel_;
  const int rb_, rc_, rho_; // core dims and dim V
  const int dq_;            // quotient dim rb*rc - rho
  const int e_;             // max quotient dim to search, <= kMaxFamilyK
  const bool family_on_b_;  // family side = the smaller core factor
  const int fdim_, odim_;   // family-side and other-side dims
  const int num_families_;  // 2^fdim - 1
  const std::size_t qsize_; // 2^dq

  // ---- Read-only tables (written only by the constructor) ----
  std::vector<int> free_pos_;
  std::vector<Family> families_;
  std::vector<uint64_t> occ_words_;
  std::vector<uint8_t> cmult_; // #{f : q in I_f}, saturating at 255
  std::vector<uint64_t> occupied_;
  LinearBasis<NB * NC> b0_;
  boost::unordered_flat_map<uint64_t, std::vector<Row>> buckets_;
  int d0_ = 0;
  uint64_t setup_ops_ = 0;

  // ---- Per-k state (written by RunK before any worker starts) ----
  int k_ = 0;
  int need_ = 0; // rho + k - d0: the dimension the rank-ones must add
  uint64_t ops_k_start_ = 0;
  std::vector<uint64_t> multi2_;          // occupied keys with cmult >= 2
  std::vector<uint8_t> fmask_;            // per-key family bitmask, F <= 8 only
  std::array<uint64_t, 256> spread_{};    // byte-lane spread LUT for fmask_
  std::vector<std::vector<Line>> flines_; // per-family lines

  // ---- Shared search state ----
  // 0 = keep going, 1 = witness, -1 = over budget.
  mutable std::atomic<int> verdict_{0};
  mutable std::atomic<uint64_t> ops_total_{0};
};

// Entry point of the v2 engine (see the FamilySearch comment): applies the
// applicability guards, then runs FamilySearch. *ops_out receives the op
// count (0 when the guards reject).
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
RankOneSpanResult RankOneSpanFamilySearch(const SliceSpanA<P, NA, NB, NC> &span,
                                          int target_rank, uint64_t op_budget,
                                          uint64_t *ops_out,
                                          bool parallel = false) {
  const int rb = span.rb, rc = span.rc, rho = span.rho;
  const int dq = rb * rc - rho;
  const int e = std::min(target_rank - rho, dq);
  *ops_out = 0;
  CHECK_GE(target_rank, rho);
  if (e > kMaxFamilyK || dq > kMaxFamilyDq ||
      std::min(rb, rc) > kMaxFamilySideDim) {
    return RankOneSpanResult::kOverBudget;
  }
  FamilySearch<P, NA, NB, NC> search(span, target_rank, op_budget, parallel);
  return search.Run(ops_out);
}

} // namespace rank_one_span_internal
