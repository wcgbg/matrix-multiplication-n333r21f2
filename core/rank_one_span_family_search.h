#pragma once

// The v2 engine of the rank-one-span exclusion over 𝔽₂: the family-structure
// search (class FamilySearch below), for quotients beyond the reach of the v1
// enumeration in core/rank_one_span_f2.h. See
// core/rank_lower_bound_rank_one_span.h for the method and the public API.
// Trusted: the verifier re-runs this search to check a RankOneSpanProof.
//
// This header owns the driver, coverage guards, setup, and budget accounting.
// Definitions in core/rank_one_span_family_search/ follow the paper's cases:
//   basic_cases.h: k <= 2 and dmax = k / k-1;
//   k4_cases.h: gamma, alpha, beta, delta;
//   k5_three_spaces.h: C3, C3', C4, C5, C9;
//   k5_lines.h: C6, C7, C8, C10.
// geometry.h and k5_completion.h hold their shared enumeration helpers.
// RunSteps below fixes traversal order independently of this file grouping.

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

  // basic_cases.h: occupied singles/pairs, then dmax = k and k-1.
  void CaseK1() const;
  void CaseK2() const;
  void CaseDmaxK() const;
  void CaseDmaxKm1() const;

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
    // Preserve the search schedule: C5 is a fast path, and C10 precedes C9.
    // File grouping follows the proof; execution order also pins budget stops.
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

  // k5_completion.h: capped completions and three-subspaces through a point.
  int FamCap4(SearchCtx &ctx, const uint64_t *qb4) const;
  bool Complete4Cap3(SearchCtx &ctx, const uint64_t *qb4) const;
  bool Complete3TwoGens(SearchCtx &ctx, const uint64_t *qb3) const;
  bool Complete2ThreeGens(SearchCtx &ctx, const uint64_t *qb2) const;
  template <class Cb>
  bool For3SubsThrough(int f, uint64_t p, const Cb &cb) const;

  // geometry.h: shared root collectors with their original charge schedules.
  std::vector<uint64_t> CollectMulti2() const;
  std::vector<std::vector<Line>> CollectFamilyLines() const;
  template <class Visit>
  void ForEachMulti2Line(const Visit &visit) const;
  int FamiliesOf(uint64_t q, int *out) const;

  // k5_three_spaces.h: C3/C3' (two three-spaces), C4/C5/C9 (one three-space).
  void CaseC3() const;
  void CaseC3Prime() const;
  void CaseC4() const;
  void CaseC5() const;
  void CaseC9() const;

  // k5_lines.h: C6 (also used when |T3| = 1), C7, C8, and C10.
  void CaseC6() const;
  void CaseC7() const;
  void CaseC8() const;
  void CaseC10() const;

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

  // k4_cases.h: the four named dimension-four remainder cases.
  void CaseGamma() const;
  void CaseAlpha() const;
  void CaseBeta() const;
  void CaseDelta() const;

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

// Case definitions follow the complete class so they share its private state.
#include "core/rank_one_span_family_search/basic_cases.h"
#include "core/rank_one_span_family_search/geometry.h"
#include "core/rank_one_span_family_search/k4_cases.h"
#include "core/rank_one_span_family_search/k5_completion.h"
#include "core/rank_one_span_family_search/k5_lines.h"
#include "core/rank_one_span_family_search/k5_three_spaces.h"
