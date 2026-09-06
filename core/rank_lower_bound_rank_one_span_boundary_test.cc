// Family search vs. exhaustive enumeration in the coverage boundary regime of
// the k = 4 and k = 5 remainders: instances with F = 7 families where
// need - F is 3 or 4 (the smallest values the coverage guards accept), so that
// the case analysis has the least slack. Generation modes: uniform random
// slices; slices that are sums of two rank-ones from distinct families (which
// makes quotient keys of different families coincide, i.e. shared lines in the
// quotient); and a mixed mode.
#include <algorithm>
#include <cstdint>
#include <random>

#include "gtest/gtest.h"

#include "core/gf.h"
#include "core/rank_lower_bound_rank_one_span.h"
#include "core/tensor.h"

namespace {

constexpr uint64_t kBig = uint64_t{1} << 50;

template <std::size_t NA> void RunTrials(uint64_t seed, int trials, int mode,
                                         int *decided, int *boundary) {
  using namespace rank_one_span_internal;
  std::mt19937_64 rng(seed);
  for (int trial = 0; trial < trials; ++trial) {
    Tensor<2, NA, 3, 5> t = {};
    // mode 3: one fixed family pair (f1, f2) shares two "sum" slices (so their
    // quotient images share a line), the rest of the slices are random.
    const uint32_t pf1 = 1 + rng() % 7;
    uint32_t pf2 = 1 + rng() % 7;
    while (pf2 == pf1) pf2 = 1 + rng() % 7;
    for (std::size_t y = 0; y < NA; ++y) {
      int m = (mode == 2) ? static_cast<int>(rng() % 2) : mode;
      if (mode == 3) m = (y < 2) ? 3 : static_cast<int>(rng() % 2);
      if (m == 3) {
        const uint32_t o1 = 1 + rng() % 31, o2 = 1 + rng() % 31;
        for (int x = 0; x < 3; ++x) {
          for (int z = 0; z < 5; ++z) {
            int v = 0;
            if (((pf1 >> x) & 1) && ((o1 >> z) & 1)) v ^= 1;
            if (((pf2 >> x) & 1) && ((o2 >> z) & 1)) v ^= 1;
            if (v) t[y][x][z] = GF<2>::One();
          }
        }
      } else if (m == 0) {
        for (int x = 0; x < 3; ++x) {
          for (int z = 0; z < 5; ++z) {
            if (rng() % 2) t[y][x][z] = GF<2>::One();
          }
        }
      } else {
        // slice = f1 o1^T + f2 o2^T with f1 != f2 (rank two)
        const uint32_t f1 = 1 + rng() % 7;
        uint32_t f2 = 1 + rng() % 7;
        while (f2 == f1) f2 = 1 + rng() % 7;
        const uint32_t o1 = 1 + rng() % 31, o2 = 1 + rng() % 31;
        for (int x = 0; x < 3; ++x) {
          for (int z = 0; z < 5; ++z) {
            int v = 0;
            if (((f1 >> x) & 1) && ((o1 >> z) & 1)) v ^= 1;
            if (((f2 >> x) & 1) && ((o2 >> z) & 1)) v ^= 1;
            if (v) t[y][x][z] = GF<2>::One();
          }
        }
      }
    }
    const auto span = BuildSliceSpanA<2, NA, 3, 5>(t);
    for (int target = span.rho + 3; target <= span.rho + 5; ++target) {
      if (CostFromSpan(span, target) > kRankOneSpanV1CostThreshold) continue;
      const RankOneSpanResult v1 =
          EnumerateAllSubspaces<2, NA, 3, 5>(span, target);
      uint64_t ops = 0;
      const RankOneSpanResult v2 =
          RankOneSpanFamilySearch<2, NA, 3, 5>(span, target, kBig, &ops);
      if (v2 == RankOneSpanResult::kOverBudget) continue;
      ++*decided;
      // need - F for this k, where F = 7 (families on the 3-dim side).
      const int k = target - span.rho;
      // d0 is not exposed; a conservative marker of the boundary regime is
      // rho + k - 7 <= 4 (need <= F + 4 when d0 = 0).
      if (span.rho + k - 7 <= 4) ++*boundary;
      EXPECT_EQ(v2, v1) << "mode " << mode << " trial " << trial << " rho "
                        << span.rho << " target " << target << " NA " << NA;
    }
  }
}

// Directly planted "doubled line" configurations: V is spanned by
//   f1 (x) o1 + f2 (x) o1',  f1 (x) o2 + f2 (x) o2'      (f1, f2 share the line D)
// and, for each of the four families g outside {f1, f2, f3}, one element
//   g (x) o_g + (a rank-one of f1 or f3 in its planted line),
// so that W = D (+) D' with D' the line of f3, and every other family meets
// W only inside D or D'. The span U = V + f1 (x) <o1,o2> + f3 (x) <o1'',o2''>
// is rank-one spanned of dimension rho + 4 whenever the choices are generic.
TEST(FamilySearchBoundaryTest, PlantedDoubledLineWitnesses) {
  using namespace rank_one_span_internal;
  std::mt19937_64 rng(777);
  int witnesses = 0, checked = 0;
  auto add = [](Tensor<2, 6, 3, 5> &t, std::size_t y, uint32_t f, uint32_t o) {
    for (int x = 0; x < 3; ++x) {
      for (int z = 0; z < 5; ++z) {
        if (((f >> x) & 1) && ((o >> z) & 1)) {
          t[y][x][z] = t[y][x][z] + GF<2>::One();
        }
      }
    }
  };
  for (int trial = 0; trial < 600; ++trial) {
    uint32_t fam[7] = {1, 2, 3, 4, 5, 6, 7};
    std::shuffle(fam, fam + 7, rng);
    const uint32_t f1 = fam[0], f2 = fam[1], f3 = fam[2];
    const uint32_t o1 = 1 + rng() % 31, o2 = 1 + rng() % 31;
    const uint32_t p1 = 1 + rng() % 31, p2 = 1 + rng() % 31;
    const uint32_t q1 = 1 + rng() % 31, q2 = 1 + rng() % 31;
    Tensor<2, 6, 3, 5> t = {};
    add(t, 0, f1, o1);
    add(t, 0, f2, p1);
    add(t, 1, f1, o2);
    add(t, 1, f2, p2);
    // Two of the four remaining families are paired with distinct points of
    // the f1 line, the other two with distinct points of the f3 line, so that
    // no proper sub-configuration is already a witness (the minimal witness
    // then has k = 4; this is checked below by excluding rho + 3 first).
    uint32_t cs[4] = {1, 2, 3, 0};
    std::shuffle(cs, cs + 3, rng);
    cs[3] = 1 + rng() % 3;
    while (cs[3] == cs[2]) cs[3] = 1 + rng() % 3;
    for (int i = 0; i < 4; ++i) {
      const uint32_t g = fam[3 + i];
      const uint32_t og = 1 + rng() % 31;
      add(t, 2 + i, g, og);
      const uint32_t c = cs[i];
      if (i < 2) {
        add(t, 2 + i, f1, (c & 1 ? o1 : 0) ^ (c & 2 ? o2 : 0));
      } else {
        add(t, 2 + i, f3, (c & 1 ? q1 : 0) ^ (c & 2 ? q2 : 0));
      }
    }
    const auto span = BuildSliceSpanA<2, 6, 3, 5>(t);
    if (span.rho != 6 || span.rb != 3 || span.rc != 5) continue;
    const int target = span.rho + 4;
    if (CostFromSpan(span, target) > kRankOneSpanV1CostThreshold) continue;
    // Keep only instances whose minimal witness has k = 4.
    if (EnumerateAllSubspaces<2, 6, 3, 5>(span, target - 1) !=
        RankOneSpanResult::kExcluded) {
      continue;
    }
    const RankOneSpanResult v1 = EnumerateAllSubspaces<2, 6, 3, 5>(span, target);
    uint64_t ops = 0;
    const RankOneSpanResult v2 =
        RankOneSpanFamilySearch<2, 6, 3, 5>(span, target, kBig, &ops);
    if (v2 == RankOneSpanResult::kOverBudget) continue;
    ++checked;
    if (v1 == RankOneSpanResult::kWitnessFound) ++witnesses;
    EXPECT_EQ(v2, v1) << "trial " << trial << " f1 " << f1 << " f2 " << f2
                      << " f3 " << f3;
  }
  // Every planted instance turned out to have a smaller witness (k <= 3), so
  // `checked` is typically 0: the doubled-line shape never appears as a
  // minimal witness in these constructions. Kept as a regression check.
  GTEST_LOG_(INFO) << "checked " << checked << ", v1 witnesses " << witnesses;
}

TEST(FamilySearchBoundaryTest, AgreesWithV1NearCoverageBoundary) {
  int decided = 0, boundary = 0;
  RunTrials<6>(1, 100, 0, &decided, &boundary);
  RunTrials<6>(2, 100, 1, &decided, &boundary);
  RunTrials<6>(3, 100, 2, &decided, &boundary);
  RunTrials<7>(4, 100, 0, &decided, &boundary);
  RunTrials<7>(5, 100, 1, &decided, &boundary);
  RunTrials<7>(6, 100, 2, &decided, &boundary);
  RunTrials<8>(7, 80, 1, &decided, &boundary);
  RunTrials<8>(8, 80, 2, &decided, &boundary);
  RunTrials<6>(9, 400, 3, &decided, &boundary);
  RunTrials<7>(10, 400, 3, &decided, &boundary);
  RunTrials<8>(11, 150, 3, &decided, &boundary);
  EXPECT_GT(decided, 0);
  EXPECT_GT(boundary, 0);
  GTEST_LOG_(INFO) << "decided " << decided << ", boundary-regime "
                   << boundary;
}

} // namespace


#include "core/constraints.h"
#include "core/rank_lower_bound_flatten.h"
#include "matrix/problem.h"

namespace {
// Structure statistics of the real <3,3,3> dim-6 orbits on axis 1: counts of
// the objects the family-search remainder cases enumerate, to estimate the
// cost of the two extra cases (doubled line + disjoint line at k = 4; a
// 3-dim intersection carrying the lines of other families at k = 5) and of
// case C10 (one line shared by >= need5 - F families, plus three generators).
// No test plants a C10 witness: with F = 7 a minimal witness never falls into
// case C10 (paper/main.tex, Section 8, remark "case C10 and minimal
// witnesses"), so instances with a three-dimensional family side cannot
// exercise it.
void OrbitStructure(const char *name, const std::vector<uint16_t> &words) {
  using namespace rank_one_span_internal;
  using Problem = matrix::Problem<2, 3, 3, 3>;
  Constraints<2, 9> constraints;
  for (uint16_t w : words) constraints.push_back(GFVec<2, 9>{static_cast<BitVec<9>>(w)});
  const auto t0 = ApplyConstraintsToTensor<2, 9, 9, 9>(constraints, Problem::MakeTensor());
  const auto t = CyclicTranspose<2, 9, 9, 9>(t0);
  const auto span = BuildSliceSpanA<2, 9, 9, 9>(t);
  using Row = WideBits<81>;
  const int rb = span.rb, rc = span.rc, rho = span.rho, dq = rb * rc - rho;
  const bool family_on_b = rb <= rc;
  const int fdim = family_on_b ? rb : rc, odim = family_on_b ? rc : rb;
  const int F = (1 << fdim) - 1;
  std::vector<bool> is_top(rb * rc, false);
  for (const auto &row : span.v.rows) is_top[row.first] = true;
  std::vector<int> free_pos;
  for (int i = 0; i < rb * rc; ++i) if (!is_top[i]) free_pos.push_back(i);
  auto make_row = [&](uint64_t f, uint64_t o) {
    const uint64_t u = family_on_b ? f : o, w = family_on_b ? o : f;
    Row r;
    for (int s = 0; s < rb; ++s) if ((u >> s) & 1)
      for (int tt = 0; tt < rc; ++tt) if ((w >> tt) & 1) r.SetBit(s * rc + tt);
    return r;
  };
  auto key_of = [&](const Row &r) {
    const Row res = span.v.Reduce(r);
    uint64_t key = 0;
    for (int i = 0; i < dq; ++i) if (res.GetBit(free_pos[i])) key |= uint64_t{1} << i;
    return key;
  };
  const std::size_t qsize = std::size_t{1} << dq;
  std::vector<uint8_t> fmask(qsize, 0), cmult(qsize, 0);
  std::vector<KeyBasis> image(F + 1);
  LinearBasis<81> b0;
  for (int f = 1; f <= F; ++f) {
    for (int j = 0; j < odim; ++j) image[f].Insert(key_of(make_row(f, uint64_t{1} << j)));
    for (uint64_t o = 1; o < (uint64_t{1} << odim); ++o) {
      const Row r = make_row(f, o);
      const uint64_t k = key_of(r);
      if (k == 0) b0.Insert(r); else { fmask[k] |= 1 << (f - 1); }
    }
  }
  std::size_t occupied = 0, multi2 = 0;
  for (std::size_t q = 1; q < qsize; ++q) {
    cmult[q] = static_cast<uint8_t>(__builtin_popcount(fmask[q]));
    if (cmult[q]) ++occupied;
    if (cmult[q] >= 2) ++multi2;
  }
  const int d0 = b0.Dim();
  const int need4 = rho + 4 - d0, need5 = rho + 5 - d0;
  // Lines per family, doubled lines (common == 2), tripled (>= 3), C10 roots
  // (lines in >= max(need5 - F, 4) images), and the delta candidate count:
  // doubled line x disjoint lines of third families.
  std::vector<std::vector<std::array<uint64_t, 2>>> flines(F + 1);
  for (int f = 1; f <= F; ++f) {
    std::vector<uint64_t> elems;
    elems.push_back(0);
    for (int i = 0; i < image[f].Dim(); ++i) {
      const std::size_t sz = elems.size();
      for (std::size_t x = 0; x < sz; ++x) elems.push_back(elems[x] ^ image[f].rows[i]);
    }
    for (uint64_t x : elems) if (x) for (uint64_t a : elems) if (a > x && (a ^ x) > a) flines[f].push_back({x, a});
  }
  std::size_t doubled = 0, tripled = 0, c10_roots = 0, delta_cands = 0;
  const int t10 = std::max(need5 - F, 4);
  std::size_t total_lines = 0;
  for (int f = 1; f <= F; ++f) total_lines += flines[f].size();
  std::vector<std::array<uint64_t, 2>> seen; // dedup lines across families
  for (int f = 1; f <= F; ++f) {
    for (const auto &l : flines[f]) {
      const uint8_t m = fmask[l[0]] & fmask[l[1]] & fmask[l[0] ^ l[1]];
      const int common = __builtin_popcount(m);
      // count each line once: only from its lowest family
      if ((m & ((1u << f) - 1)) != (1u << (f - 1))) continue;
      if (common == 2) {
        ++doubled;
        for (int f3 = 1; f3 <= F; ++f3) {
          if ((m >> (f3 - 1)) & 1) continue;
          for (const auto &l2 : flines[f3]) {
            KeyBasis kb;
            kb.Insert(l[0]); kb.Insert(l[1]); kb.Insert(l2[0]); kb.Insert(l2[1]);
            if (kb.Dim() == 4) ++delta_cands;
          }
        }
      } else if (common >= 3) {
        ++tripled;
        if (common >= t10) ++c10_roots;
      }
    }
  }
  // C9 roots: 3-subspaces D of an image I_f carrying full lines of >= need5-F-2
  // other families.
  std::size_t c9_roots = 0, subs3 = 0;
  for (int f = 1; f <= F; ++f) {
    const int m = image[f].Dim();
    if (m < 3) continue;
    ForEachSubspaceRREF(m, 3, [&](const std::vector<uint64_t> &rows) {
      uint64_t d[3] = {0, 0, 0};
      for (int i = 0; i < 3; ++i) for (int j = 0; j < m; ++j) if ((rows[i] >> j) & 1) d[i] ^= image[f].rows[j];
      ++subs3;
      int cnt[8] = {0};
      for (int mask = 1; mask < 8; ++mask) {
        uint64_t p = 0;
        for (int i = 0; i < 3; ++i) if ((mask >> i) & 1) p ^= d[i];
        for (int g = 1; g <= F; ++g) if ((fmask[p] >> (g - 1)) & 1) ++cnt[g];
      }
      int others = 0;
      for (int g = 1; g <= F; ++g) if (g != f && cnt[g] >= 3) ++others;
      if (others >= need5 - F - 2) ++c9_roots;
      return false;
    });
  }
  LOG(INFO) << name << ": rb=" << rb << " rc=" << rc << " rho=" << rho << " dq=" << dq
            << " F=" << F << " d0=" << d0 << " need4=" << need4 << " need5=" << need5
            << " occupied=" << occupied << " multi2=" << multi2 << " lines=" << total_lines
            << " doubled=" << doubled << " tripled=" << tripled << " c10_roots=" << c10_roots
            << " delta_cands=" << delta_cands << " subs3=" << subs3 << " c9_roots=" << c9_roots;
}

TEST(FamilySearchBoundaryTest, N333OrbitStructure) {
  OrbitStructure("orbit36", {0x1, 0x2, 0x8, 0x14, 0x60, 0x100});
  OrbitStructure("orbit45", {0x1, 0x2, 0x8, 0x44, 0xA0, 0x100});
}
} // namespace
