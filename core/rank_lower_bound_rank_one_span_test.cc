#include "core/rank_lower_bound_rank_one_span.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/proto_io.h"
#include "core/tensor.h"
#include "matrix/problem.h"

namespace {

using rank_one_span_internal::SubspaceCount2;

constexpr uint64_t kBig = 100'000'000;

TEST(SubspaceCount2Test, SmallValuesAndSaturation) {
  // Subspaces of F_2^9 of dim <= e: 1, +511, +43435, +788035, +3309747.
  EXPECT_EQ(SubspaceCount2(9, 0), 1u);
  EXPECT_EQ(SubspaceCount2(9, 1), 512u);
  EXPECT_EQ(SubspaceCount2(9, 2), 43'947u);
  EXPECT_EQ(SubspaceCount2(9, 3), 831'982u);
  EXPECT_EQ(SubspaceCount2(9, 4), 4'141'729u);
  // e is capped at dq.
  EXPECT_EQ(SubspaceCount2(2, 5), 1u + 3u + 1u);
  EXPECT_EQ(SubspaceCount2(200, 100), std::numeric_limits<uint64_t>::max());
}

TEST(RankOneSpanTest, DiagonalTensor) {
  // T = e0⊗e0⊗e0 + e1⊗e1⊗e1 has rank 2.
  Tensor<2, 2, 2, 2> t = {};
  t[0][0][0] = GF<2>::One();
  t[1][1][1] = GF<2>::One();
  EXPECT_EQ((RankOneSpanExcludeA<2, 2, 2, 2>(t, 1, kBig)),
            RankOneSpanResult::kExcluded);
  EXPECT_EQ((RankOneSpanExcludeA<2, 2, 2, 2>(t, 2, kBig)),
            RankOneSpanResult::kWitnessFound);
  EXPECT_EQ((RankOneSpanExcludeA<2, 2, 2, 2>(t, 4, kBig)),
            RankOneSpanResult::kWitnessFound);
}

TEST(RankOneSpanTest, ZeroTensor) {
  Tensor<2, 2, 2, 2> t = {};
  // rank <= 0 must not be excluded, rank <= -1 is vacuously excluded.
  EXPECT_EQ((RankOneSpanExcludeA<2, 2, 2, 2>(t, 0, kBig)),
            RankOneSpanResult::kWitnessFound);
  EXPECT_EQ((RankOneSpanExcludeA<2, 2, 2, 2>(t, -1, kBig)),
            RankOneSpanResult::kExcluded);
}

TEST(RankOneSpanTest, MatMul222) {
  // rank_F2(<2,2,2>) = 7 (Strassen is optimal, and 7 is also the F2 rank).
  using Problem = matrix::Problem<2, 2, 2, 2>;
  const auto t = Problem::MakeTensor();
  // Trivial: below the flattening rank 4.
  EXPECT_EQ((RankOneSpanExcludeA<2, 4, 4, 4>(t, 3, kBig)),
            RankOneSpanResult::kExcluded);
  // Real exclusions at 5 (e=1, 4k subspaces) and 6 (e=2, ~2.8M subspaces).
  EXPECT_EQ((RankOneSpanExcludeA<2, 4, 4, 4>(t, 5, kBig)),
            RankOneSpanResult::kExcluded);
  auto [result, axis] = RankOneSpanExclude<2, 4, 4, 4>(t, 6, kBig);
  EXPECT_EQ(result, RankOneSpanResult::kExcluded);
  // Budget guard.
  EXPECT_EQ((RankOneSpanExcludeA<2, 4, 4, 4>(t, 6, 10)),
            RankOneSpanResult::kOverBudget);
  EXPECT_EQ((RankOneSpanExclude<2, 4, 4, 4>(t, 6, 10).first),
            RankOneSpanResult::kOverBudget);
}

// The three dim-6 orbits of the n333 certificate this technique was built for
// (first factor restricted to a 3-dim subspace of the 3x3 matrices; the
// backtracking DFS saturates at 11 on all three). Constraint words are the
// certificate's canonical RREF rows, bit 3i+j = entry (i,j).
Tensor<2, 9, 9, 9> N333ConstrainedTensor(const std::vector<uint16_t> &words) {
  using Problem = matrix::Problem<2, 3, 3, 3>;
  Constraints<2, 9> constraints;
  for (uint16_t w : words) {
    constraints.push_back(GFVec<2, 9>{static_cast<BitVec<9>>(w)});
  }
  return ApplyConstraintsToTensor<2, 9, 9, 9>(constraints,
                                              Problem::MakeTensor());
}

TEST(RankOneSpanTest, N333Orbit24) {
  const auto t = N333ConstrainedTensor({0x1, 0x2, 0x4, 0x8, 0x50, 0xA0});
  auto [excl11, axis11] = RankOneSpanExclude<2, 9, 9, 9>(t, 11, kBig);
  EXPECT_EQ(excl11, RankOneSpanResult::kExcluded); // proves rank >= 12
  auto [wit12, axis12] = RankOneSpanExclude<2, 9, 9, 9>(t, 12, kBig);
  EXPECT_EQ(wit12, RankOneSpanResult::kWitnessFound); // rank <= 12 = ub
}

TEST(RankOneSpanTest, N333Orbit27) {
  const auto t = N333ConstrainedTensor({0x1, 0x2, 0x4, 0x8, 0xA0, 0x130});
  EXPECT_EQ((RankOneSpanExclude<2, 9, 9, 9>(t, 11, kBig).first),
            RankOneSpanResult::kExcluded);
  EXPECT_EQ((RankOneSpanExclude<2, 9, 9, 9>(t, 12, kBig).first),
            RankOneSpanResult::kWitnessFound);
}

TEST(RankOneSpanTest, N333Orbit28) {
  const auto t = N333ConstrainedTensor({0x1, 0x2, 0x4, 0x50, 0xA0, 0x118});
  EXPECT_EQ((RankOneSpanExclude<2, 9, 9, 9>(t, 12, kBig).first),
            RankOneSpanResult::kExcluded); // proves rank >= 13
  EXPECT_EQ((RankOneSpanExclude<2, 9, 9, 9>(t, 13, kBig).first),
            RankOneSpanResult::kWitnessFound); // rank <= 13 = ub
}

// ---- Family-search (v2 engine) tests ----

// Max size of a line-free subset of F_2^k \ 0 (no {a, b, a^b} fully inside),
// by brute force. Documents why the family search uses runtime coverage
// arithmetic instead of cap-set constants: caps are as large as 2^(k-1).
int MaxLineFree(int k) {
  const int n = (1 << k) - 1;
  int best = 0;
  for (uint32_t sub = 0; sub < (uint32_t{1} << n); ++sub) {
    bool ok = true;
    for (int a = 1; a <= n && ok; ++a) {
      if (!((sub >> (a - 1)) & 1)) {
        continue;
      }
      for (int b = a + 1; b <= n && ok; ++b) {
        if (((sub >> (b - 1)) & 1) &&
            (((a ^ b) > b) && ((sub >> ((a ^ b) - 1)) & 1))) {
          ok = false;
        }
      }
    }
    if (ok) {
      best = std::max(best, std::popcount(sub));
    }
  }
  return best;
}

TEST(FamilySearchTest, MaxLineFreeSizes) {
  EXPECT_EQ(MaxLineFree(2), 2);
  EXPECT_EQ(MaxLineFree(3), 4);
  EXPECT_EQ(MaxLineFree(4), 8);
}

// The family search must agree with the exhaustive v1 enumeration wherever it
// decides. <2,2,2> (rank 7): v1 decides all targets; the family search decides
// targets up to rho + 2 here (its k = 3 coverage arithmetic fails with 15
// families) and must match on those.
TEST(FamilySearchTest, AgreesWithV1OnMatMul222) {
  using namespace rank_one_span_internal;
  using Problem = matrix::Problem<2, 2, 2, 2>;
  const auto t = Problem::MakeTensor();
  const auto span = BuildSliceSpanA<2, 4, 4, 4>(t);
  for (int target = span.rho; target <= 7; ++target) {
    const RankOneSpanResult v1 =
        RankOneSpanExcludeA<2, 4, 4, 4>(t, target, kBig);
    uint64_t ops = 0;
    const RankOneSpanResult v2 =
        RankOneSpanFamilySearch<2, 4, 4, 4>(span, target, kBig, &ops);
    if (v2 != RankOneSpanResult::kOverBudget) {
      EXPECT_EQ(v2, v1) << "target " << target;
    }
  }
}

// Cross-check on the orbits the v1 engine certified (24/27/28): both engines
// run on the axis-1 rotation (slice along B; 3x9x6 core, dq = 9) and must
// agree on every decided target.
TEST(FamilySearchTest, AgreesWithV1OnN333V1Orbits) {
  using namespace rank_one_span_internal;
  const std::vector<std::vector<uint16_t>> orbit_words = {
      {0x1, 0x2, 0x4, 0x8, 0x50, 0xA0},  // orbit 24, rank 12
      {0x1, 0x2, 0x4, 0x8, 0xA0, 0x130}, // orbit 27, rank 12
      {0x1, 0x2, 0x4, 0x50, 0xA0, 0x118} // orbit 28, rank 13
  };
  for (const auto &words : orbit_words) {
    const auto t = N333ConstrainedTensor(words);
    const auto t1 = CyclicTranspose<2, 9, 9, 9>(t);
    const auto span = BuildSliceSpanA<2, 9, 9, 9>(t1);
    ASSERT_EQ(span.rho, 9);
    for (int target = 10; target <= 13; ++target) {
      const RankOneSpanResult v1 =
          RankOneSpanExcludeA<2, 9, 9, 9>(t1, target, kBig);
      uint64_t ops = 0;
      const RankOneSpanResult v2 =
          RankOneSpanFamilySearch<2, 9, 9, 9>(span, target, kBig, &ops);
      if (v2 != RankOneSpanResult::kOverBudget) {
        EXPECT_EQ(v2, v1) << "target " << target;
      }
    }
  }
}

// Randomized soundness: sums of r random rank-ones have rank <= r, so the
// family search must never exclude at r.
TEST(FamilySearchTest, NeverExcludesConstructedRank) {
  using namespace rank_one_span_internal;
  std::mt19937_64 rng(12345);
  for (int trial = 0; trial < 40; ++trial) {
    Tensor<2, 4, 4, 4> t = {};
    const int r = 1 + static_cast<int>(rng() % 6);
    for (int i = 0; i < r; ++i) {
      const uint32_t a = 1 + rng() % 15, b = 1 + rng() % 15, c = 1 + rng() % 15;
      for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
          for (int z = 0; z < 4; ++z) {
            if (((a >> x) & 1) && ((b >> y) & 1) && ((c >> z) & 1)) {
              t[x][y][z] = t[x][y][z] + GF<2>::One();
            }
          }
        }
      }
    }
    const auto span = BuildSliceSpanA<2, 4, 4, 4>(t);
    if (r < span.rho) {
      continue; // impossible; the construction canceled below its flattening
    }
    uint64_t ops = 0;
    EXPECT_NE((RankOneSpanFamilySearch<2, 4, 4, 4>(span, r, kBig, &ops)),
              RankOneSpanResult::kExcluded)
        << "trial " << trial << " r " << r;
  }
}

// The parallel mode must reach the same verdicts as the sequential engines
// (op counts may differ; the verdict is order-independent).
TEST(FamilySearchTest, ParallelAgreesWithV1OnN333V1Orbits) {
  using namespace rank_one_span_internal;
  const auto t = N333ConstrainedTensor({0x1, 0x2, 0x4, 0x8, 0x50, 0xA0});
  const auto t1 = CyclicTranspose<2, 9, 9, 9>(t);
  const auto span = BuildSliceSpanA<2, 9, 9, 9>(t1);
  for (int target = 10; target <= 13; ++target) {
    const RankOneSpanResult v1 =
        RankOneSpanExcludeA<2, 9, 9, 9>(t1, target, kBig);
    uint64_t ops = 0;
    const RankOneSpanResult v2 = RankOneSpanFamilySearch<2, 9, 9, 9>(
        span, target, kBig, &ops, /*parallel=*/true);
    if (v2 != RankOneSpanResult::kOverBudget) {
      EXPECT_EQ(v2, v1) << "target " << target;
    }
  }
}

// Deterministic: same verdict and op count on repeated runs.
TEST(FamilySearchTest, Deterministic) {
  using namespace rank_one_span_internal;
  const auto t = N333ConstrainedTensor({0x1, 0x2, 0x20, 0x50, 0x9C, 0x110});
  const auto t1 = CyclicTranspose<2, 9, 9, 9>(t);
  const auto span = BuildSliceSpanA<2, 9, 9, 9>(t1);
  uint64_t ops1 = 0, ops2 = 0;
  const auto r1 =
      RankOneSpanFamilySearch<2, 9, 9, 9>(span, 12, 1'000'000'000'000, &ops1);
  const auto r2 =
      RankOneSpanFamilySearch<2, 9, 9, 9>(span, 12, 1'000'000'000'000, &ops2);
  EXPECT_EQ(r1, r2);
  EXPECT_EQ(ops1, ops2);
}

// Orbit 62 (lb 12, ub 14): excluding rank <= 12 needs only k <= 3 and is the
// fast real-orbit regression for the family search. Proves lb >= 13.
TEST(FamilySearchTest, N333Orbit62ExcludesRank12) {
  using namespace rank_one_span_internal;
  const auto t = N333ConstrainedTensor({0x1, 0x2, 0x20, 0x50, 0x9C, 0x110});
  const auto t1 = CyclicTranspose<2, 9, 9, 9>(t);
  const auto span = BuildSliceSpanA<2, 9, 9, 9>(t1);
  ASSERT_EQ(span.rho, 9);
  uint64_t ops = 0;
  EXPECT_EQ(
      (RankOneSpanFamilySearch<2, 9, 9, 9>(span, 12, 1'000'000'000'000, &ops)),
      RankOneSpanResult::kExcluded);
  LOG(INFO) << "orbit 62 target 12: ops=" << ops;
}

// ---- k = 5 case-analysis tests ----

// Random 9x3x5 tensors: the family search's k = 5 case machinery must agree
// with the exhaustive v1 enumeration wherever it decides. This is the shape
// regime where line-heavy remainder witnesses occur (F = 7 families, quotient
// 6..9).
// Random 9x3x5 instance number `trial` of the k = 5 test family: even trials
// are planted (per-family limited distinct w's), odd trials uniform random.
Tensor<2, 9, 3, 5> RandomK5Tensor(std::mt19937_64 &rng, int trial) {
  Tensor<2, 9, 3, 5> t = {};
  if (trial % 2 == 0) {
    uint32_t wpool[8][2];
    for (int f = 1; f <= 7; ++f) {
      wpool[f][0] = 1 + rng() % 31;
      wpool[f][1] = 1 + rng() % 31;
    }
    const int r = 12 + static_cast<int>(rng() % 3);
    for (int i = 0; i < r; ++i) {
      const uint32_t u = 1 + (i % 7);
      const uint32_t v = 1 + rng() % 511;
      const uint32_t w = wpool[u][rng() % 2];
      for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 9; ++y) {
          for (int z = 0; z < 5; ++z) {
            if (((u >> x) & 1) && ((v >> y) & 1) && ((w >> z) & 1)) {
              t[y][x][z] = t[y][x][z] + GF<2>::One();
            }
          }
        }
      }
    }
  } else {
    for (int y = 0; y < 9; ++y) {
      for (int x = 0; x < 3; ++x) {
        for (int z = 0; z < 5; ++z) {
          if (rng() % 2) {
            t[y][x][z] = GF<2>::One();
          }
        }
      }
    }
  }
  return t;
}

TEST(FamilySearchK5Test, AgreesWithV1OnSmallRandom) {
  using namespace rank_one_span_internal;
  std::mt19937_64 rng(424242);
  int decided_at_e5 = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Tensor<2, 9, 3, 5> t = RandomK5Tensor(rng, trial);
    const auto span = BuildSliceSpanA<2, 9, 3, 5>(t);
    for (int target = span.rho; target <= span.rho + 5; ++target) {
      const uint64_t v1_cost = CostFromSpan(span, target);
      if (v1_cost > kRankOneSpanV1CostThreshold) {
        continue; // no affordable ground truth
      }
      const RankOneSpanResult v1 =
          EnumerateAllSubspaces<2, 9, 3, 5>(span, target);
      uint64_t ops = 0;
      const RankOneSpanResult v2 =
          RankOneSpanFamilySearch<2, 9, 3, 5>(span, target, kBig, &ops);
      if (v2 != RankOneSpanResult::kOverBudget) {
        EXPECT_EQ(v2, v1) << "trial " << trial << " target " << target
                          << " rho " << span.rho;
        if (target == span.rho + 5) {
          ++decided_at_e5;
        }
      }
    }
  }
  EXPECT_GE(decided_at_e5, 5); // the k = 5 path must actually be exercised
}

// A planted all-3s-profile witness (from the Python design validation): the
// k = 5 remainder machinery must find it.
Tensor<2, 7, 3, 4> PlantedMulti3Tensor() {
  const uint32_t slices[7] = {3554, 3974, 293, 663, 1144, 989, 470};
  Tensor<2, 7, 3, 4> t = {};
  for (int i = 0; i < 7; ++i) {
    for (int x = 0; x < 3; ++x) {
      for (int z = 0; z < 4; ++z) {
        if ((slices[i] >> (x * 4 + z)) & 1) {
          t[i][x][z] = GF<2>::One();
        }
      }
    }
  }
  return t;
}

TEST(FamilySearchK5Test, PlantedMulti3Witness) {
  using namespace rank_one_span_internal;
  const auto span = BuildSliceSpanA<2, 7, 3, 4>(PlantedMulti3Tensor());
  ASSERT_EQ(span.rho, 6);
  uint64_t ops = 0;
  EXPECT_EQ((RankOneSpanFamilySearch<2, 7, 3, 4>(span, 11, kBig, &ops)),
            RankOneSpanResult::kWitnessFound);
}

// ---- Golden op counts ----
//
// The family search's op count is load-bearing, not a statistic: the verifier
// re-runs every rank-one-span exclusion under kRankOneSpanVerifierBudget, so a
// change of the charge schedule can turn an accepted certificate into a
// rejected one. These tests pin the deterministic sequential schedule (values
// captured at commit 4378bca6a, before the family search was restructured).
// They may change only with a deliberate change of the schedule, after which
// every certificate carrying rank-one-span proofs must be re-verified against
// the verifier budget.

struct OpCountCase {
  const char *name;
  std::vector<uint16_t> words; // n333 constraint words; axis-1 span
  int target;
  uint64_t budget;
  RankOneSpanResult expected;
  uint64_t expected_ops;
};

void CheckOpCounts(const std::vector<OpCountCase> &cases) {
  using namespace rank_one_span_internal;
  for (const OpCountCase &c : cases) {
    const auto t1 = CyclicTranspose<2, 9, 9, 9>(N333ConstrainedTensor(c.words));
    const auto span = BuildSliceSpanA<2, 9, 9, 9>(t1);
    uint64_t ops = 0;
    const RankOneSpanResult result =
        RankOneSpanFamilySearch<2, 9, 9, 9>(span, c.target, c.budget, &ops);
    EXPECT_EQ(result, c.expected) << c.name;
    EXPECT_EQ(ops, c.expected_ops) << c.name;
    LOG(INFO) << "golden " << c.name << ": result=" << static_cast<int>(result)
              << " ops=" << ops;
  }
}

// Orbits 24/27/28 at targets 10..13 (k = 1..4: the dmax cases and the k = 4
// cases gamma/alpha/beta/delta, with both excluded and witness verdicts) and
// orbit 62 at target 12 (the k <= 3 exclusion, 8.6e7 ops).
TEST(FamilySearchGoldenTest, SequentialOpCounts) {
  using R = RankOneSpanResult;
  const std::vector<uint16_t> o24 = {0x1, 0x2, 0x4, 0x8, 0x50, 0xA0};
  const std::vector<uint16_t> o27 = {0x1, 0x2, 0x4, 0x8, 0xA0, 0x130};
  const std::vector<uint16_t> o28 = {0x1, 0x2, 0x4, 0x50, 0xA0, 0x118};
  const std::vector<uint16_t> o62 = {0x1, 0x2, 0x20, 0x50, 0x9C, 0x110};
  CheckOpCounts({
      {"orbit 24 target 10", o24, 10, kBig, R::kExcluded, 1618},
      {"orbit 24 target 11", o24, 11, kBig, R::kExcluded, 107920},
      {"orbit 24 target 12", o24, 12, kBig, R::kWitnessFound, 131258},
      {"orbit 24 target 13", o24, 13, kBig, R::kWitnessFound, 131258},
      {"orbit 27 target 10", o27, 10, kBig, R::kExcluded, 1534},
      {"orbit 27 target 11", o27, 11, kBig, R::kExcluded, 151558},
      {"orbit 27 target 12", o27, 12, kBig, R::kWitnessFound, 151743},
      {"orbit 27 target 13", o27, 13, kBig, R::kWitnessFound, 151743},
      {"orbit 28 target 10", o28, 10, kBig, R::kExcluded, 1842},
      {"orbit 28 target 11", o28, 11, kBig, R::kExcluded, 60495},
      {"orbit 28 target 12", o28, 12, kBig, R::kExcluded, 1880719},
      {"orbit 28 target 13", o28, 13, kBig, R::kOverBudget, 100560420},
      {"orbit 62 target 12", o62, 12, 1'000'000'000'000, R::kExcluded,
       86208011},
  });
}

// A budget that trips inside the k = 3 dmax = k case of orbit 62: pins the
// flush schedule and the early-exit points, which the excluded-path goldens
// cannot see.
TEST(FamilySearchGoldenTest, OverBudgetOpCount) {
  CheckOpCounts({{"orbit 62 target 12 budget 3e7",
                  {0x1, 0x2, 0x20, 0x50, 0x9C, 0x110},
                  12,
                  30'000'000,
                  RankOneSpanResult::kOverBudget,
                  30186082}});
}

// The k = 5 witness path on the planted instance.
TEST(FamilySearchGoldenTest, K5WitnessOpCount) {
  using namespace rank_one_span_internal;
  const auto span = BuildSliceSpanA<2, 7, 3, 4>(PlantedMulti3Tensor());
  uint64_t ops = 0;
  EXPECT_EQ((RankOneSpanFamilySearch<2, 7, 3, 4>(span, 11, kBig, &ops)),
            RankOneSpanResult::kWitnessFound);
  LOG(INFO) << "golden planted k5 witness: ops=" << ops;
  EXPECT_EQ(ops, 1140u);
}

// FNV-1a checksum of (trial, target, verdict, ops) over the random 9x3x5
// sweep of AgreesWithV1OnSmallRandom: exercises the k = 5 cases C3..C10 and
// pins their schedule in one number.
TEST(FamilySearchGoldenTest, K5SweepChecksum) {
  using namespace rank_one_span_internal;
  std::mt19937_64 rng(424242);
  uint64_t hash = 14695981039346656037ull;
  const auto fold = [&hash](uint64_t x) {
    for (int i = 0; i < 8; ++i) {
      hash ^= (x >> (8 * i)) & 0xFF;
      hash *= 1099511628211ull;
    }
  };
  int pairs = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Tensor<2, 9, 3, 5> t = RandomK5Tensor(rng, trial);
    const auto span = BuildSliceSpanA<2, 9, 3, 5>(t);
    for (int target = span.rho; target <= span.rho + 5; ++target) {
      if (CostFromSpan(span, target) > kRankOneSpanV1CostThreshold) {
        continue;
      }
      uint64_t ops = 0;
      const RankOneSpanResult result =
          RankOneSpanFamilySearch<2, 9, 3, 5>(span, target, kBig, &ops);
      fold(static_cast<uint64_t>(trial));
      fold(static_cast<uint64_t>(target));
      fold(static_cast<uint64_t>(static_cast<int>(result)));
      fold(ops);
      ++pairs;
    }
  }
  LOG(INFO) << "golden k5 sweep: pairs=" << pairs << " hash=" << hash;
  EXPECT_EQ(pairs, 360);
  EXPECT_EQ(hash, 4228596118090779759ull);
}

// Golden soundness sweep over the small cached matrix certificates: the
// exclusion must never contradict a verified upper bound, and wherever the
// certificate has a tight bound (lb == ub) and the enumeration is affordable,
// excluding lb - 1 must succeed (completeness).
template <class Problem> void GoldenSoundnessSweep(const std::string &path) {
  constexpr int P = Problem::kP;
  constexpr std::size_t NA = Problem::kNA;
  constexpr std::size_t NB = Problem::kNB;
  constexpr std::size_t NC = Problem::kNC;
  constexpr uint64_t kBudget = 300'000;

  const pb::Certificate cert = ReadProtoFromFile<pb::Certificate>(path);
  ASSERT_EQ(cert.problem_name(), Problem::Name());
  for (const pb::ConstrainedTensor &rt : cert.constrained_tensors()) {
    if (!rt.has_rank_upper_bound()) {
      continue;
    }
    const Constraints<P, static_cast<int>(NA)> constraints =
        ConstraintsFromBytes<P, static_cast<int>(NA)>(rt.constraints());
    const Tensor<P, NA, NB, NC> tensor =
        ApplyConstraintsToTensor<P, NA, NB, NC>(constraints,
                                                Problem::MakeTensor());
    const int ub = rt.rank_upper_bound();
    EXPECT_NE((RankOneSpanExclude<P, NA, NB, NC>(tensor, ub, kBudget).first),
              RankOneSpanResult::kExcluded)
        << path << " orbit " << rt.index() << ": excluded rank <= ub=" << ub;
    if (rt.has_rank_lower_bound() && rt.rank_lower_bound() == ub && ub > 0) {
      const auto result =
          RankOneSpanExclude<P, NA, NB, NC>(tensor, ub - 1, kBudget).first;
      EXPECT_NE(result, RankOneSpanResult::kWitnessFound)
          << path << " orbit " << rt.index() << ": found rank <= " << ub - 1
          << " below the certified bound";
    }
  }
}

TEST(RankOneSpanGoldenTest, MatrixN222) {
  GoldenSoundnessSweep<matrix::Problem<2, 2, 2, 2>>(
      "certs/matrix/cert_matrix_q02_n222.pb.txt");
}

TEST(RankOneSpanGoldenTest, MatrixN223) {
  GoldenSoundnessSweep<matrix::Problem<2, 2, 2, 3>>(
      "certs/matrix/cert_matrix_q02_n223.pb.txt");
}

TEST(RankOneSpanGoldenTest, MatrixN224) {
  GoldenSoundnessSweep<matrix::Problem<2, 2, 2, 4>>(
      "certs/matrix/cert_matrix_q02_n224.pb.txt");
}

TEST(RankOneSpanGoldenTest, MatrixN233) {
  GoldenSoundnessSweep<matrix::Problem<2, 2, 3, 3>>(
      "certs/matrix/cert_matrix_q02_n233.pb.txt");
}

} // namespace
