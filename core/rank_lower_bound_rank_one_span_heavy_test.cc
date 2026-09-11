// Heavy (minutes-long) family-search regressions: the k = 4 exclusions that
// close the n333 dim-6 gap orbits 36/50/62 and lift orbit 45 to 14. Run
// manually (tags = ["manual"]); these are the same computations the targeted
// certificate rerun performs and the verifier replays.
//
// The pinned op counts are the totals of the deterministic charge schedule
// (captured at commit 4378bca6a). On the excluded path every worker completes
// its share, so the total does not depend on scheduling even in parallel
// mode. They may change only with a deliberate change of the schedule, after
// which the certificates must be re-verified against the verifier budget.

#include "core/rank_lower_bound_rank_one_span.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "core/constraints.h"
#include "core/rank_lower_bound_flatten.h"
#include "core/tensor.h"
#include "matrix/problem.h"

namespace {

Tensor<2, 9, 9, 9> N333ConstrainedTensor(const std::vector<uint16_t> &words) {
  using Problem = matrix::Problem<2, 3, 3, 3>;
  Constraints<2, 9> constraints;
  for (uint16_t w : words) {
    constraints.push_back(GFVec<2, 9>{static_cast<BitVec<9>>(w)});
  }
  return ApplyConstraintsToTensor<2, 9, 9, 9>(constraints,
                                              Problem::MakeTensor());
}

constexpr uint64_t kBudget = 20'000'000'000'000;

RankOneSpanResult FamilyExcludeAxis1(const std::vector<uint16_t> &words,
                                     int target, uint64_t *ops) {
  using namespace rank_one_span_internal;
  const auto t = N333ConstrainedTensor(words);
  const auto t1 = CyclicTranspose<2, 9, 9, 9>(t);
  const auto span = BuildSliceSpanA<2, 9, 9, 9>(t1);
  return RankOneSpanFamilySearch<2, 9, 9, 9>(span, target, kBudget, ops,
                                             /*parallel=*/true);
}

TEST(FamilySearchHeavyTest, N333Orbit36ExcludesRank13) {
  uint64_t ops = 0;
  EXPECT_EQ(FamilyExcludeAxis1({0x1, 0x2, 0x8, 0x14, 0x60, 0x100}, 13, &ops),
            RankOneSpanResult::kExcluded); // lb 13 -> 14 = ub
  LOG(INFO) << "orbit 36 target 13: ops=" << ops;
  EXPECT_EQ(ops, 154'922'960'679u);
}

TEST(FamilySearchHeavyTest, N333Orbit50ExcludesRank13) {
  uint64_t ops = 0;
  EXPECT_EQ(FamilyExcludeAxis1({0x1, 0x2, 0x8, 0x60, 0x84, 0x110}, 13, &ops),
            RankOneSpanResult::kExcluded); // lb 13 -> 14 = ub
  LOG(INFO) << "orbit 50 target 13: ops=" << ops;
  EXPECT_EQ(ops, 142'823'512'599u);
}

TEST(FamilySearchHeavyTest, N333Orbit62ExcludesRank13) {
  uint64_t ops = 0;
  EXPECT_EQ(FamilyExcludeAxis1({0x1, 0x2, 0x20, 0x50, 0x9C, 0x110}, 13, &ops),
            RankOneSpanResult::kExcluded); // lb 12 -> 14 = ub (with target 12)
  LOG(INFO) << "orbit 62 target 13: ops=" << ops;
  EXPECT_EQ(ops, 27'498'205'063u);
}

TEST(FamilySearchHeavyTest, N333Orbit45ExcludesRank13) {
  uint64_t ops = 0;
  EXPECT_EQ(FamilyExcludeAxis1({0x1, 0x2, 0x8, 0x44, 0xA0, 0x100}, 13, &ops),
            RankOneSpanResult::kExcluded); // lb 13 -> 14 (ub 15)
  LOG(INFO) << "orbit 45 target 13: ops=" << ops;
  EXPECT_EQ(ops, 156'645'723'218u);
}

// The k = 5 exclusions closing the ub-15 dim-6 group at 15. A kWitnessFound
// here would mean the flip-graph upper bound 15 is not tight for that orbit —
// report it rather than treating the test failure as a code bug.
struct K5Orbit {
  int index = 0;
  std::vector<uint16_t> words;
};

class FamilySearchK5HeavyTest : public testing::TestWithParam<K5Orbit> {};

TEST_P(FamilySearchK5HeavyTest, ExcludesRank14) {
  uint64_t ops = 0;
  const auto result = FamilyExcludeAxis1(GetParam().words, 14, &ops);
  LOG(INFO) << "orbit " << GetParam().index << " target 14: ops=" << ops
            << " result="
            << (result == RankOneSpanResult::kExcluded       ? "excluded"
                : result == RankOneSpanResult::kWitnessFound ? "witness_found"
                                                             : "over_budget");
  EXPECT_EQ(result, RankOneSpanResult::kExcluded)
      << "orbit " << GetParam().index
      << (result == RankOneSpanResult::kWitnessFound
              ? " found a rank<=14 witness: ub 15 is improvable"
              : " ran over budget: no claim");
}

INSTANTIATE_TEST_SUITE_P(
    N333Ub15Orbits, FamilySearchK5HeavyTest,
    testing::Values(K5Orbit{33, {0x1, 0x2, 0x8, 0x14, 0x44, 0xA0}},
                    K5Orbit{45, {0x1, 0x2, 0x8, 0x44, 0xA0, 0x100}},
                    K5Orbit{47, {0x1, 0x2, 0x8, 0x44, 0xA0, 0x130}},
                    K5Orbit{57, {0x1, 0x2, 0xC, 0x60, 0x84, 0x110}},
                    K5Orbit{63, {0x1, 0xA, 0x10, 0x44, 0xA0, 0x100}},
                    K5Orbit{68, {0x1, 0xA, 0x10, 0x44, 0xA4, 0x120}},
                    K5Orbit{79, {0x1, 0xA, 0x14, 0x60, 0xA0, 0x102}},
                    K5Orbit{82, {0x1, 0xA, 0x14, 0x60, 0xA0, 0x124}},
                    K5Orbit{83, {0x1, 0xA, 0x20, 0x44, 0x90, 0x112}},
                    K5Orbit{84, {0x1, 0xA, 0x20, 0x54, 0x80, 0x102}}),
    [](const testing::TestParamInfo<K5Orbit> &info) {
      return "Orbit" + std::to_string(info.param.index);
    });

} // namespace
