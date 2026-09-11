#include "profiles/q02_n333/search.h"

#include <numeric>

#include "gtest/gtest.h"

namespace profiles::q02_n333 {
namespace {

std::vector<Perm> IdentityGroup() {
  Perm identity;
  std::iota(identity.begin(), identity.end(), 0);
  return {identity};
}

CapTable SmallTable(int rank) {
  CapTable table(rank);
  for (std::vector<u16> rows :
       std::vector<std::vector<u16>>{{1}, {2}, {1, 2}}) {
    Rref(rows);
    table.lb[KeyOfRref(rows.data(), rows.size())] = 0;
  }
  return table;
}

TEST(BinarySearchTest, OptionsAndIndependentWorkers) {
  const auto group = IdentityGroup();
  const auto table = SmallTable(1);
  Options options;
  options.split_depth = 0;
  options.r = 1;
  options.prop3 = false;
  options.ones_subset = {1, 2};
  options.max_solutions = 1;
  Search original(table, group, options);
  ASSERT_TRUE(original.Prepare({}));
  auto worker = original;
  options.r = 20;
  EXPECT_EQ(worker.RunFrom({}, 0, 0).stop_reason, StopReason::kNodeBudget);
  EXPECT_EQ(original.nodes_visited(), 0);
  EXPECT_TRUE(original.solutions().empty());
  EXPECT_EQ(worker.RunFrom({}, 0, 100).stop_reason, StopReason::kSolutionLimit);
  EXPECT_EQ(worker.solutions().size(), 1);
  EXPECT_EQ(worker.need(), 1);
  EXPECT_TRUE(original.solutions().empty());
  EXPECT_EQ(original.RunFrom({}, 0, 100).stop_reason,
            StopReason::kSolutionLimit);
  EXPECT_EQ(worker.solutions(), original.solutions());
}

TEST(BinarySearchTest, RankOptionsSetCapacities) {
  const auto group = IdentityGroup();
  Options options;
  options.split_depth = 0;
  options.prop3 = false;
  options.r = 1;
  const auto table1 = SmallTable(1);
  Search one(table1, group, options);
  EXPECT_TRUE(one.Check({1}));
  EXPECT_FALSE(one.Check({1, 2}));
  options.r = 2;
  const auto table2 = SmallTable(2);
  Search two(table2, group, options);
  EXPECT_TRUE(two.Check({1, 2}));
}

TEST(BinaryGeometryTest, CanonicalizationAndMatrixOperations) {
  std::vector<u16> rows{3, 1, 2};
  EXPECT_EQ(Rref(rows), 2);
  EXPECT_EQ(rows, (std::vector<u16>{2, 1}));
  EXPECT_TRUE(InSpan(rows.data(), rows.size(), 3));
  EXPECT_FALSE(InSpan(rows.data(), rows.size(), 4));
  EXPECT_EQ(Rank(0x111), 3);
  for (u16 matrix = 0; matrix < 512; ++matrix) {
    EXPECT_EQ(Transpose(Transpose(matrix)), matrix);
    EXPECT_EQ(Mul(matrix, 0x111), matrix);
  }
}

TEST(BinaryOuterCasesTest, StableOrderingAndOptions) {
  const auto stab = StabilizerOfB0(BuildGroup());
  Options options;
  options.split_depth = 0;
  options.min_w = 0;
  options.prop3 = false;
  const auto cases = EnumerateOuter(stab, options);
  ASSERT_EQ(cases.size(), 35);
  EXPECT_TRUE(cases[0].w.empty());
  for (size_t i = 1; i < cases.size(); ++i) {
    EXPECT_TRUE(cases[i - 1].w.size() < cases[i].w.size() ||
                (cases[i - 1].w.size() == cases[i].w.size() &&
                 cases[i - 1].w < cases[i].w));
  }
  options.min_w = 8;
  const auto restricted = EnumerateOuter(stab, options);
  ASSERT_FALSE(restricted.empty());
  std::vector<std::vector<u16>> expected, actual;
  for (const auto &c : cases)
    if (c.w.size() == 8)
      expected.push_back(c.w);
  for (const auto &c : restricted)
    actual.push_back(c.w);
  EXPECT_EQ(actual, expected);
}

TEST(SearchCompletionTest, BudgetAfterFindingSolutionRemainsIncomplete) {
  const auto group = IdentityGroup();
  const auto table = SmallTable(1);
  Options options;
  options.split_depth = 0;
  options.r = 1;
  options.max_solutions = 100;
  options.prop3 = false;
  options.ones_subset = {1, 2};
  Search search(table, group, options);
  ASSERT_TRUE(search.Prepare({}));
  const auto partial = search.RunFrom({}, 0, 1);
  EXPECT_FALSE(partial.solutions.empty());
  EXPECT_FALSE(partial.exhausted());
  EXPECT_EQ(partial.stop_reason, StopReason::kNodeBudget);
  const auto complete = search.RunFrom({}, 0, 100);
  EXPECT_TRUE(complete.exhausted());
  EXPECT_FALSE(complete.solutions.empty());
}

TEST(ProfileOptionsDeathTest, DefaultsMatchPaperAndRestrictionsRequireOptIn) {
  Options options;
  EXPECT_EQ(options.r, 20);
  EXPECT_EQ(options.min_w, 0);
  EXPECT_EQ(options.split_depth, 4);
  EXPECT_FALSE(options.prop3);
  ValidateBinaryOptions(options);
  options.r = 21;
  EXPECT_DEATH(ValidateBinaryOptions(options), "--experimental=true");
  options.experimental = true;
  ValidateBinaryOptions(options);
  options.prop3 = true;
  EXPECT_DEATH(ValidateBinaryOptions(options), "--prop3 is valid only at r=20");
  options.prop3 = false;
  options.r = 33;
  EXPECT_DEATH(ValidateBinaryOptions(options),
               "--r exceeds the supported rank");
  options.r = 23;
  options.experimental = false;
  options.check_list = {1};
  ValidateBinaryOptions(options);
  options.check_list = {65537};
  EXPECT_DEATH(ValidateBinaryOptions(options), "factor code is out of range");
}

TEST(ProfileOptionsDeathTest, RejectsMalformedListsAndInvalidRanges) {
  for (const std::string csv :
       {"-1", "1x", "1,", ",1", "1,,2", "4294967296", " 1"})
    EXPECT_DEATH(profiles::ParseList(csv), "invalid unsigned integer list")
        << csv;
  EXPECT_DEATH(profiles::ParseList<uint16_t>("65536"),
               "invalid unsigned integer list");
  EXPECT_EQ(profiles::ParseList("1,511"), (std::vector<uint32_t>{1, 511}));
  EXPECT_TRUE(profiles::ParseList("").empty());
  EXPECT_EQ(profiles::ParseList<size_t>("0,34"), (std::vector<size_t>{0, 34}));
  EXPECT_EQ(profiles::ParseList<uint64_t>("18446744073709551615"),
            (std::vector<uint64_t>{UINT64_MAX}));
  Options options;
  options.min_w = -1;
  EXPECT_DEATH(ValidateBinaryOptions(options), "--min_w must be nonnegative");
  options.min_w = 0;
  options.max_solutions = 0;
  EXPECT_DEATH(ValidateBinaryOptions(options),
               "solution limit must be positive");
  options.max_solutions = 1;
  options.ones_subset = {1, 1};
  EXPECT_DEATH(ValidateBinaryOptions(options),
               "--ones_subset must not contain duplicate codes");
  options.ones_subset = {0};
  EXPECT_DEATH(ValidateBinaryOptions(options), "factor code must be nonzero");
}

} // namespace
} // namespace profiles::q02_n333
