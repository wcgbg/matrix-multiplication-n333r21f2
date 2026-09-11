#include "profiles/fp/search.h"

#include "gtest/gtest.h"

namespace profiles::fp {
namespace {

// Only these four spans can be touched by candidates 1 and 2. Synthetic
// bounds let us test option propagation independently of certificate proofs.
ProblemData<2, 2, 2> SmallData() {
  using G = Geometry<2, 2, 2>;
  ProblemData<2, 2, 2> data;
  data.BuildPoints();
  data.BuildGroup();
  for (const auto &rows : std::vector<std::vector<G::Digits>>{
           {}, {G::Decode(1)}, {G::Decode(2)}, {G::Decode(1), G::Decode(2)}})
    data.table_[G::Key(G::Rref(rows))] = 0;
  return data;
}

TEST(ProfileSearchTest, RankOptionsSetCapacities) {
  const auto data = SmallData();
  profiles::Options options;
  options.r = 1;
  Search<2, 2, 2> one(data, options);
  one.SetupCandidates({1, 2});
  EXPECT_TRUE(one.Check({1}));
  EXPECT_FALSE(one.Check({1, 2}));
  options.r = 2;
  Search<2, 2, 2> two(data, options);
  two.SetupCandidates({1, 2});
  EXPECT_TRUE(two.Check({1, 2}));
  EXPECT_TRUE(two.Check({1, 1}));
}

TEST(ProfileSearchTest, WorkersShareDataButHaveIndependentStateAndOptions) {
  const auto data = SmallData();
  profiles::Options options;
  options.r = 1;
  options.max_solutions = 1;
  Search<2, 2, 2> original(data, options);
  original.SetupCandidates({1, 2});
  auto worker = original;
  EXPECT_EQ(&worker.data(), &data);
  EXPECT_EQ(&original.data(), &data);
  // Each search holds its own immutable option snapshot.
  options.r = 20;
  EXPECT_EQ(worker.RunFrom({}, 0, 0).stop_reason, StopReason::kNodeBudget);
  EXPECT_EQ(worker.nodes_visited(), 1);
  EXPECT_EQ(original.nodes_visited(), 0);
  EXPECT_TRUE(original.solutions().empty());
  EXPECT_EQ(worker.RunFrom({}, 0, 100).stop_reason, StopReason::kSolutionLimit);
  EXPECT_EQ(worker.solutions().size(), 1);
  EXPECT_TRUE(original.solutions().empty());
  EXPECT_EQ(original.RunFrom({}, 0, 100).stop_reason,
            StopReason::kSolutionLimit);
  EXPECT_EQ(worker.solutions(), original.solutions());
}

template <int P> void CheckGeometry() {
  using G = Geometry<P, 2, 2>;
  const auto a = G::Decode(1), b = G::Decode(P);
  const auto ab = G::Decode(P + 1);
  const auto span = G::Rref({a, b});
  EXPECT_EQ(span.dim, 2);
  EXPECT_EQ(G::Key(span), G::Key(G::Rref({ab, a, b})));
  EXPECT_TRUE(G::InSpan(span, ab));
  EXPECT_FALSE(G::InSpan(span, G::Decode(P * P)));
  EXPECT_EQ(G::MatRank(G::Decode(1 + P * P * P)), 2);
  for (int code = 0; code < G::Q; ++code)
    EXPECT_EQ(G::Encode(G::Decode(code)), code);
}

TEST(ProfileGeometryTest, BinaryAndTernaryCanonicalization) {
  CheckGeometry<2>();
  CheckGeometry<3>();
}

TEST(SearchCompletionTest, BudgetAfterFindingSolutionRemainsIncomplete) {
  const auto data = SmallData();
  profiles::Options options;
  options.r = 1;
  options.max_solutions = 100;
  Search<2, 2, 2> search(data, options);
  search.SetupCandidates({1, 2});
  const auto partial = search.RunFrom({}, 0, 1);
  EXPECT_FALSE(partial.solutions.empty());
  EXPECT_FALSE(partial.exhausted());
  EXPECT_EQ(partial.stop_reason, StopReason::kNodeBudget);
  const auto complete = search.RunFrom({}, 0, 100);
  EXPECT_TRUE(complete.exhausted());
  EXPECT_FALSE(complete.solutions.empty());
}

} // namespace
} // namespace profiles::fp
