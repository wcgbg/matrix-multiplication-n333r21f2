#include "subspace_bounds/subspace_bounds.h"

#include <algorithm>
#include <array>
#include <set>

#include "matrix/problem.h"
#include "gtest/gtest.h"

namespace {

// GL(3, P) is transitive on each dimension for a 1x3 first factor. Use one
// coordinate space per orbit and distinct synthetic bounds to test expansion.
template <int P> pb::Certificate Certificate() {
  using Problem = matrix::Problem<P, 1, 3, 1>;
  pb::Certificate cert;
  cert.set_problem_name(Problem::Name());
  for (int dim = 0; dim <= 3; ++dim) {
    Constraints<P, 3> rows;
    int code = 1;
    for (int i = 0; i < dim; ++i, code *= P)
      rows.push_back(SubspaceVecFromCode<P, 3>(code));
    GaussJordanRREF<P, 3>(&rows);
    auto *rt = cert.add_constrained_tensors();
    rt->set_constraints(ConstraintsToBytes(rows));
    rt->set_rank_lower_bound(10 - dim);
  }
  return cert;
}

template <int P> void CheckExpansion() {
  using Problem = matrix::Problem<P, 1, 3, 1>;
  auto records = ExpandSubspaceBounds<Problem>(Certificate<P>());
  std::array<int, 4> counts{};
  std::set<SubspaceBound<3>> unique;
  for (const auto &record : records) {
    ASSERT_LE(record.dim, 3);
    ++counts[record.dim];
    EXPECT_EQ(record.lb, 10 - record.dim);
    EXPECT_TRUE(unique.insert(record).second);
    Constraints<P, 3> rows;
    for (int i = 0; i < record.dim; ++i) {
      ASSERT_GT(record.rows[i], 0);
      ASSERT_LT(record.rows[i], IntPow(P, 3));
      rows.push_back(SubspaceVecFromCode<P, 3>(record.rows[i]));
    }
    EXPECT_EQ((GaussJordanRREF<P, 3>(&rows)), record.dim);
    for (int i = 0; i < record.dim; ++i)
      EXPECT_EQ(SubspaceCodeFromVec(rows[i]), record.rows[i]);
    for (int i = record.dim; i < 3; ++i)
      EXPECT_EQ(record.rows[i], 0);
  }
  const int lines = P * P + P + 1;
  EXPECT_EQ(counts, (std::array<int, 4>{1, lines, lines, 1}));
  // Every nonzero vector gives a line, including non-coordinate lines.
  for (int code = 1; code < IntPow(P, 3); ++code) {
    Constraints<P, 3> line{SubspaceVecFromCode<P, 3>(code)};
    GaussJordanRREF<P, 3>(&line);
    SubspaceBound<3> key;
    key.dim = 1;
    key.rows[0] = SubspaceCodeFromVec(line[0]);
    EXPECT_TRUE(unique.contains(key));
  }
}

TEST(SubspaceBoundsTest, BinaryExpansion) { CheckExpansion<2>(); }
TEST(SubspaceBoundsTest, TernaryExpansion) { CheckExpansion<3>(); }

TEST(SubspaceBoundsDeathTest, RejectsWrongProblem) {
  using Problem = matrix::Problem<2, 1, 3, 1>;
  EXPECT_DEATH(ExpandSubspaceBounds<Problem>(Certificate<3>()), "problem_name");
}

TEST(SubspaceBoundsDeathTest, RejectsMissingBound) {
  using Problem = matrix::Problem<2, 1, 3, 1>;
  auto cert = Certificate<2>();
  cert.mutable_constrained_tensors(1)->clear_rank_lower_bound();
  EXPECT_DEATH(ExpandSubspaceBounds<Problem>(cert), "has no bound");
}

TEST(SubspaceBoundsDeathTest, RejectsMissingOrbit) {
  using Problem = matrix::Problem<2, 1, 3, 1>;
  auto cert = Certificate<2>();
  cert.mutable_constrained_tensors()->RemoveLast();
  EXPECT_DEATH(ExpandSubspaceBounds<Problem>(cert), "not found in OrbitMap");
}

} // namespace
