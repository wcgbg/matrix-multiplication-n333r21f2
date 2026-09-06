// Golden regression for RankLowerBoundForcedProductA: recompute every
// forced-product proof of the small cached certificates on its recorded cyclic
// position with the verifier's default arguments (mirroring
// VerifyForcedProductProof) and pin the exact value. The values were captured
// at commit 4378bca6a; a different value means the technique's behaviour
// changed, which requires re-verifying every certificate.

#include "core/rank_lower_bound_forced_product.h"

#include <cstddef>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/proto_io.h"
#include "core/rank_lower_bound_flatten.h"
#include "core/tensor.h"
#include "matrix/problem.h"

namespace {

template <class Problem>
int RecomputeForcedProduct(const pb::ConstrainedTensor &rt) {
  constexpr int P = Problem::kP;
  constexpr std::size_t NA = Problem::kNA;
  constexpr std::size_t NB = Problem::kNB;
  constexpr std::size_t NC = Problem::kNC;
  const Constraints<P, static_cast<int>(NA)> constraints =
      ConstraintsFromBytes<P, static_cast<int>(NA)>(rt.constraints());
  const Tensor<P, NA, NB, NC> tensor = ApplyConstraintsToTensor<P, NA, NB, NC>(
      constraints, Problem::MakeTensor());
  const uint32_t proj =
      rt.rank_lower_bound_proof().forced_product_proof().projection_type();
  if (proj == 0) {
    return RankLowerBoundForcedProductA<P, NA, NB, NC>(tensor);
  }
  const Tensor<P, NB, NC, NA> t1 = CyclicTranspose<P, NA, NB, NC>(tensor);
  if (proj == 1) {
    return RankLowerBoundForcedProductA<P, NB, NC, NA>(t1);
  }
  CHECK_EQ(proj, 2u);
  const Tensor<P, NC, NA, NB> t2 = CyclicTranspose<P, NB, NC, NA>(t1);
  return RankLowerBoundForcedProductA<P, NC, NA, NB>(t2);
}

struct ForcedProductGolden {
  int index;
  int expected;
};

template <class Problem>
void CheckCertificate(const std::string &path,
                      const std::vector<ForcedProductGolden> &goldens) {
  const pb::Certificate cert = ReadProtoFromFile<pb::Certificate>(path);
  ASSERT_EQ(cert.problem_name(), Problem::Name());
  std::size_t seen = 0;
  for (const pb::ConstrainedTensor &rt : cert.constrained_tensors()) {
    if (!rt.rank_lower_bound_proof().has_forced_product_proof()) {
      continue;
    }
    const int computed = RecomputeForcedProduct<Problem>(rt);
    LOG(INFO) << path << " orbit " << rt.index() << " projection "
              << rt.rank_lower_bound_proof()
                     .forced_product_proof()
                     .projection_type()
              << ": forced product = " << computed;
    EXPECT_GE(computed, rt.rank_lower_bound()) << path << " orbit "
                                               << rt.index();
    bool found = false;
    for (const ForcedProductGolden &g : goldens) {
      if (g.index == static_cast<int>(rt.index())) {
        EXPECT_EQ(computed, g.expected) << path << " orbit " << rt.index();
        found = true;
      }
    }
    EXPECT_TRUE(found) << path << " orbit " << rt.index()
                       << " has no golden value";
    ++seen;
  }
  EXPECT_EQ(seen, goldens.size()) << path;
}

TEST(ForcedProductGoldenTest, MatrixN222) {
  CheckCertificate<matrix::Problem<2, 2, 2, 2>>(
      "certs/matrix/cert_matrix_q02_n222.pb.txt", {{4, 6}});
}

TEST(ForcedProductGoldenTest, MatrixN233) {
  CheckCertificate<matrix::Problem<2, 2, 3, 3>>(
      "certs/matrix/cert_matrix_q02_n233.pb.txt",
      {{5, 9}, {12, 12}, {15, 12}, {17, 12}});
}

TEST(ForcedProductGoldenTest, MatrixN233OverF3) {
  CheckCertificate<matrix::Problem<3, 2, 3, 3>>(
      "certs/matrix/cert_matrix_q03_n233.pb.txt", {{5, 9}, {12, 12}});
}

// The enumeration size kQ^(r2p_size0 * r1_bc_rows) must fit in 64 bits even
// when the budget is disabled (max_iterations_log2 = 64).
TEST(ForcedProductDeathTest, RejectsEnumerationSizeOverflow) {
  using forced_product_internal::NumIterationsWithinBudget;
  EXPECT_EQ(NumIterationsWithinBudget(2, 3, 4, 64), uint64_t{1} << 12);
  EXPECT_EQ(NumIterationsWithinBudget(2, 9, 7, 64), uint64_t{1} << 63);
  EXPECT_FALSE(NumIterationsWithinBudget(2, 9, 7, 24).has_value());
  EXPECT_DEATH(NumIterationsWithinBudget(2, 8, 8, 64),
               "does not fit in 64 bits");
  EXPECT_DEATH(NumIterationsWithinBudget(3, 5, 9, 64),
               "does not fit in 64 bits");
}

} // namespace
