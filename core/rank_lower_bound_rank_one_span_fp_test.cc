#include "core/rank_lower_bound_rank_one_span_fp.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/proto_io.h"
#include "core/rank_lower_bound_flatten.h" // CyclicTranspose
#include "core/rank_lower_bound_rank_one_span.h"
#include "core/tensor.h"
#include "matrix/problem.h"
#include "matrix/tensor.h"

namespace {

constexpr uint64_t kBudget = 50'000'000;
constexpr auto kExcluded = RankOneSpanResult::kExcluded;
constexpr auto kWitness = RankOneSpanResult::kWitnessFound;
constexpr auto kOver = RankOneSpanResult::kOverBudget;

TEST(SubspaceCountPTest, SmallValues) {
  using rank_one_span_fp_internal::SubspaceCountP;
  // Σ_{k≤e} GB(n, k)_3: GB(6,1) = 364, GB(6,2) = 11011, GB(6,3) = 33880.
  EXPECT_EQ(SubspaceCountP<3>(6, 0), 1u);
  EXPECT_EQ(SubspaceCountP<3>(6, 1), 365u);
  EXPECT_EQ(SubspaceCountP<3>(6, 2), 11376u);
  EXPECT_EQ(SubspaceCountP<3>(6, 3), 45256u);
  // Over F_2 it must match the F_2 engine's counter.
  for (int dq = 0; dq <= 12; ++dq) {
    for (int e = 0; e <= dq; ++e) {
      const uint64_t a = SubspaceCountP<2>(dq, e);
      const uint64_t b = rank_one_span_internal::SubspaceCount2(dq, e);
      EXPECT_EQ(a, b) << dq << " " << e;
    }
  }
}

// diag(1, 2, 1) over F_3 has rank 3.
TEST(RankOneSpanFpTest, DiagonalF3) {
  Tensor<3, 3, 3, 3> t{};
  t[0][0][0] = GF<3>{1};
  t[1][1][1] = GF<3>{2};
  t[2][2][2] = GF<3>{1};
  auto fp = [&](int target) { return RankOneSpanExcludeFpA<3, 3, 3, 3>(t, target, kBudget); };
  EXPECT_EQ(fp(2), kExcluded);
  EXPECT_EQ(fp(3), kWitness);
  // The dispatchers route P = 3 here.
  const auto via_a = RankOneSpanExcludeA<3, 3, 3, 3>(t, 2, kBudget);
  EXPECT_EQ(via_a, kExcluded);
  const auto via_wrapper = RankOneSpanExclude<3, 3, 3, 3>(t, 2, kBudget);
  EXPECT_EQ(via_wrapper.first, kExcluded);
}

// <1,2,2> over F_3 (a 1x2 times a 2x2 matrix) has rank 4 = NB.
TEST(RankOneSpanFpTest, MatMul122F3) {
  const auto t = matrix::BuildMulTensor<3, 1, 2, 2>(); // Tensor<3,1,2,4,2>
  auto fp = [&](int target) { return RankOneSpanExcludeFpA<3, 2, 4, 2>(t, target, kBudget); };
  EXPECT_EQ(fp(3), kExcluded);
  EXPECT_EQ(fp(4), kWitness);
}

// <1,2,3> over F_3 has rank 6. Sliced along C (two cyclic transposes) the
// core is 2x6 with rho = 3, so target 5 sweeps the 2-dim subspaces of a
// 9-dim quotient (8,069,620 of them) and must exclude.
TEST(RankOneSpanFpTest, MatMul123F3AlongC) {
  const auto t = matrix::BuildMulTensor<3, 1, 2, 3>(); // Tensor<3,1,2,6,3>
  const auto t1 = CyclicTranspose<3, 2, 6, 3>(t);       // Tensor<3,1,6,3,2>
  const auto t2 = CyclicTranspose<3, 6, 3, 2>(t1);      // Tensor<3,1,3,2,6>
  auto along_c = [&](int target, bool parallel) {
    return RankOneSpanExcludeFpA<3, 3, 2, 6>(t2, target, kBudget, parallel);
  };
  EXPECT_EQ(along_c(4, false), kExcluded);
  EXPECT_EQ(along_c(5, true), kExcluded);
  // Along B the slice span is the whole 2x3 core: rank 6 is witnessed at k = 0.
  auto along_b = [&](int target) { return RankOneSpanExcludeFpA<3, 6, 3, 2>(t1, target, kBudget); };
  EXPECT_EQ(along_b(6), kWitness);
  EXPECT_EQ(along_b(5), kExcluded);
}

constexpr uint64_t kAgreeBudget = 2'000'000;

// Compare the F_2 engine and the generic engine at P = 2 on one tensor.
template <std::size_t A, std::size_t B, std::size_t C>
void CompareEngines(const Tensor<2, A, B, C> &tensor, int target, const std::string &label,
                    int *compared) {
  const uint64_t cost_f2 = RankOneSpanCostA<2, A, B, C>(tensor, target);
  const uint64_t cost_fp = RankOneSpanCostFpA<2, A, B, C>(tensor, target);
  EXPECT_EQ(cost_f2, cost_fp) << label;
  const auto f2 = RankOneSpanExcludeA<2, A, B, C>(tensor, target, kAgreeBudget);
  const auto fp = RankOneSpanExcludeFpA<2, A, B, C>(tensor, target, kAgreeBudget);
  if (f2 != kOver && fp != kOver) {
    EXPECT_EQ(f2, fp) << label;
    ++*compared;
  }
}

// The generic engine instantiated at P = 2 must agree with the trusted F_2
// engine (cost and verdict) on every orbit of the small F_2 certificates,
// on all three slicing axes.
template <int N0, int N1, int N2> void AgreeWithF2Engine(const std::string &path) {
  using Problem = matrix::Problem<2, N0, N1, N2>;
  constexpr std::size_t NA = Problem::kNA, NB = Problem::kNB, NC = Problem::kNC;
  const pb::Certificate cert = ReadProtoFromFile<pb::Certificate>(path);
  const auto full = Problem::MakeTensor();
  int compared = 0;
  for (const pb::ConstrainedTensor &rt : cert.constrained_tensors()) {
    const auto constraints = ConstraintsFromBytes<2, static_cast<int>(NA)>(rt.constraints());
    const Tensor<2, NA, NB, NC> t =
        ApplyConstraintsToTensor<2, NA, NB, NC>(constraints, full);
    const Tensor<2, NB, NC, NA> t1 = CyclicTranspose<2, NA, NB, NC>(t);
    const Tensor<2, NC, NA, NB> t2 = CyclicTranspose<2, NB, NC, NA>(t1);
    const int lb = rt.rank_lower_bound();
    for (int target = std::max(0, lb - 2); target <= lb; ++target) {
      const std::string label =
          path + " orbit " + std::to_string(rt.index()) + " target " + std::to_string(target);
      CompareEngines<NA, NB, NC>(t, target, label + " axis 0", &compared);
      CompareEngines<NB, NC, NA>(t1, target, label + " axis 1", &compared);
      CompareEngines<NC, NA, NB>(t2, target, label + " axis 2", &compared);
    }
  }
  EXPECT_GT(compared, 0) << path;
}

TEST(RankOneSpanFpTest, AgreesWithF2EngineN222) {
  AgreeWithF2Engine<2, 2, 2>("certs/matrix/cert_matrix_q02_n222.pb.txt");
}
TEST(RankOneSpanFpTest, AgreesWithF2EngineN223) {
  AgreeWithF2Engine<2, 2, 3>("certs/matrix/cert_matrix_q02_n223.pb.txt");
}
TEST(RankOneSpanFpTest, AgreesWithF2EngineN233) {
  AgreeWithF2Engine<2, 3, 3>("certs/matrix/cert_matrix_q02_n233.pb.txt");
}

// Soundness against the certified F_3 bounds of <2,3,3>: the exclusion at
// lb - 1 must never report a witness (that would mean rank <= lb - 1).
TEST(RankOneSpanFpTest, NeverContradictsF3Certificate) {
  using Problem = matrix::Problem<3, 2, 3, 3>;
  const pb::Certificate cert =
      ReadProtoFromFile<pb::Certificate>("certs/matrix/cert_matrix_q03_n233.pb.txt");
  const auto full = Problem::MakeTensor();
  int decided = 0;
  for (const pb::ConstrainedTensor &rt : cert.constrained_tensors()) {
    const auto constraints = ConstraintsFromBytes<3, 6>(rt.constraints());
    const Tensor<3, 6, 9, 6> t = ApplyConstraintsToTensor<3, 6, 9, 6>(constraints, full);
    const int lb = rt.rank_lower_bound();
    if (lb == 0) {
      continue;
    }
    const auto verdict = RankOneSpanExclude<3, 6, 9, 6>(t, lb - 1, 2'000'000);
    EXPECT_NE(verdict.first, kWitness) << "orbit " << rt.index();
    if (verdict.first == kExcluded) {
      ++decided;
    }
  }
  EXPECT_GT(decided, 0);
}

// The pivot-set enumeration uses 32-bit masks; a quotient of dimension 32 or
// more must be rejected rather than shift out of range.
TEST(RankOneSpanFpDeathTest, RejectsQuotientTooWideForPivotMasks) {
  using rank_one_span_fp_internal::PivotSetsOfSize;
  EXPECT_EQ(PivotSetsOfSize(3, 2).size(), 3u);
  EXPECT_DEATH(PivotSetsOfSize(32, 1), "32-bit pivot masks");
}

} // namespace
