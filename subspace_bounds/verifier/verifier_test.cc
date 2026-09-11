#include "subspace_bounds/verifier/verifier.h"

#include <string>
#include <utility>
#include <vector>

#include "tbb/global_control.h"
#include "gtest/gtest.h"

#include "core/backtracking_proof.h"
#include "core/certificate.pb.h"
#include "matrix/problem.h"
#include "subspace_bounds/search/orbit_enumerator_slow.h"
#include "subspace_bounds/search/rank_lower_bound_computer.h"

namespace {

// Enumerate orbits and run the full DP search, writing the certificate (and the
// backtracking-proof archive) under a temporary directory.
template <class Problem>
std::pair<pb::Certificate, std::string>
BuildVerifiedCertificate(const ProcessOptions &options = ProcessOptions{}) {
  const std::string temp_dir = testing::TempDir();
  const std::string output_path = temp_dir + "/" + Problem::Name() + ".pb.txt";

  const typename Problem::SymmetryGroup group;
  const OrbitEnumeratorSlow<Problem> enumerator(&group);
  pb::Certificate certificate = enumerator.Search();

  ProcessOrbits<Problem>(options, output_path, &certificate);
  return {certificate, GetBacktrackingProofArchivePath(output_path)};
}

template <class Problem>
int RunPipeline(const ProcessOptions &options = ProcessOptions{}) {
  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>(options);
  if (certificate.constrained_tensors_size() == 0)
    return -1;
  EXPECT_EQ(certificate.characteristic(), Problem::kP);
  return VerifyRankLowerBound<Problem>(certificate, archive_path);
}

int FirstBacktrackingOrbit(const pb::Certificate &certificate) {
  for (int i = 0; i < certificate.constrained_tensors_size(); ++i) {
    if (certificate.constrained_tensors(i)
            .rank_lower_bound_proof()
            .has_backtracking_proof()) {
      return i;
    }
  }
  return -1;
}

TEST(VerifierTest, VerifiesMatMul222) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  // rank_F2(<2,2,2>) = 7 (Strassen / Winograd), reached by the default DP.
  EXPECT_EQ(RunPipeline<Problem>(), 7);
}

TEST(VerifierTest, MatMul222WithRankOneSpan) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  // Same pipeline with the rank-one-span technique enabled: whatever mix of
  // proof variants results (including rank_one_span_proof on the orbits where
  // the exclusion is affordable) must verify to the same bound.
  EXPECT_EQ(RunPipeline<Problem>(
                ProcessOptions{.rank1span_max_subspaces = 1'000'000}),
            7);
}

TEST(VerifierTest, AcceptsSoundHandWrittenRankOneSpanProof) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>();
  pb::ConstrainedTensor *last = certificate.mutable_constrained_tensors(
      certificate.constrained_tensors_size() - 1);
  ASSERT_GE(last->rank_lower_bound(), 6); // true rank is 7
  // Downgrade the top orbit's claim to 6 with a rank-one-span proof: the
  // verifier re-runs the exclusion at target 5 (e = 1, cheap) and must accept.
  last->set_rank_lower_bound(6);
  last->clear_rank_lower_bound_proof();
  last->mutable_rank_lower_bound_proof()
      ->mutable_rank_one_span_proof()
      ->set_slice_axis(0);
  EXPECT_EQ(VerifyRankLowerBound<Problem>(certificate, archive_path), 6);
}

// Odd prime field: the GF<3> arithmetic path of every technique and of the
// verifier (the F_2 BitVec fast paths are not used).
TEST(VerifierTest, MatMul122OverF3) {
  using Problem = matrix::Problem<3, 1, 2, 2>;
  // <1,2,2> has rank 4 = NB over every field (its B-flattening is a 4x4
  // permutation), so the DP is exact here.
  EXPECT_EQ(RunPipeline<Problem>(), 4);
}

TEST(VerifierTest, MatMul222OverF3) {
  using Problem = matrix::Problem<3, 2, 2, 2>;
  // rank_F3(<2,2,2>) = 7.
  EXPECT_EQ(RunPipeline<Problem>(
                ProcessOptions{.rank1span_max_subspaces = 1'000'000}),
            7);
}

TEST(VerifierDeathTest, DetectsInflatedBound) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  // EXPECT_DEATH only works with a single thread.
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>();
  ASSERT_GT(certificate.constrained_tensors_size(), 0);
  pb::ConstrainedTensor *last = certificate.mutable_constrained_tensors(
      certificate.constrained_tensors_size() - 1);
  last->set_rank_lower_bound(last->rank_lower_bound() + 1);
  EXPECT_DEATH(VerifyRankLowerBound<Problem>(certificate, archive_path), "");
}

TEST(VerifierDeathTest, DetectsInflatedRankOneSpanBound) {
  using Problem = matrix::Problem<2, 1, 1, 1>;
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>();
  pb::ConstrainedTensor *last = certificate.mutable_constrained_tensors(
      certificate.constrained_tensors_size() - 1);
  // <1,1,1> has rank 1; claim 2 via rank-one-span. The re-run at target 1
  // finds a witness (V itself is rank-one spanned) and must abort.
  last->set_rank_lower_bound(2);
  last->clear_rank_lower_bound_proof();
  last->mutable_rank_lower_bound_proof()
      ->mutable_rank_one_span_proof()
      ->set_slice_axis(0);
  EXPECT_DEATH(VerifyRankLowerBound<Problem>(certificate, archive_path),
               "rank-one-span exclusion did not hold");
}

TEST(VerifierDeathTest, RejectsRankOneSpanInvalidAxis) {
  using Problem = matrix::Problem<2, 1, 1, 1>;
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>();
  pb::ConstrainedTensor *last = certificate.mutable_constrained_tensors(
      certificate.constrained_tensors_size() - 1);
  last->clear_rank_lower_bound_proof();
  last->mutable_rank_lower_bound_proof()
      ->mutable_rank_one_span_proof()
      ->set_slice_axis(3);
  EXPECT_DEATH(VerifyRankLowerBound<Problem>(certificate, archive_path),
               "Invalid slice_axis");
}

TEST(VerifierDeathTest, RejectsBacktrackingLeafPastDepthBound) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>();
  const int orbit = FirstBacktrackingOrbit(certificate);
  ASSERT_GE(orbit, 0);

  BacktrackingProofArchive archive =
      BacktrackingProofArchive::Load(archive_path);
  BacktrackingProof proof = archive.Get(orbit);
  ASSERT_FALSE(proof.Empty());
  const int claimed_bound =
      certificate.constrained_tensors(orbit).rank_lower_bound();
  ASSERT_GT(claimed_bound, 0);
  ASSERT_LE(claimed_bound, 31);
  proof.dfs_constraints_size_array[0] = static_cast<uint8_t>(claimed_bound);
  archive.Set(orbit, proof);
  archive.Save(archive_path);

  EXPECT_DEATH(VerifyRankLowerBound<Problem>(certificate, archive_path),
               "exceeds the L-1 decomposition depth");
}

TEST(VerifierDeathTest, RejectsBacktrackingMaskOutsidePath) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);

  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>();
  const int orbit = FirstBacktrackingOrbit(certificate);
  ASSERT_GE(orbit, 0);

  BacktrackingProofArchive archive =
      BacktrackingProofArchive::Load(archive_path);
  BacktrackingProof proof = archive.Get(orbit);
  ASSERT_FALSE(proof.Empty());
  const uint8_t depth = proof.dfs_constraints_size_array[0];
  ASSERT_GT(depth, 0);
  ASSERT_LT(depth, 32);
  proof.mask_array[0] |= uint32_t{1} << depth;
  archive.Set(orbit, proof);
  archive.Save(archive_path);

  EXPECT_DEATH(VerifyRankLowerBound<Problem>(certificate, archive_path),
               "outside the recorded path");
}

template <class Problem> void CheckWitnessDecoding() {
  const typename Problem::SymmetryGroup group;
  Constraints<Problem::kP, Problem::kNA> q{
      DecodeGFVec<Problem::kP, Problem::kNA>(1)};
  const auto qi = static_cast<uint32_t>(group.query.Identity());
  const auto si = static_cast<uint32_t>(group.store.Identity());
  EXPECT_EQ(CanonicalFromWitness<Problem>(group, q, qi, si), q);
  for (int i = 0; i < group.query.Size(); ++i) {
    const auto code = static_cast<uint32_t>(group.query.At(i));
    EXPECT_EQ(static_cast<uint32_t>(group.query.DecodeChecked(code)), code);
  }
  for (int i = 0; i < group.store.Size(); ++i) {
    const auto code = static_cast<uint32_t>(group.store.At(i));
    EXPECT_EQ(static_cast<uint32_t>(group.store.DecodeChecked(code)), code);
  }
  EXPECT_DEATH(CanonicalFromWitness<Problem>(group, q, 0, si),
               "singular query");
  EXPECT_DEATH(CanonicalFromWitness<Problem>(group, q, qi, 0),
               "singular store");
  EXPECT_DEATH(CanonicalFromWitness<Problem>(group, q, qi, UINT32_MAX),
               "invalid store");
  EXPECT_DEATH(CanonicalFromWitness<Problem>(group, q, UINT32_MAX, si),
               "invalid");
}

TEST(VerifierDeathTest, ChecksBinaryAndTernaryWitnessEncodings) {
  CheckWitnessDecoding<matrix::Problem<2, 2, 2, 2>>();
  CheckWitnessDecoding<matrix::Problem<3, 2, 2, 2>>();
}

TEST(VerifierDeathTest, RejectsTransposeOnRectangularProblem) {
  const matrix::SymmetryGroup<2, 2, 3, 3> binary;
  EXPECT_DEATH(binary.query.DecodeChecked(
                   static_cast<uint32_t>(binary.query.Identity()) | (1u << 16)),
               "transpose");
  const matrix::FpSymmetryGroup<3, 2, 3, 3> ternary;
  EXPECT_DEATH(
      ternary.query.DecodeChecked(
          static_cast<uint32_t>(ternary.query.Identity()) | (1u << 31)),
      "transpose");
}

TEST(VerifierDeathTest, RejectsMalformedCertificateBeforeCheckingProof) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  const Problem::SymmetryGroup group;
  const RankMap<Problem> map;
  const BacktrackingProofArchive archive;
  pb::ConstrainedTensor rt;
  rt.set_constraints(std::string(1, char(0x80)));
  EXPECT_DEATH(VerifyOne<Problem>(rt, group, map, archive),
               "outside the vector");
  rt.set_constraints(std::string(1, '\0'));
  EXPECT_DEATH(VerifyOne<Problem>(rt, group, map, archive),
               "dependent certificate");
  pb::DegenerateProof proof;
  proof.set_extra_constraint(16);
  EXPECT_DEATH(VerifyDegenerateProof<Problem>({}, 1, proof, group, map),
               "extra constraint code");
}

} // namespace
