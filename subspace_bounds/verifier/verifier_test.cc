#include "subspace_bounds/verifier/verifier.h"

#include <filesystem>
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
  auto [certificate, archive_path] =
      BuildVerifiedCertificate<Problem>(options);
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

// Archive regeneration: with the .btp gone (or one entry stale), running the
// DP with regenerate_backtracking_proofs re-derives exactly the missing
// traces at the recorded bounds; the bounds stay as they were and the
// verifier accepts the rebuilt archive.
TEST(VerifierTest, RegeneratesBacktrackingArchive) {
  using Problem = matrix::Problem<2, 2, 2, 2>;
  auto [certificate, archive_path] = BuildVerifiedCertificate<Problem>();
  ASSERT_GE(FirstBacktrackingOrbit(certificate), 0);
  const std::string output_path =
      std::string(testing::TempDir()) + "/" + Problem::Name() + ".pb.txt";
  ASSERT_EQ(GetBacktrackingProofArchivePath(output_path), archive_path);
  std::vector<int> bounds;
  for (const auto &rt : certificate.constrained_tensors()) {
    bounds.push_back(rt.rank_lower_bound());
  }

  ProcessOptions regen;
  regen.regenerate_backtracking_proofs = true;

  // Whole archive missing.
  std::filesystem::remove(archive_path);
  ProcessOrbits<Problem>(regen, output_path, &certificate);
  for (int i = 0; i < certificate.constrained_tensors_size(); ++i) {
    EXPECT_EQ(certificate.constrained_tensors(i).rank_lower_bound(), bounds[i]) << i;
  }
  EXPECT_EQ(VerifyRankLowerBound<Problem>(certificate, archive_path), bounds.back());

  // One stale entry.
  const int orbit = FirstBacktrackingOrbit(certificate);
  {
    BacktrackingProofArchive archive = BacktrackingProofArchive::Load(archive_path);
    archive.Clear(orbit);
    archive.Save(archive_path);
  }
  ProcessOrbits<Problem>(regen, output_path, &certificate);
  EXPECT_FALSE(BacktrackingProofArchive::Load(archive_path).Get(orbit).Empty());
  EXPECT_EQ(VerifyRankLowerBound<Problem>(certificate, archive_path), bounds.back());
}

} // namespace
