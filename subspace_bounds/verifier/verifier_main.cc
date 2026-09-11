// Verify a certificate produced by rank_lower_bound_main.
//
// Usage:
//   bazel run --config=opt //subspace_bounds/verifier:verifier_main --
//   /abs/path/cert.pb.txt
//
// Reads a Certificate and verifies each orbit's proof using a RankMap of
// previously verified canonical representatives. Degenerate and backtracking
// witnesses recover representatives directly through the symmetry actions.
// Backtracking traces are replayed from the archive (cert.btp) next to the
// certificate. Logs the proven bound for the unconstrained tensor; failed
// checks abort via CHECK.

#include <string>

#include "gflags/gflags.h"
#include "ng-log/logging.h"

#include "core/backtracking_proof.h"
#include "core/certificate.pb.h"
#include "core/proto_io.h"
#include "matrix/chosen_problem.h"
#include "subspace_bounds/verifier/verifier.h"

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);

  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " <certificate_path>";
  }
  const std::string path = argv[1];

  using Problem = ChosenProblem;

  const pb::Certificate certificate = ReadProtoFromFile<pb::Certificate>(path);
  const std::string archive_path = GetBacktrackingProofArchivePath(path);

  const int result = VerifyRankLowerBound<Problem>(certificate, archive_path);

  LOG(INFO) << "UNCONSTRAINED TENSOR RANK LOWER BOUND: " << result;
  LOG(INFO) << "OK. Verified " << path;
  return 0;
}
