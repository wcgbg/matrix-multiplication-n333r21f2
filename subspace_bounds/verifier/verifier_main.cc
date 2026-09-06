// Verify a certificate produced by rank_lower_bound_main.
//
// Usage:
//   bazel run --config=opt //subspace_bounds/verifier:verifier_main -- /abs/path/cert.pb.txt
//
// Reads a Certificate, re-checks every orbit's rank_lower_bound_proof against a
// freshly built OrbitMap for the chosen problem's symmetry group, and logs the
// proven lower bound for the unconstrained tensor. Backtracking proofs are
// replayed from the single archive (cert.btp) next to the certificate. Any
// failed check aborts via CHECK. Mirrors
// proof_verifier/rank_lower_bound_verifier_main.cc.

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
