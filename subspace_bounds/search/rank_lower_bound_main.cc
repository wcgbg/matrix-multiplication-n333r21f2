// Run the dynamic-programming search that improves the rank lower bounds in a
// certificate produced by orbit_enumerator_main.
//
// Usage:
//   bazel run --config=opt //subspace_bounds/search:rank_lower_bound_main --
//   /abs/path/orbits.pb.txt
//
// Reads a Certificate, builds an OrbitMap for the chosen problem's symmetry
// group, and runs ProcessOrbits. The result (with improved rank_lower_bound and
// rank_lower_bound_proof fields) is written to --output_path, defaulting to a
// "_updated" sibling of the input. Mirrors
// rank_search/rank_lower_bound_computer_main.cc.

#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "ng-log/logging.h"

#include "core/certificate.pb.h"
#include "core/proto_io.h"
#include "matrix/chosen_problem.h"
#include "subspace_bounds/search/rank_lower_bound_computer.h"

DEFINE_string(output_path, "",
              "Output path (.pb.txt / .pb). Defaults to the input path "
              "(overwrites the input)");
DEFINE_bool(ignore_rank_lower_bound, false,
            "Clear all existing rank lower bounds before searching");
DEFINE_bool(
    ignore_non_basic_rank_lower_bound, false,
    "Clear rank lower bounds whose proof is not self-contained (i.e. neither "
    "flatten_matrix_proof, forced_product_proof, nor rank_one_span_proof) "
    "before searching");
DEFINE_uint64(backtracking_step_limit, std::numeric_limits<uint64_t>::max(),
              "Max backtracking search steps (all threads) per orbit; 0 to "
              "disable backtracking");
DEFINE_uint64(backtracking_max_map_size, 10'000'000,
              "Max size of the per-thread backtracking cache before stochastic "
              "halving");
DEFINE_bool(basic_method, true, "Enable the Flatten + Forced Product methods");
DEFINE_bool(degenerate_method, true, "Enable the Degenerate Reduction method");
DEFINE_int32(dim_min, 0, "Min subspace dimension to process (0 to NA)");
DEFINE_int32(dim_max, std::numeric_limits<int32_t>::max(),
             "Max subspace dimension to process (0 to NA)");
DEFINE_int32(forced_product_max_iterations_log2, 24,
             "Skip the Forced Product technique when its enumeration size "
             "num_iterations exceeds 2^this; larger enumerates more "
             "combinations (slower, potentially stronger bound)");
DEFINE_uint64(rank1span_max_subspaces, 10'000'000,
              "Operation budget for the rank-one-span exclusion technique "
              "(subspace enumeration or family search); the technique is "
              "skipped on an orbit when its up-front cost estimate exceeds "
              "this; 0 disables it. Prime fields only (the family search "
              "engine is F2-only; other primes use the exhaustive engine)");
DEFINE_string(recompute_orbits, "",
              "Comma-separated orbit `index` values to recompute from scratch: "
              "their rank lower bounds and proofs are cleared, then only these "
              "orbits are processed (within --dim_min/--dim_max) while every "
              "other orbit keeps its bound and proof and only seeds the orbit "
              "map; the archive slots of the listed orbits are rewritten. For "
              "per-orbit reruns and ablations (e.g. with a technique disabled)");
DEFINE_bool(regenerate_backtracking_proofs, false,
            "Archive regeneration: re-derive only the backtracking traces of "
            "orbits whose .btp entry is missing or does not match the "
            "certificate's proof_size, at the recorded bound (target = lb, "
            "never lb + 1); no other technique runs and no bound is searched "
            "for. Rewrites proof_size; the bound itself changes only if the "
            "search happens to prove more");

namespace {

using Problem = ChosenProblem;

// The certificate at `path`, checked against the compiled-in problem.
pb::Certificate LoadCertificateForChosenProblem(const std::string &path) {
  pb::Certificate certificate = ReadProtoFromFile<pb::Certificate>(path);
  CHECK_EQ(certificate.characteristic(), Problem::kP);
  CHECK_EQ(certificate.extension_degree(), 1);
  CHECK_EQ(certificate.na(), Problem::kNA);
  CHECK_EQ(certificate.nb(), Problem::kNB);
  CHECK_EQ(certificate.nc(), Problem::kNC);
  CHECK_EQ(certificate.problem_name(), Problem::Name())
      << "Certificate problem_name does not match the compiled-in problem";
  CHECK_GT(certificate.constrained_tensors_size(), 0);
  return certificate;
}

// --ignore_rank_lower_bound: every bound and proof goes, and so does the
// backtracking-proof archive next to the input.
void ClearAllLowerBounds(pb::Certificate *certificate,
                         const std::string &input_path) {
  for (pb::ConstrainedTensor &rt :
       *certificate->mutable_constrained_tensors()) {
    rt.clear_rank_lower_bound();
    rt.clear_rank_lower_bound_proof();
  }
  const std::string archive_path =
      GetBacktrackingProofArchivePath(input_path);
  if (std::filesystem::exists(archive_path)) {
    std::filesystem::remove(archive_path);
  }
}

// --ignore_non_basic_rank_lower_bound: a bound whose proof is not
// self-contained is reset to 0 rather than cleared, so the orbit still has a
// recorded bound and flatten / forced product do not run on it again.
void ClearNonBasicLowerBounds(pb::Certificate *certificate) {
  for (pb::ConstrainedTensor &rt :
       *certificate->mutable_constrained_tensors()) {
    const pb::RankLowerBoundProof &proof = rt.rank_lower_bound_proof();
    if (!proof.has_flatten_matrix_proof() &&
        !proof.has_forced_product_proof() &&
        !proof.has_rank_one_span_proof()) {
      rt.set_rank_lower_bound(0);
      rt.clear_rank_lower_bound_proof();
    }
  }
}

// Comma-separated integers.
std::vector<int> ParseIntList(const std::string &csv) {
  std::vector<int> values;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    values.push_back(std::stoi(item));
  }
  return values;
}

// --recompute_orbits: the listed orbits lose their bound and proof; every
// listed index must exist.
void ClearOrbitsForRecompute(pb::Certificate *certificate,
                             const std::vector<int> &indices) {
  for (int index : indices) {
    bool found = false;
    for (pb::ConstrainedTensor &rt :
         *certificate->mutable_constrained_tensors()) {
      if (static_cast<int>(rt.index()) == index) {
        LOG(INFO) << "Recomputing orbit index=" << index << " from scratch"
                  << " (previous bound "
                  << (rt.has_rank_lower_bound()
                          ? std::to_string(rt.rank_lower_bound())
                          : std::string("none"))
                  << ")";
        rt.clear_rank_lower_bound();
        rt.clear_rank_lower_bound_proof();
        found = true;
      }
    }
    CHECK(found) << "--recompute_orbits: no orbit with index " << index;
  }
}

ProcessOptions OptionsFromFlags(std::vector<int> only_orbits) {
  return {
      .basic_method = FLAGS_basic_method,
      .degenerate_method = FLAGS_degenerate_method,
      .backtracking_step_limit = FLAGS_backtracking_step_limit,
      .backtracking_max_map_size = FLAGS_backtracking_max_map_size,
      .dim_min = FLAGS_dim_min,
      .dim_max = FLAGS_dim_max,
      .forced_product_max_iterations_log2 =
          FLAGS_forced_product_max_iterations_log2,
      .rank1span_max_subspaces = FLAGS_rank1span_max_subspaces,
      .regenerate_backtracking_proofs = FLAGS_regenerate_backtracking_proofs,
      .only_orbits = std::move(only_orbits),
  };
}

// The last orbit is the unconstrained tensor; its bound is THE bound.
void LogUnconstrainedBound(const pb::Certificate &certificate) {
  CHECK_GT(certificate.constrained_tensors_size(), 0);
  const pb::ConstrainedTensor &last_rt = certificate.constrained_tensors(
      certificate.constrained_tensors_size() - 1);
  CHECK(last_rt.constraints().empty());
  LOG(INFO) << "UNCONSTRAINED TENSOR RANK LOWER BOUND: "
            << last_rt.rank_lower_bound();
}

} // namespace

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);

  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " <certificate_path>";
  }
  const std::string path = argv[1];

  pb::Certificate certificate = LoadCertificateForChosenProblem(path);
  if (FLAGS_ignore_rank_lower_bound) {
    ClearAllLowerBounds(&certificate, path);
  }
  if (FLAGS_ignore_non_basic_rank_lower_bound) {
    ClearNonBasicLowerBounds(&certificate);
  }
  std::vector<int> recompute_orbits;
  if (!FLAGS_recompute_orbits.empty()) {
    recompute_orbits = ParseIntList(FLAGS_recompute_orbits);
    ClearOrbitsForRecompute(&certificate, recompute_orbits);
  }

  std::string output_path = FLAGS_output_path;
  if (output_path.empty()) {
    output_path = path; // overwrite the input
  }

  ProcessOrbits<Problem>(OptionsFromFlags(recompute_orbits), output_path,
                         &certificate);
  LogUnconstrainedBound(certificate);

  LOG(INFO) << "Done. Wrote " << output_path;
  return 0;
}
