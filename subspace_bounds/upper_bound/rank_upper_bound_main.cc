// Run the flip-graph local search to fill in rank *upper* bounds for every
// constrained tensor in a certificate produced by orbit_enumerator_main.
//
// Usage:
//   bazel run --config=opt //subspace_bounds/upper_bound:rank_upper_bound_main --
//   /abs/path/orbits.pb.txt
//
// Reads a Certificate, reconstructs each ConstrainedTensor's 3-tensor (the
// chosen problem's base tensor with the orbit's constraints applied), and runs
// RankUpperBound (subspace_bounds/upper_bound/rank_upper_bound.h). The smallest rank found and
// its verified decomposition string are written back into the tensor's
// rank_upper_bound and rank_upper_bound_proof fields. The result is written to
// --output_path, defaulting to overwriting the input. Mirrors
// rank_lower_bound_main.cc.

#include <atomic>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>

#include "gflags/gflags.h"
#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/proto_io.h"
#include "core/tensor.h"
#include "matrix/chosen_problem.h"
#include "subspace_bounds/upper_bound/rank_upper_bound.h"

DEFINE_string(output_path, "",
              "Output path (.pb.txt / .pb). Defaults to the input path "
              "(overwrites the input)");
DEFINE_int64(path_limit, 100'000, "Flip/plus steps per search path");
DEFINE_int32(max_steps_at_a_rank, 1'000,
             "Steps to spend at a rank before forcing a Plus move");
DEFINE_int64(num_paths, 100, "Independent restarts per tensor");
DEFINE_bool(last_only, false, "Only process the last constrained tensor");
DEFINE_uint64(seed, std::mt19937_64::default_seed,
              "Base RNG seed (each path gets a deterministic offset)");
DEFINE_bool(skip_if_lower_bound_matches, false,
            "Skip a tensor if its rank_lower_bound == rank_upper_bound");

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
  constexpr std::size_t NA = Problem::kNA;
  constexpr std::size_t NB = Problem::kNB;
  constexpr std::size_t NC = Problem::kNC;
  constexpr int P = Problem::kP;
  // The flip-graph upper bound search is F₂-only (see
  // subspace_bounds/upper_bound/flip_scheme.h).
  static_assert(P == 2,
                "rank_upper_bound_main only supports the F_2 flip-graph "
                "search; q != 2 runs skip this stage entirely (run.py).");

  pb::Certificate certificate = ReadProtoFromFile<pb::Certificate>(path);
  CHECK_EQ(certificate.characteristic(), P);
  CHECK_EQ(certificate.extension_degree(), 1);
  CHECK_EQ(certificate.na(), static_cast<int>(NA));
  CHECK_EQ(certificate.nb(), static_cast<int>(NB));
  CHECK_EQ(certificate.nc(), static_cast<int>(NC));
  CHECK_EQ(certificate.problem_name(), Problem::Name())
      << "Certificate problem_name does not match the compiled-in problem";
  CHECK_GT(certificate.constrained_tensors_size(), 0);

  const Tensor<P, NA, NB, NC> base_tensor = Problem::MakeTensor();

  std::string output_path = FLAGS_output_path;
  if (output_path.empty()) {
    output_path = path; // overwrite the input
  }

  const int constrained_tensors_size = certificate.constrained_tensors_size();
  const int begin_idx = FLAGS_last_only ? constrained_tensors_size - 1 : 0;
  std::atomic<int> progress = 0;
  std::atomic<int> num_matched = 0;
  tbb::parallel_for(begin_idx, constrained_tensors_size, [&](int idx) {
    pb::ConstrainedTensor &rt = *certificate.mutable_constrained_tensors(idx);
    bool skip = FLAGS_skip_if_lower_bound_matches &&
                rt.has_rank_lower_bound() && rt.has_rank_upper_bound() &&
                rt.rank_lower_bound() == rt.rank_upper_bound();
    if (!skip) {
      const Constraints<P, NA> constraints =
          ConstraintsFromBytes<P, NA>(rt.constraints());
      const Tensor<P, NA, NB, NC> tensor =
          ApplyConstraintsToTensor<P, NA, NB, NC>(constraints, base_tensor);

      const upper_bound::UpperBoundResult<NA, NB, NC> result =
          upper_bound::RankUpperBound<NA, NB, NC>(
              tensor, {
                          .seed = FLAGS_seed,
                          .path_limit = FLAGS_path_limit,
                          .max_steps_at_a_rank = FLAGS_max_steps_at_a_rank,
                          .num_paths = FLAGS_num_paths,
                      });

      if (!rt.has_rank_upper_bound() || result.rank < rt.rank_upper_bound()) {
        rt.set_rank_upper_bound(result.rank);
        rt.set_rank_upper_bound_proof(result.scheme);
      }
    }
    if (rt.has_rank_lower_bound()) {
      CHECK_LE(rt.rank_lower_bound(), rt.rank_upper_bound()) << rt.index();
    }
    if (rt.rank_lower_bound() == rt.rank_upper_bound()) {
      num_matched.fetch_add(1);
    }
    std::cerr << "  " << progress.fetch_add(1) << "/"
              << (constrained_tensors_size - begin_idx) << "    \r";
  });
  std::cerr << std::endl;

  LOG(INFO) << "#(LB==UB): " << num_matched << " / "
            << (constrained_tensors_size - begin_idx);

  CHECK_GT(certificate.constrained_tensors_size(), 0);
  const pb::ConstrainedTensor &last_rt = certificate.constrained_tensors(
      certificate.constrained_tensors_size() - 1);
  CHECK(last_rt.constraints().empty());
  LOG(INFO) << "UNCONSTRAINED TENSOR RANK UPPER BOUND: "
            << last_rt.rank_upper_bound();

  WriteProtoToFile(certificate, output_path);
  LOG(INFO) << "Done. Wrote " << output_path;
  return 0;
}
