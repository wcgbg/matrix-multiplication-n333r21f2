// CLI for the generic capacity-constrained profile enumerator.
// Compile-time problem selection stays in this translation unit.
#include "core/proto_io.h"
#include "gflags/gflags.h"
#include "matrix/chosen_problem.h"
#include "ng-log/logging.h"
#include "profiles/fp/run.h"

#include <algorithm>
#include <thread>

DEFINE_string(certificate, "", "certificate containing per-orbit lower bounds");
DEFINE_int32(r, 14, "number of terms; capacities are r - L(S)");
DEFINE_uint64(budget, 1'000'000'000, "DFS node budget per job");
DEFINE_int32(max_solutions, 100, "stop after this many canonical profiles");
DEFINE_int32(log_seconds, 30, "progress log interval");
DEFINE_bool(symmetry, true, "orderly generation under G");
DEFINE_int32(split_depth, 0, "split the search into jobs by the first choices");
DEFINE_string(check_list, "",
              "comma-separated factor codes: check this list only");
DEFINE_string(ones_subset, "",
              "testing: restrict candidates to these point codes");

// Keep unsupported selections buildable for bazel build //...; fail before
// loading a certificate. Only instantiate the supported search engines.
template <class Problem>
int RunSelected(const std::string &path, const profiles::Options &options) {
  constexpr int p = Problem::kP, n0 = Problem::kN0, n1 = Problem::kN1;
  if constexpr ((p == 3 && n0 == 2 && n1 == 3) ||
                (p == 2 &&
                 ((n0 == 2 && (n1 == 2 || n1 == 3)) || (n0 == 3 && n1 == 2)))) {
    return profiles::fp::Run<Problem>(ReadProtoFromFile<pb::Certificate>(path),
                                      options);
  } else {
    LOG(FATAL) << "unsupported profile problem: " << Problem::Name();
  }
}

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);
  CHECK(!FLAGS_certificate.empty()) << "--certificate is required";
  profiles::Options options;
  options.r = FLAGS_r;
  options.budget = FLAGS_budget;
  options.max_solutions = FLAGS_max_solutions;
  options.log_seconds = FLAGS_log_seconds;
  options.symmetry = FLAGS_symmetry;
  options.threads = std::max(1u, std::thread::hardware_concurrency());
  options.split_depth = FLAGS_split_depth;
  options.check_list = profiles::ParseList(FLAGS_check_list);
  options.ones_subset = profiles::ParseList(FLAGS_ones_subset);
  profiles::ValidateOptions(options);
  profiles::ValidateCodes(options.check_list,
                          IntPow(ChosenProblem::kP, ChosenProblem::kNA));
  profiles::ValidateCodes(options.ones_subset,
                          IntPow(ChosenProblem::kP, ChosenProblem::kNA));
  return RunSelected<ChosenProblem>(FLAGS_certificate, options);
}
