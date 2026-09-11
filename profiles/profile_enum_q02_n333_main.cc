// CLI for the F_2 <3,3,3> capacity-constrained profile enumerator.
// Search and correctness commentary live in profiles/q02_n333/.
#include "core/proto_io.h"
#include "gflags/gflags.h"
#include "matrix/chosen_problem.h"
#include "ng-log/logging.h"
#include "profiles/q02_n333/run.h"

#include <algorithm>
#include <thread>

DEFINE_string(certificate, "", "certificate containing per-orbit lower bounds");
DEFINE_int32(min_w, 0, "smallest |W| (rank->=2 factors) to consider");
DEFINE_int32(max_w, 8, "largest |W| to consider");
DEFINE_uint64(budget, 100'000'000, "DFS node budget per job");
DEFINE_string(cases, "",
              "comma-separated outer case indices to run (default all)");
DEFINE_bool(list_only, false, "only enumerate and print the outer cases");
DEFINE_int32(max_solutions, 50, "stop a job after this many profiles");
DEFINE_int32(log_seconds, 10, "progress log interval");
DEFINE_int32(r, 20, "number of terms; capacities are r - L(S)");
DEFINE_bool(prop3, false,
            "use the excess lemma (needs r = 20; not needed for the result): "
            "contextual caps m(B_x) <= 7, "
            "m(S_{x,c}) <= 12 and the filter 2 n2 + 3 n3 >= 11");
DEFINE_string(check_list, "",
              "comma-separated matrices (9-bit ints): only check this list "
              "against every capacity and report the first violation");
DEFINE_bool(symmetry, true,
            "orderly generation: keep only rank-one sets that are lex-minimal "
            "in their orbit under the stabilizer of W");
DEFINE_string(
    ones_subset, "",
    "testing: restrict the rank-one candidates to this comma-separated "
    "subset (the symmetry group is restricted to its setwise stabilizer)");
DEFINE_int32(
    split_depth, 4,
    "split each outer case into jobs by its first split_depth rank-one "
    "choices (0 = one job per case) so one case can use every hardware "
    "thread");

DEFINE_bool(
    experimental, false,
    "allow restricted/non-paper enumeration; no full classification claim");

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);
  CHECK_EQ(ChosenProblem::Name(), "matrix_q02_n333")
      << "this enumerator requires F_2 <3,3,3>";
  CHECK(!FLAGS_certificate.empty()) << "--certificate is required";
  LOG(INFO) << "r=" << FLAGS_r << " prop3=" << (FLAGS_prop3 ? "on" : "off")
            << " |W| in [" << FLAGS_min_w << "," << FLAGS_max_w << "]";

  profiles::q02_n333::Options options;
  options.r = FLAGS_r;
  options.experimental = FLAGS_experimental;
  options.budget = FLAGS_budget;
  options.max_solutions = FLAGS_max_solutions;
  options.log_seconds = FLAGS_log_seconds;
  options.symmetry = FLAGS_symmetry;
  options.threads = std::max(1u, std::thread::hardware_concurrency());
  options.split_depth = FLAGS_split_depth;
  options.min_w = FLAGS_min_w;
  options.max_w = FLAGS_max_w;
  options.prop3 = FLAGS_prop3;
  options.list_only = FLAGS_list_only;
  if (FLAGS_check_list.empty() && !FLAGS_list_only)
    options.cases = profiles::ParseList<size_t>(FLAGS_cases);
  options.check_list = profiles::ParseList(FLAGS_check_list);
  if (!FLAGS_list_only || !FLAGS_check_list.empty())
    options.ones_subset = profiles::ParseList(FLAGS_ones_subset);
  profiles::q02_n333::ValidateBinaryOptions(options);
  return profiles::q02_n333::Run(
      ReadProtoFromFile<pb::Certificate>(FLAGS_certificate), options);
}
