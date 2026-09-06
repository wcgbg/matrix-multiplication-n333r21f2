// Generates golden certificates for the tests that pin the
// OrbitEnumeratorSlow and ProcessOrbits implementations. For each matrix format
// exercised by the unit tests it writes four certificates:
//   <output_dir>/orbit_enumerator/<problem>.pb.txt
//   <output_dir>/rank_lower_bound_basic/<problem>.pb.txt
//   <output_dir>/rank_lower_bound_basic_degenerate/<problem>.pb.txt
//   <output_dir>/rank_lower_bound_all/<problem>.pb.txt
// The three rank-lower-bound runs are chained: each ProcessOrbits call mutates
// the certificate the next one starts from. Run once when the algorithm
// changes to refresh the checked-in goldens.

#include <filesystem>
#include <format>
#include <string>

#include "gflags/gflags.h"
#include "ng-log/logging.h"

#include "core/certificate.pb.h"
#include "core/proto_io.h"
#include "matrix/problem.h"
#include "subspace_bounds/search/orbit_enumerator_slow.h"
#include "subspace_bounds/search/rank_lower_bound_computer.h"

DEFINE_string(output_dir, "subspace_bounds/search/testdata",
              "Parent directory; the binary writes into four subdirectories: "
              "orbit_enumerator/, rank_lower_bound_basic/, "
              "rank_lower_bound_basic_degenerate/, rank_lower_bound_all/");

namespace {

template <class Problem>
void GenerateOne(const std::string &output_dir,
                 bool orbit_enumerator_only = false) {
  const typename Problem::SymmetryGroup group;
  const OrbitEnumeratorSlow<Problem> enumerator(&group);
  pb::Certificate cert = enumerator.Search(/*fill_verbose_fields=*/false);
  WriteProtoToFile(cert, std::format("{}/orbit_enumerator/{}.pb.txt",
                                     output_dir, Problem::Name()));

  if (orbit_enumerator_only) {
    return;
  }

  ProcessOrbits<Problem>({.basic_method = true,
                          .degenerate_method = false,
                          .backtracking_step_limit = 0},
                         /*output_path=*/"", &cert);
  WriteProtoToFile(cert, std::format("{}/rank_lower_bound_basic/{}.pb.txt",
                                     output_dir, Problem::Name()));

  ProcessOrbits<Problem>({.basic_method = true,
                          .degenerate_method = true,
                          .backtracking_step_limit = 0},
                         /*output_path=*/"", &cert);
  WriteProtoToFile(cert,
                   std::format("{}/rank_lower_bound_basic_degenerate/{}.pb.txt",
                               output_dir, Problem::Name()));

  ProcessOrbits<Problem>({.basic_method = true,
                          .degenerate_method = true,
                          .backtracking_step_limit = 100'000},
                         /*output_path=*/"", &cert);
  WriteProtoToFile(cert, std::format("{}/rank_lower_bound_all/{}.pb.txt",
                                     output_dir, Problem::Name()));
}

} // namespace

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);

  const std::string &dir = FLAGS_output_dir;
  for (const char *subdir :
       {"orbit_enumerator", "rank_lower_bound_basic",
        "rank_lower_bound_basic_degenerate", "rank_lower_bound_all"}) {
    std::filesystem::create_directories(std::format("{}/{}", dir, subdir));
  }

  // The last two formats are enumerator-only: their DP goldens would be large
  // and slow to regenerate.
  GenerateOne<matrix::Problem<2, 1, 2, 3>>(dir);
  GenerateOne<matrix::Problem<2, 1, 3, 2>>(dir);
  GenerateOne<matrix::Problem<2, 2, 1, 3>>(dir);
  GenerateOne<matrix::Problem<2, 2, 2, 2>>(dir);
  GenerateOne<matrix::Problem<2, 2, 2, 3>>(dir);
  GenerateOne<matrix::Problem<2, 2, 3, 1>>(dir);
  GenerateOne<matrix::Problem<2, 2, 3, 2>>(dir);
  GenerateOne<matrix::Problem<2, 2, 3, 3>>(dir);
  GenerateOne<matrix::Problem<2, 3, 1, 2>>(dir);
  GenerateOne<matrix::Problem<2, 3, 2, 1>>(dir);
  GenerateOne<matrix::Problem<2, 3, 2, 2>>(dir);
  GenerateOne<matrix::Problem<2, 3, 2, 3>>(dir);
  GenerateOne<matrix::Problem<2, 3, 3, 2>>(dir, true);
  GenerateOne<matrix::Problem<2, 3, 3, 3>>(dir, true);

  return 0;
}
