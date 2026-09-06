#include <format>
#include <string>

#include "gflags/gflags.h"
#include "ng-log/logging.h"

#include "core/certificate.pb.h"
#include "core/proto_io.h"
#include "matrix/chosen_problem.h"
#include "subspace_bounds/search/orbit_enumerator.h"
#include "subspace_bounds/search/orbit_enumerator_slow.h"

DEFINE_bool(slow, false, "Run the simple but slow reference enumerator");
DEFINE_string(output_path, "",
              "Output path (.pb.txt for text, .pb for binary). Defaults to "
              "orbits_<problem>.pb.txt");
DEFINE_bool(verbose, false,
            "Also write verbose fields (constraints_text, tensor_text) to the "
            "certificate (larger output)");

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);

  using Problem = ChosenProblem;
  const typename Problem::SymmetryGroup group;
  LOG(INFO) << "group.query.size = " << group.query.Size();
  LOG(INFO) << "group.store.size = " << group.store.Size();

  pb::Certificate certificate;
  if (FLAGS_slow) {
    const OrbitEnumeratorSlow<Problem> enumerator(&group);
    certificate = enumerator.Search(FLAGS_verbose);
  } else {
    const OrbitEnumerator<Problem> enumerator(&group);
    certificate = enumerator.Search(FLAGS_verbose);
  }

  std::string output_path = FLAGS_output_path;
  if (output_path.empty()) {
    output_path = std::format("cert_{}.pb.txt", Problem::Name());
  }
  WriteProtoToFile(certificate, output_path);

  return 0;
}
