#include "core/proto_io.h"

#include <cstdlib>
#include <string>

#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"

#include "core/certificate.pb.h"

namespace {

pb::Certificate MakeCertificate() {
  pb::Certificate cert;
  cert.set_problem_name("proto_io_test");
  cert.set_characteristic(2);
  cert.set_extension_degree(4);
  return cert;
}

std::string TempPath(const std::string &suffix) {
  return std::string(testing::TempDir()) + "/proto_io_test_" + suffix;
}

void ExpectRoundTrip(const std::string &suffix) {
  const std::string path = TempPath(suffix);
  const pb::Certificate original = MakeCertificate();
  WriteProtoToFile(original, path);
  const pb::Certificate read = ReadProtoFromFile<pb::Certificate>(path);
  EXPECT_EQ(read.problem_name(), original.problem_name());
  EXPECT_EQ(read.characteristic(), original.characteristic());
  EXPECT_EQ(read.extension_degree(), original.extension_degree());
  std::filesystem::remove(path);
}

TEST(ProtoIoTest, TextRoundTrip) { ExpectRoundTrip("a.pb.txt"); }

TEST(ProtoIoTest, BinaryRoundTrip) { ExpectRoundTrip("a.pb"); }

// Reading a real gzipped golden must yield the same message as its plaintext
// sibling.
TEST(ProtoIoTest, GzipTextGoldenMatchesPlain) {
  const std::string dir =
      "subspace_bounds/search/testdata/orbit_enumerator/matrix_q02_n222";
  const auto plain = ReadProtoFromFile<pb::Certificate>(dir + ".pb.txt");
  const auto gzipped = ReadProtoFromFile<pb::Certificate>(dir + ".pb.txt.gz");

  google::protobuf::util::MessageDifferencer diff;
  std::string report;
  diff.ReportDifferencesToString(&report);
  EXPECT_TRUE(diff.Compare(plain, gzipped)) << report;
}

} // namespace
