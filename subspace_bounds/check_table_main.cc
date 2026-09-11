// Expand a certificate in memory and check that its subspace-bound table is
// antitone: L(S') >= L(S) for every hyperplane S' of every subspace S. The
// profile enumerators rely on this (their capacity check is complete only if
// cap(S') <= cap(S) for S' inside S); the dynamic program guarantees it by
// construction (degenerate reduction), and this tool re-checks the table.
//
// Usage:
//   bazel run --config=opt //subspace_bounds:check_table_main --
//   /abs/path/cert.pb.txt
//
// Compiled for one problem (P, NA come from matrix/chosen_problem.h, like
// the profile enumerators). Prints the number of covering pairs (S', S)
// examined and the number of violations; exits with status 1 on any violation.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/bit_vec.h"
#include "core/constraints.h"
#include "core/gf.h"
#include "core/gf_vec.h"
#include "core/proto_io.h"
#include "matrix/chosen_problem.h"
#include "subspace_bounds/subspace_bounds.h"

DEFINE_bool(histogram, false,
            "print lower-bound counts by constraint dimension");

namespace {

using Problem = ::ChosenProblem;
constexpr int P = Problem::kP;
constexpr int NA = Problem::kNA;
static_assert(IntPow(P, NA) <= 65535, "a row must fit its uint16 base-P code");
using Vec = GFVec<P, NA>;
using Cons = Constraints<P, NA>;
using F = GF<P>;

using Rec = SubspaceBound<NA>;

Rec Canonical(const std::vector<Vec> &basis) {
  Cons c;
  for (const Vec &v : basis)
    c.push_back(v);
  const int rank = GaussJordanRREF<P, NA>(&c);
  c.erase(c.begin(), c.end() - rank);
  Rec r;
  r.dim = static_cast<uint8_t>(rank);
  for (int i = 0; i < rank; ++i)
    r.rows[i] = SubspaceCodeFromVec(c[i]);
  return r;
}

} // namespace

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);
  CHECK_EQ(argc, 2) << "Usage: " << argv[0] << " <cert.pb.txt>";

  auto recs = ExpandSubspaceBounds<Problem>(
      ReadProtoFromFile<pb::Certificate>(argv[1]));
  if (FLAGS_histogram) {
    std::map<int, std::map<int, size_t>> histogram;
    for (const auto &rec : recs)
      ++histogram[rec.dim][rec.lb];
    for (const auto &[dim, counts] : histogram) {
      std::printf("%d [", dim);
      const char *separator = "";
      for (const auto &[lb, count] : counts) {
        std::printf("%s(%d, %zu)", separator, lb, count);
        separator = ", ";
      }
      std::printf("]\n");
    }
  }
  std::sort(recs.begin(), recs.end());
  for (size_t i = 1; i < recs.size(); ++i)
    CHECK(recs[i - 1] < recs[i]) << "duplicate subspace in the table";
  LOG(INFO) << "Loaded " << recs.size() << " subspaces";

  auto lookup = [&](const Rec &key) -> int {
    auto it = std::lower_bound(recs.begin(), recs.end(), key);
    CHECK(it != recs.end() && !(key < *it))
        << "subspace missing from the table";
    return it->lb;
  };

  std::atomic<uint64_t> pairs{0}, violations{0};
  tbb::parallel_for(size_t{0}, recs.size(), [&](size_t idx) {
    const Rec &s = recs[idx];
    const int d = s.dim;
    if (d == 0)
      return;
    std::vector<Vec> basis;
    for (int i = 0; i < d; ++i)
      basis.push_back(SubspaceVecFromCode<P, NA>(s.rows[i]));
    // Hyperplanes of S = kernels of the nonzero functionals a on F_P^d up to
    // scalars: with j the first index where a_j != 0 (normalized to 1), the
    // kernel is spanned by b_i - a_i b_j for i != j.
    std::array<int, NA> a{};
    uint64_t local_pairs = 0, local_viol = 0;
    // Enumerate a in base P, skipping zero and non-normalized vectors.
    uint64_t total = 1;
    for (int i = 0; i < d; ++i)
      total *= P;
    for (uint64_t n = 1; n < total; ++n) {
      uint64_t m = n;
      for (int i = 0; i < d; ++i) {
        a[i] = static_cast<int>(m % P);
        m /= P;
      }
      int j = 0;
      while (a[j] == 0)
        ++j;
      if (a[j] != 1)
        continue;
      std::vector<Vec> sub;
      for (int i = 0; i < d; ++i) {
        if (i == j)
          continue;
        Vec v = basis[i];
        if (a[i] != 0) {
          // v = b_i - a_i b_j
          for (int c = 0; c < NA; ++c) {
            const int x = (v[c].value + (P - a[i]) * basis[j][c].value) % P;
            v.Set(c, F{static_cast<uint8_t>(x)});
          }
        }
        sub.push_back(v);
      }
      const Rec sk = Canonical(sub);
      CHECK_EQ(sk.dim, d - 1);
      ++local_pairs;
      if (lookup(sk) < s.lb)
        ++local_viol;
    }
    pairs += local_pairs;
    violations += local_viol;
  });

  LOG(INFO) << "Checked " << pairs.load()
            << " covering pairs (S' < S): " << violations.load()
            << " violations of L(S') >= L(S)";
  std::printf("antitone check: subspaces=%zu pairs=%llu violations=%llu\n",
              recs.size(), static_cast<unsigned long long>(pairs.load()),
              static_cast<unsigned long long>(violations.load()));
  return violations.load() == 0 ? 0 : 1;
}
