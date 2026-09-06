// Check that a subspace-bound table (the output of subspace_bounds_main) is
// antitone: L(S') >= L(S) for every hyperplane S' of every subspace S. The
// profile enumerators rely on this (their capacity check is complete only if
// cap(S') <= cap(S) for S' inside S); the dynamic program guarantees it by
// construction (degenerate reduction), and this tool re-checks the table.
//
// Usage:
//   bazel run --config=opt //subspace_bounds:check_table_main -- /abs/path/table.bin
//
// Compiled for one problem (P, NA come from matrix/chosen_problem.h, like
// subspace_bounds_main). Prints the number of covering pairs (S', S) examined
// and the number of violations; exits with status 1 on any violation.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/bit_vec.h"
#include "core/constraints.h"
#include "core/gf.h"
#include "core/gf_vec.h"
#include "matrix/chosen_problem.h"

namespace {

using Problem = ::ChosenProblem;
constexpr int P = Problem::kP;
constexpr int NA = Problem::kNA;
static_assert(IntPow(P, NA) <= 65535, "a row must fit its uint16 base-P code");
using Vec = GFVec<P, NA>;
using Cons = Constraints<P, NA>;
using F = GF<P>;

template <int PP = P> GFVec<PP, NA> VecFromCode(uint32_t code) {
  if constexpr (PP == 2) {
    return GFVec<PP, NA>{static_cast<BitVec<NA>>(code)};
  } else {
    GFVec<PP, NA> v{};
    for (int c = 0; c < NA; ++c) {
      v.Set(c, F{static_cast<uint8_t>(code % P)});
      code /= P;
    }
    return v;
  }
}

template <int PP> uint16_t CodeFromVec(const GFVec<PP, NA> &v) {
  if constexpr (PP == 2) {
    return static_cast<uint16_t>(v.data);
  } else {
    uint32_t code = 0;
    for (int c = NA - 1; c >= 0; --c) code = code * P + v[c].value;
    return static_cast<uint16_t>(code);
  }
}

// A table record: the canonical rows of S (as written by subspace_bounds_main)
// and L(S). Records are sorted by (dim, rows) for binary-search lookups.
struct Rec {
  uint8_t dim = 0;
  std::array<uint16_t, NA> rows{};
  uint8_t lb = 0;
  bool operator<(const Rec &o) const {
    if (dim != o.dim) return dim < o.dim;
    return rows < o.rows;
  }
};

Rec Canonical(const std::vector<Vec> &basis) {
  Cons c;
  for (const Vec &v : basis) c.push_back(v);
  const int rank = GaussJordanRREF<P, NA>(&c);
  c.erase(c.begin(), c.end() - rank);
  Rec r;
  r.dim = static_cast<uint8_t>(rank);
  for (int i = 0; i < rank; ++i) r.rows[i] = CodeFromVec(c[i]);
  return r;
}

} // namespace

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);
  CHECK_EQ(argc, 2) << "Usage: " << argv[0] << " <table.bin>";

  std::vector<Rec> recs;
  {
    FILE *f = std::fopen(argv[1], "rb");
    CHECK(f != nullptr) << "cannot open " << argv[1];
    for (;;) {
      Rec r;
      if (std::fread(&r.dim, 1, 1, f) != 1) break;
      CHECK_LE(r.dim, NA);
      CHECK_EQ(std::fread(r.rows.data(), sizeof(uint16_t), r.dim, f), r.dim);
      CHECK_EQ(std::fread(&r.lb, 1, 1, f), 1u);
      // Re-canonicalise defensively (the stored rows should already be
      // canonical); a mismatch would indicate a convention drift.
      std::vector<Vec> basis;
      for (int i = 0; i < r.dim; ++i) basis.push_back(VecFromCode(r.rows[i]));
      Rec canon = Canonical(basis);
      CHECK_EQ(canon.dim, r.dim) << "stored rows are dependent";
      CHECK(canon.rows == r.rows) << "stored rows are not canonical";
      recs.push_back(r);
    }
    std::fclose(f);
  }
  std::sort(recs.begin(), recs.end());
  for (size_t i = 1; i < recs.size(); ++i)
    CHECK(recs[i - 1] < recs[i]) << "duplicate subspace in the table";
  LOG(INFO) << "Loaded " << recs.size() << " subspaces";

  auto lookup = [&](const Rec &key) -> int {
    auto it = std::lower_bound(recs.begin(), recs.end(), key);
    CHECK(it != recs.end() && !(key < *it)) << "subspace missing from the table";
    return it->lb;
  };

  std::atomic<uint64_t> pairs{0}, violations{0};
  tbb::parallel_for(size_t{0}, recs.size(), [&](size_t idx) {
    const Rec &s = recs[idx];
    const int d = s.dim;
    if (d == 0) return;
    std::vector<Vec> basis;
    for (int i = 0; i < d; ++i) basis.push_back(VecFromCode(s.rows[i]));
    // Hyperplanes of S = kernels of the nonzero functionals a on F_P^d up to
    // scalars: with j the first index where a_j != 0 (normalised to 1), the
    // kernel is spanned by b_i - a_i b_j for i != j.
    std::array<int, NA> a{};
    uint64_t local_pairs = 0, local_viol = 0;
    // Enumerate a in base P, skipping zero and non-normalised vectors.
    uint64_t total = 1;
    for (int i = 0; i < d; ++i) total *= P;
    for (uint64_t n = 1; n < total; ++n) {
      uint64_t m = n;
      for (int i = 0; i < d; ++i) {
        a[i] = static_cast<int>(m % P);
        m /= P;
      }
      int j = 0;
      while (a[j] == 0) ++j;
      if (a[j] != 1) continue;
      std::vector<Vec> sub;
      for (int i = 0; i < d; ++i) {
        if (i == j) continue;
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
      if (lookup(sk) < s.lb) ++local_viol;
    }
    pairs += local_pairs;
    violations += local_viol;
  });

  LOG(INFO) << "Checked " << pairs.load() << " covering pairs (S' < S): "
            << violations.load() << " violations of L(S') >= L(S)";
  std::printf("antitone check: subspaces=%zu pairs=%llu violations=%llu\n",
              recs.size(), static_cast<unsigned long long>(pairs.load()),
              static_cast<unsigned long long>(violations.load()));
  return violations.load() == 0 ? 0 : 1;
}
