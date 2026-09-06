// Expand a certificate's per-orbit rank lower bounds to EVERY subspace of the
// first-factor space A* = F_2^NA (all constraint sets, not just the canonical
// orbit representatives).
//
// Usage:
//   bazel run --config=opt //subspace_bounds:subspace_bounds_main -- \
//       /abs/path/cert_matrix_q02_n333.pb.txt /abs/path/out.bin
//
// Output (binary, little-endian): for every subspace S of F_2^NA, in
// enumeration order,
//   uint8  dim(S)
//   uint16 rows[dim(S)]   basis of S in the framework's column-reversed RREF
//   uint8  L(S)           certified rank lower bound of the orbit of S
// Here L(S) is the certified lower bound R(T / S) for the tensor with its first
// factor quotiented by S, i.e. the bound after restricting the first argument
// to the annihilator of S. Capacity: a rank-r decomposition has at most
// r - L(S) terms whose first factor lies in S.

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/certificate.pb.h"
#include "core/bit_vec.h"
#include "core/constraints.h"
#include "core/gf.h"
#include "core/gf_vec.h"
#include "core/proto_io.h"
#include "matrix/chosen_problem.h"
#include "subspace_bounds/search/orbit_map.h"

namespace {

using Problem = ::ChosenProblem;
constexpr int P = Problem::kP;
constexpr int NA = Problem::kNA;
static_assert(IntPow(P, NA) <= 65535, "a row must fit its uint16 base-P code");
using Vec = GFVec<P, NA>;
using Cons = Constraints<P, NA>;
using F = GF<P>;

// A row of (F_P^NA)* is stored as the base-P code sum_c digit_c * P^c (for
// P = 2 this is the plain bit pattern, so F_2 tables are unchanged).
// (Templates so that `if constexpr` discards the branch for the other field.)
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

struct Subspace {
  uint8_t dim = 0;
  std::array<uint16_t, NA> rows{};
  uint8_t lb = 0;
};

// Enumerate all subspaces of F_P^NA as standard RREFs: pivot columns
// p_0 < ... < p_{d-1}, pivot entries 1, zeros left of each pivot and in the
// other rows' pivot columns, arbitrary digits elsewhere. (The exact convention
// is irrelevant: the rows are re-canonicalised by GaussJordanRREF before the
// orbit lookup.)
void Enumerate(std::vector<Subspace> *out) {
  uint32_t pow[NA + 1];
  pow[0] = 1;
  for (int c = 1; c <= NA; ++c) pow[c] = pow[c - 1] * P;
  for (int d = 0; d <= NA; ++d) {
    for (uint32_t pivots = 0; pivots < (1u << NA); ++pivots) {
      if (__builtin_popcount(pivots) != d) continue;
      std::vector<int> piv;
      for (int c = 0; c < NA; ++c)
        if (pivots >> c & 1) piv.push_back(c);
      std::vector<std::pair<int, int>> free_pos; // (row, col)
      for (int i = 0; i < d; ++i)
        for (int c = piv[i] + 1; c < NA; ++c)
          if (!(pivots >> c & 1)) free_pos.push_back({i, c});
      const int nfree = free_pos.size();
      uint64_t total = 1;
      for (int f = 0; f < nfree; ++f) total *= P;
      std::vector<int> digit(nfree, 0);
      for (uint64_t n = 0; n < total; ++n) {
        Subspace s;
        s.dim = static_cast<uint8_t>(d);
        for (int i = 0; i < d; ++i) s.rows[i] = static_cast<uint16_t>(pow[piv[i]]);
        for (int f = 0; f < nfree; ++f)
          s.rows[free_pos[f].first] +=
              static_cast<uint16_t>(digit[f] * pow[free_pos[f].second]);
        out->push_back(s);
        // Next digit vector (base P counter).
        for (int f = 0; f < nfree; ++f) {
          if (++digit[f] < P) break;
          digit[f] = 0;
        }
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);
  CHECK_EQ(argc, 3) << "Usage: " << argv[0] << " <cert.pb.txt> <out.bin>";

  const pb::Certificate cert = ReadProtoFromFile<pb::Certificate>(argv[1]);
  CHECK_EQ(cert.problem_name(), Problem::Name());
  typename Problem::SymmetryGroup group;
  OrbitMap<Problem> orbit_map(&group);
  int n_orbits = 0;
  for (const pb::ConstrainedTensor &rt : cert.constrained_tensors()) {
    CHECK(rt.has_rank_lower_bound()) << "orbit " << rt.index() << " has no bound";
    orbit_map.Set(ConstraintsFromBytes<P, NA>(rt.constraints()),
                  rt.rank_lower_bound());
    ++n_orbits;
  }
  LOG(INFO) << "Loaded " << n_orbits << " orbits";

  std::vector<Subspace> subs;
  Enumerate(&subs);
  LOG(INFO) << "Enumerated " << subs.size() << " subspaces";

  tbb::parallel_for(size_t{0}, subs.size(), [&](size_t i) {
    Subspace &s = subs[i];
    Cons c;
    for (int r = 0; r < s.dim; ++r) c.push_back(VecFromCode(s.rows[r]));
    const int rank = GaussJordanRREF<P, NA>(&c);
    CHECK_EQ(rank, s.dim);
    c.erase(c.begin(), c.end() - rank);
    s.lb = static_cast<uint8_t>(orbit_map.Get(c));
    for (int r = 0; r < s.dim; ++r) s.rows[r] = CodeFromVec(c[r]);
  });

  FILE *f = std::fopen(argv[2], "wb");
  CHECK(f != nullptr) << "cannot open " << argv[2];
  for (const Subspace &s : subs) {
    std::fwrite(&s.dim, 1, 1, f);
    std::fwrite(s.rows.data(), sizeof(uint16_t), s.dim, f);
    std::fwrite(&s.lb, 1, 1, f);
  }
  std::fclose(f);
  LOG(INFO) << "Wrote " << subs.size() << " records to " << argv[2];
  return 0;
}
