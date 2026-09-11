#pragma once

// Expand certified orbit bounds in memory. A record describes the constraint
// space S^perp: dim is the codimension of the restriction space S, and lb is
// its certified rank lower bound. This does not verify certificate proofs.

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "ng-log/logging.h"
#include "tbb/parallel_for.h"

#include "core/bit_vec.h"
#include "core/certificate.pb.h"
#include "core/constraints.h"
#include "core/gf.h"
#include "core/gf_vec.h"
#include "subspace_bounds/search/orbit_map.h"

// A row of (F_P^NA)* is stored as the base-P code sum_c digit_c * P^c (for
// P = 2 this is the plain bit pattern).
// (Templates so that `if constexpr` discards the branch for the other field.)
template <int P, int NA> GFVec<P, NA> SubspaceVecFromCode(uint32_t code) {
  if constexpr (P == 2) {
    return GFVec<P, NA>{static_cast<BitVec<NA>>(code)};
  } else {
    GFVec<P, NA> v{};
    for (int c = 0; c < NA; ++c) {
      v.Set(c, GF<P>{static_cast<uint8_t>(code % P)});
      code /= P;
    }
    return v;
  }
}

template <int P, int NA> uint16_t SubspaceCodeFromVec(const GFVec<P, NA> &v) {
  if constexpr (P == 2) {
    return static_cast<uint16_t>(v.data);
  } else {
    uint32_t code = 0;
    for (int c = NA - 1; c >= 0; --c)
      code = code * P + v[c].value;
    return static_cast<uint16_t>(code);
  }
}

template <int NA> struct SubspaceBound {
  uint8_t dim = 0;
  std::array<uint16_t, NA> rows{};
  uint8_t lb = 0;

  // Order by subspace only, for canonical-key lookups.
  bool operator<(const SubspaceBound &other) const {
    if (dim != other.dim)
      return dim < other.dim;
    return rows < other.rows;
  }
};

namespace subspace_bounds_internal {

// Enumerate all subspaces of F_P^NA as standard RREFs: pivot columns
// p_0 < ... < p_{d-1}, pivot entries 1, zeros left of each pivot and in the
// other rows' pivot columns, arbitrary digits elsewhere. (The exact convention
// is irrelevant: the rows are re-canonicalized by GaussJordanRREF before the
// orbit lookup.)
template <int P, int NA> void Enumerate(std::vector<SubspaceBound<NA>> *out) {
  uint32_t pow[NA + 1];
  pow[0] = 1;
  for (int c = 1; c <= NA; ++c)
    pow[c] = pow[c - 1] * P;
  for (int d = 0; d <= NA; ++d) {
    for (uint32_t pivots = 0; pivots < (1u << NA); ++pivots) {
      if (__builtin_popcount(pivots) != d)
        continue;
      std::vector<int> piv;
      for (int c = 0; c < NA; ++c)
        if (pivots >> c & 1)
          piv.push_back(c);
      std::vector<std::pair<int, int>> free_pos; // (row, col)
      for (int i = 0; i < d; ++i)
        for (int c = piv[i] + 1; c < NA; ++c)
          if (!(pivots >> c & 1))
            free_pos.push_back({i, c});
      const int nfree = free_pos.size();
      uint64_t total = 1;
      for (int f = 0; f < nfree; ++f)
        total *= P;
      std::vector<int> digit(nfree, 0);
      for (uint64_t n = 0; n < total; ++n) {
        SubspaceBound<NA> s;
        s.dim = static_cast<uint8_t>(d);
        for (int i = 0; i < d; ++i)
          s.rows[i] = static_cast<uint16_t>(pow[piv[i]]);
        for (int f = 0; f < nfree; ++f)
          s.rows[free_pos[f].first] +=
              static_cast<uint16_t>(digit[f] * pow[free_pos[f].second]);
        out->push_back(s);
        // Next digit vector (base P counter).
        for (int f = 0; f < nfree; ++f) {
          if (++digit[f] < P)
            break;
          digit[f] = 0;
        }
      }
    }
  }
}

} // namespace subspace_bounds_internal

template <class Problem>
std::vector<SubspaceBound<Problem::kNA>>
ExpandSubspaceBounds(const pb::Certificate &cert) {
  constexpr int P = Problem::kP;
  constexpr int NA = Problem::kNA;
  static_assert(IntPow(P, NA) <= 65535,
                "a row must fit its uint16 base-P code");
  CHECK_EQ(cert.problem_name(), Problem::Name());
  typename Problem::SymmetryGroup group;
  OrbitMap<Problem> orbit_map(&group);
  int n_orbits = 0;
  for (const pb::ConstrainedTensor &rt : cert.constrained_tensors()) {
    CHECK(rt.has_rank_lower_bound())
        << "orbit " << rt.index() << " has no bound";
    orbit_map.Set(ConstraintsFromBytes<P, NA>(rt.constraints()),
                  rt.rank_lower_bound());
    ++n_orbits;
  }
  LOG(INFO) << "Loaded " << n_orbits << " orbits";

  std::vector<SubspaceBound<NA>> subs;
  subspace_bounds_internal::Enumerate<P, NA>(&subs);
  LOG(INFO) << "Enumerated " << subs.size() << " subspaces";

  tbb::parallel_for(size_t{0}, subs.size(), [&](size_t i) {
    SubspaceBound<NA> &s = subs[i];
    Constraints<P, NA> c;
    for (int r = 0; r < s.dim; ++r)
      c.push_back(SubspaceVecFromCode<P, NA>(s.rows[r]));
    const int rank = GaussJordanRREF<P, NA>(&c);
    CHECK_EQ(rank, s.dim);
    c.erase(c.begin(), c.end() - rank);
    s.lb = static_cast<uint8_t>(orbit_map.Get(c));
    for (int r = 0; r < s.dim; ++r)
      s.rows[r] = SubspaceCodeFromVec(c[r]);
  });

  return subs;
}
