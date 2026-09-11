#pragma once

#include "profiles/fp/problem_data.h"
#include <random>

namespace profiles::fp {

template <int P, int N0, int N1>
void SelfTest(const ProblemData<P, N0, N1> &data) {
  using G = Geometry<P, N0, N1>;
  using Digits = typename G::Digits;
  using Span = typename G::Span;
  static constexpr int NA = G::NA;

  // L is constant on rank classes of points (a convention check between
  // the table's coordinates and our matrix layout), and G-invariant.
  for (int rk = 1; rk <= N0; ++rk) {
    const int a = data.first_of_rank_[rk];
    if (a < 0)
      continue;
    const int la = data.L(G::Rref({data.point_digits_[a]}));
    for (int i = a; i < data.npoints_ && data.point_rank_[i] == rk; ++i) {
      CHECK_EQ(data.L(G::Rref({data.point_digits_[i]})), la) << "point " << i;
    }
    LOG(INFO) << "  rank-" << rk << " points: L = " << la;
  }
  std::mt19937_64 rng(11);
  for (int t = 0; t < 3000; ++t) {
    const int k = 1 + rng() % (NA - 1);
    std::vector<Digits> gens, img;
    const auto &h = data.perms_[rng() % data.perms_.size()];
    for (int i = 0; i < k; ++i) {
      const int p = rng() % data.npoints_;
      gens.push_back(data.point_digits_[p]);
      img.push_back(data.point_digits_[h[p]]);
    }
    const Span s = G::Rref(gens), si = G::Rref(img);
    CHECK_EQ(s.dim, si.dim);
    if (s.dim == NA)
      continue;
    CHECK_EQ(data.L(s), data.L(si));
  }
  LOG(INFO) << "self-test passed";
}

} // namespace profiles::fp
