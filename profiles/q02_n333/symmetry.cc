#include "profiles/q02_n333/symmetry.h"
#include <algorithm>
#include <set>

namespace profiles::q02_n333 {

std::vector<Perm> BuildGroup() {
  std::vector<u16> gl;
  for (int a = 1; a < 512; ++a) {
    if (Rank(a) == 3)
      gl.push_back(a);
  }
  CHECK_EQ(gl.size(), size_t{168});
  std::vector<Perm> g;
  g.reserve(2 * 168 * 168);
  for (int t = 0; t < 2; ++t) {
    for (u16 a : gl) {
      for (u16 m : gl) {
        Perm p;
        for (int u = 0; u < 512; ++u) {
          const u16 v =
              t ? Transpose(static_cast<u16>(u)) : static_cast<u16>(u);
          p[u] = Mul(Mul(a, v), m);
        }
        g.push_back(p);
      }
    }
  }
  return g;
}

std::vector<u16> Image(const Perm &p, const std::vector<u16> &s) {
  std::vector<u16> out(s.size());
  for (size_t i = 0; i < s.size(); ++i)
    out[i] = p[s[i]];
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<u16> Canonical(const std::vector<Perm> &group,
                           const std::vector<u16> &s) {
  std::vector<u16> best = Image(group[0], s);
  for (size_t i = 1; i < group.size(); ++i) {
    std::vector<u16> img = Image(group[i], s);
    if (img < best)
      best = std::move(img);
  }
  return best;
}

// Stabilizer of B0 = span{1, 2, 4} (matrices supported on row 1).
std::vector<Perm> StabilizerOfB0(const std::vector<Perm> &group) {
  std::vector<Perm> stab;
  for (const Perm &p : group) {
    if (p[1] < 8 && p[2] < 8 && p[4] < 8)
      stab.push_back(p);
  }
  CHECK_EQ(stab.size(), size_t{4032});
  return stab;
}

std::vector<OuterCase> EnumerateOuter(const std::vector<Perm> &stab,
                                      const Options &options) {
  std::set<std::vector<u16>> seen;
  std::vector<OuterCase> cases;
  if (options.min_w == 0)
    cases.push_back({{}, 0, 0}); // all r factors rank-one
  // cid is coset (rows 2-3 of U) ID; cid 0 is B0 itself
  for (int cid = 1; cid < 64; ++cid) {
    std::vector<u16> cands;
    for (int r1 = 0; r1 < 8; ++r1) {
      const u16 u = static_cast<u16>(cid << 3 | r1);
      if (Rank(u) >= 2)
        cands.push_back(u);
    }
    const int n = cands.size();
    for (uint32_t mask = 1; mask < (1u << n); ++mask) {
      const int k = __builtin_popcount(mask);
      if (k < options.min_w || k > options.max_w)
        continue;
      std::vector<u16> w;
      // numbers of rank-2 and rank-3 matrices in W
      int n2 = 0;
      int n3 = 0;
      for (int i = 0; i < n; ++i) {
        if (mask >> i & 1) {
          w.push_back(cands[i]);
          (Rank(cands[i]) == 2 ? n2 : n3)++;
        }
      }
      if (options.prop3 && 2 * n2 + 3 * n3 < 11)
        continue; // --prop3 excess lemma
      const auto canon = Canonical(stab, w);
      if (seen.insert(canon).second)
        cases.push_back({canon, n2, n3});
    }
  }
  std::sort(cases.begin(), cases.end(),
            [](const OuterCase &a, const OuterCase &b) {
              if (a.w.size() != b.w.size())
                return a.w.size() < b.w.size();
              return a.w < b.w;
            });
  return cases;
}

} // namespace profiles::q02_n333
