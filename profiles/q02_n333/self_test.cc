#include "profiles/q02_n333/self_test.h"
#include <random>
#include <set>

namespace profiles::q02_n333 {

void SelfTest(const CapTable &table, const std::vector<Perm> &group) {
  // Known capacities: rank-one block (L = 15), B_x (L = 9), S_{x,c} (L = 6),
  // singletons (L = 19).
  CHECK_EQ(table.LOfGens({0x1, 0x2, 0x4}), 15);
  CHECK_EQ(table.LOfGens({0x1, 0x8, 0x40}), 15);
  CHECK_EQ(table.LOfGens({0x8, 0x10, 0x20, 0x40, 0x80, 0x100}), 9);
  CHECK_EQ(table.LOfGens({0x1, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100}), 6);
  // Check every premise of the r=20 common-block reduction, including for
  // user-supplied certificates. Each two-space is visited once via its two
  // smallest nonzero vectors.
  for (int a = 1; a < 512; ++a)
    CHECK_EQ(table.LOfGens({static_cast<u16>(a)}), 19);
  int checked = 0;
  for (int a = 1; a < 512; ++a) {
    for (int b = a + 1; b < 512; ++b) {
      if ((a ^ b) < b)
        continue;
      const bool has_one = Rank(a) == 1 || Rank(b) == 1 || Rank(a ^ b) == 1;
      CHECK_EQ(table.LOfGens({static_cast<u16>(a), static_cast<u16>(b)}),
               has_one ? 18 : 19);
      ++checked;
    }
  }
  CHECK_EQ(checked, 43435);
  // Group: distinct elements, capacities invariant.
  {
    std::set<std::vector<u16>> distinct;
    for (const Perm &p : group) {
      distinct.insert(std::vector<u16>(p.begin(), p.end()));
    }
    CHECK_EQ(distinct.size(), group.size());
  }
  std::mt19937_64 rng(7);
  for (int t = 0; t < 2000; ++t) {
    std::vector<u16> gens;
    const int k = 1 + rng() % 5;
    for (int i = 0; i < k; ++i)
      gens.push_back(1 + rng() % 511);
    const Perm &p = group[rng() % group.size()];
    std::vector<u16> img;
    for (u16 v : gens)
      img.push_back(p[v]);
    CHECK_EQ(table.LOfGens(gens), table.LOfGens(img));
  }
  LOG(INFO) << "self-test passed";
}

} // namespace profiles::q02_n333
