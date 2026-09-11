#pragma once

#include "profiles/q02_n333/algebra.h"
#include "profiles/q02_n333/options.h"
#include <array>
#include <vector>

namespace profiles::q02_n333 {

using Perm = std::array<u16, 512>;
std::vector<Perm> BuildGroup();
std::vector<u16> Image(const Perm &p, const std::vector<u16> &s);
std::vector<u16> Canonical(const std::vector<Perm> &group,
                           const std::vector<u16> &s);
std::vector<Perm> StabilizerOfB0(const std::vector<Perm> &group);

struct OuterCase {
  // canonical under the stabilizer of B0
  std::vector<u16> w;
  // numbers of rank-2 and rank-3 matrices in W
  int n2 = 0;
  int n3 = 0;
};

std::vector<OuterCase> EnumerateOuter(const std::vector<Perm> &stab,
                                      const Options &options);

} // namespace profiles::q02_n333
