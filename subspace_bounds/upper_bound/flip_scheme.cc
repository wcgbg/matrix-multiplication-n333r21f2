#include "subspace_bounds/upper_bound/flip_scheme.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>

#include "ng-log/logging.h"

namespace upper_bound {
namespace {

inline int Plus1Mod3(int i) { return (i + 1) % 3; }
inline int Plus2Mod3(int i) { return (i + 2) % 3; }

// abc -> bca -> cab -> abc, by rotating the three components left by `step`.
inline RankOneTensor RotateABC(const RankOneTensor &abc, int step) {
  DCHECK_GE(step, 0);
  return {abc[step % 3], abc[(step + 1) % 3], abc[(step + 2) % 3]};
}

// Uniform random integer in {0, 1, 2}.
inline int Random3(std::mt19937_64 *rng) {
  while (true) {
    auto r = (*rng)() % 4u;
    if (r < 3) {
      return static_cast<int>(r);
    }
  }
}

// Random permutation of [0, 1, ..., n-1].
std::vector<int> RandomPermutation(int n, std::mt19937_64 *rng) {
  CHECK_GT(n, 0);
  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), *rng);
  return perm;
}

std::string Join(const std::vector<std::string> &parts,
                 const std::string &sep) {
  std::string result;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i != 0) {
      result += sep;
    }
    result += parts[i];
  }
  return result;
}

} // namespace

FlipScheme::FlipScheme(const std::array<int, 3> &dims,
                       const std::vector<RankOneTensor> &terms)
    : dims_(dims) {
  for (int i = 0; i < 3; ++i) {
    CHECK_GT(dims_[i], 0);
    CHECK_LT(dims_[i], 64) << "mode width must fit in a uint64_t bitmask";
  }
  for (const RankOneTensor &abc : terms) {
    Add(abc);
  }
}

void FlipScheme::Add(const RankOneTensor &abc) {
  Vec a = abc[0];
  Vec b = abc[1];
  Vec c = abc[2];
  if (a == 0 || b == 0 || c == 0) {
    return;
  }
  a_to_bc_array_[0].insert({a, {b, c}});
  a_to_bc_array_[1].insert({b, {c, a}});
  a_to_bc_array_[2].insert({c, {a, b}});
}

void FlipScheme::Remove(const RankOneTensor &abc) {
  for (int abc_idx = 0; abc_idx < 3; ++abc_idx) {
    auto &a_to_bc = a_to_bc_array_[abc_idx];
    Vec a = abc[abc_idx];
    Vec b = abc[Plus1Mod3(abc_idx)];
    Vec c = abc[Plus2Mod3(abc_idx)];
    auto range = a_to_bc.equal_range(a);
    bool removed = false;
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second[0] == b && it->second[1] == c) {
        a_to_bc.erase(it);
        removed = true;
        break;
      }
    }
    CHECK(removed);
  }
}

std::vector<RankOneTensor> FlipScheme::RankOneTensors() const {
  std::vector<RankOneTensor> r1_tensors;
  r1_tensors.reserve(Rank());
  for (const auto &a_and_bc : a_to_bc_array_[0]) {
    r1_tensors.push_back(
        {a_and_bc.first, a_and_bc.second[0], a_and_bc.second[1]});
  }
  std::sort(r1_tensors.begin(), r1_tensors.end());
  return r1_tensors;
}

void FlipScheme::CheckConsistency() const {
  std::array<std::vector<RankOneTensor>, 3> r1_tensors_arr3;
  for (int abc_idx = 0; abc_idx < 3; ++abc_idx) {
    for (const auto &a_and_bc : a_to_bc_array_[abc_idx]) {
      RankOneTensor abc = {a_and_bc.first, a_and_bc.second[0],
                           a_and_bc.second[1]};
      r1_tensors_arr3[abc_idx].push_back(RotateABC(abc, 3 - abc_idx));
    }
    std::sort(r1_tensors_arr3[abc_idx].begin(), r1_tensors_arr3[abc_idx].end());
  }
  CHECK(r1_tensors_arr3[0] == r1_tensors_arr3[1]);
  CHECK(r1_tensors_arr3[0] == r1_tensors_arr3[2]);
}

std::string FlipScheme::ToString() const {
  std::ostringstream oss;
  oss << "rank=" << Rank() << ". ";

  std::vector<std::string> terms;
  for (const RankOneTensor &abc : RankOneTensors()) {
    std::array<std::string, 3> factors;
    for (int abc_idx = 0; abc_idx < 3; ++abc_idx) {
      std::vector<std::string> bits;
      for (int i = 0; i < dims_[abc_idx]; ++i) {
        if ((abc[abc_idx] >> i) & Vec(1)) {
          bits.push_back(std::string(1, char('a' + abc_idx)) +
                         std::to_string(i));
        }
      }
      factors[abc_idx] = "(" + Join(bits, "+") + ")";
    }
    terms.push_back(Join({factors[0], factors[1], factors[2]}, "*"));
  }

  oss << Join(terms, " + ");
  return oss.str();
}

int FlipScheme::FlipBC(int abc_idx, int flip_idx,
                       const std::vector<RankOneTensor> &tensors) {
  int rank_before_flip = Rank();
  const RankOneTensor &x = tensors.at(flip_idx);
  RankOneTensor new_x = x;
  for (size_t i = 0; i < tensors.size(); i++) {
    if (static_cast<int>(i) == flip_idx) {
      continue;
    }
    RankOneTensor y = tensors.at(i);
    CHECK_EQ(x[0], y[0]);
    new_x[1] ^= y[1];
    RankOneTensor new_y = y;
    new_y[2] ^= x[2];
    Remove(RotateABC(y, 3 - abc_idx));
    Add(RotateABC(new_y, 3 - abc_idx));
  }
  Remove(RotateABC(x, 3 - abc_idx));
  Add(RotateABC(new_x, 3 - abc_idx));
  int rank_after_flip = Rank();
  return rank_after_flip - rank_before_flip;
}

int FlipScheme::FlipBC(int abc_idx, const std::vector<RankOneTensor> &tensors,
                       int flip_idx) {
  int rank_before_flip = Rank();
  const RankOneTensor &y = tensors.at(flip_idx);
  RankOneTensor new_y = y;
  for (size_t i = 0; i < tensors.size(); i++) {
    if (static_cast<int>(i) == flip_idx) {
      continue;
    }
    const RankOneTensor &x = tensors.at(i);
    CHECK_EQ(x[0], y[0]);
    new_y[2] ^= x[2];
    RankOneTensor new_x = x;
    new_x[1] ^= y[1];
    Remove(RotateABC(x, 3 - abc_idx));
    Add(RotateABC(new_x, 3 - abc_idx));
  }
  Remove(RotateABC(y, 3 - abc_idx));
  Add(RotateABC(new_y, 3 - abc_idx));
  int rank_after_flip = Rank();
  return rank_after_flip - rank_before_flip;
}

bool FlipScheme::TryToFlipBC(int abc_idx,
                             const std::vector<RankOneTensor> &tensors,
                             std::mt19937_64 *rng) {
  if (tensors.size() <= 1) {
    return false;
  }
  std::vector<RankOneTensor> tensors_in_mask;
  while (true) {
    tensors_in_mask.clear();
    tensors_in_mask.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      if ((*rng)() & 1) {
        tensors_in_mask.push_back(tensors[i]);
      }
    }
    if (tensors_in_mask.size() >= 2) {
      break;
    }
  }
  if ((*rng)() & 1) {
    FlipBC(abc_idx, 0, tensors_in_mask);
  } else {
    FlipBC(abc_idx, tensors_in_mask, 0);
  }
  return true;
}

bool FlipScheme::Step(std::mt19937_64 *rng) {
  for (int abc_idx : RandomPermutation(3, rng)) {
    const auto &a_to_bc = a_to_bc_array_[abc_idx];
    std::vector<std::pair<
        std::unordered_multimap<Vec, std::array<Vec, 2>>::const_iterator,
        std::unordered_multimap<Vec, std::array<Vec, 2>>::const_iterator>>
        ranges;
    for (auto it = a_to_bc.begin(); it != a_to_bc.end();) {
      Vec a = it->first;
      auto range = a_to_bc.equal_range(a);
      CHECK(range.first == it);
      CHECK(range.first != range.second);
      if (std::next(range.first) != range.second) {
        // range size is greater than 1.
        ranges.push_back(range);
      }
      it = range.second;
    }
    std::shuffle(ranges.begin(), ranges.end(), *rng);
    for (const auto &range : ranges) {
      size_t range_size = std::distance(range.first, range.second);
      CHECK_GT(range_size, 1u);
      Vec a = range.first->first;
      std::vector<RankOneTensor> tensors_in_range;
      tensors_in_range.reserve(range_size);
      for (auto it = range.first; it != range.second; ++it) {
        tensors_in_range.push_back({a, it->second[0], it->second[1]});
      }
      std::shuffle(tensors_in_range.begin(), tensors_in_range.end(), *rng);
      if (TryToFlipBC(abc_idx, tensors_in_range, rng)) {
        return true;
      }
    }
  }
  return false;
}

void FlipScheme::Plus(int abc_idx, const RankOneTensor &x,
                      const RankOneTensor &y) {
  RankOneTensor new_x = x;
  new_x[1] ^= y[1];
  RankOneTensor new_y = y;
  new_y[0] ^= x[0];
  RankOneTensor new_xy = {x[0], y[1], x[2] ^ y[2]};
  Remove(RotateABC(x, 3 - abc_idx));
  Remove(RotateABC(y, 3 - abc_idx));
  Add(RotateABC(new_x, 3 - abc_idx));
  Add(RotateABC(new_y, 3 - abc_idx));
  Add(RotateABC(new_xy, 3 - abc_idx));
}

void FlipScheme::Plus(std::mt19937_64 *rng) {
  int rank = Rank();
  if (rank < 2) {
    return;
  }
  std::vector<RankOneTensor> tensors;
  tensors.reserve(rank);
  for (auto &a_and_bc : a_to_bc_array_[0]) {
    tensors.push_back({a_and_bc.first, a_and_bc.second[0], a_and_bc.second[1]});
  }
  std::shuffle(tensors.begin(), tensors.end(), *rng);
  // Find three rank-one tensors whose sum has a zero component.
  for (int i = 0; i < rank; ++i) {
    for (int j = i + 1; j < rank; ++j) {
      for (int k = j + 1; k < rank; ++k) {
        const RankOneTensor &ti = tensors[i];
        const RankOneTensor &tj = tensors[j];
        const RankOneTensor &tk = tensors[k];
        RankOneTensor sum = {ti[0] ^ tj[0] ^ tk[0], ti[1] ^ tj[1] ^ tk[1],
                             ti[2] ^ tj[2] ^ tk[2]};
        if (sum[0] == 0 || sum[1] == 0 || sum[2] == 0) {
          std::array<const RankOneTensor *, 3> tijk = {&ti, &tj, &tk};
          std::shuffle(tijk.begin(), tijk.end(), *rng);
          int abc_idx = Random3(rng);
          RankOneTensor x = RotateABC(*tijk[0], abc_idx);
          RankOneTensor y = RotateABC(*tijk[1], abc_idx);
          Plus(abc_idx, x, y);
          for (int s = 0; s < Rank(); ++s) {
            if (!Step(rng)) {
              break;
            }
          }
          return;
        }
      }
    }
  }
  // Didn't find one. Plus two terms i0 < i1 chosen uniformly at random.
  std::uniform_int_distribution<int> dist0(0, rank - 1);
  std::uniform_int_distribution<int> dist1(0, rank - 2);
  int i0 = dist0(*rng);
  int i1 = dist1(*rng);
  if (i0 <= i1) {
    i1++;
  }
  int abc_idx = Random3(rng);
  const auto &a_to_bc = a_to_bc_array_[abc_idx];
  auto it0 = std::next(a_to_bc.begin(), i0);
  auto it1 = std::next(a_to_bc.begin(), i1);
  RankOneTensor tensor0 = {it0->first, it0->second[0], it0->second[1]};
  RankOneTensor tensor1 = {it1->first, it1->second[0], it1->second[1]};
  Plus(abc_idx, tensor0, tensor1);
  for (int s = 0; s < Rank(); ++s) {
    if (!Step(rng)) {
      break;
    }
  }
}

} // namespace upper_bound
