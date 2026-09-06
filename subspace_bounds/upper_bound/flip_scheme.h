#pragma once

// Flip-graph engine for tensor rank *upper* bounds over F_2.
//
// A FlipScheme holds a decomposition of a fixed 3-tensor as a sum of rank-one
// tensors a (X) b (X) c, where a, b, c are vectors over F_2 of width dims_[0],
// dims_[1], dims_[2] respectively (each width < 64, stored as a uint64_t
// bitmask). The number of rank-one terms is the current rank. Two
// rank-preserving / rank-changing rewrites drive a randomized local search:
//
//   * Step (a "flip"): pick a group of terms sharing one component and rewrite
//     them; the represented tensor is unchanged and the rank may drop.
//   * Plus: merge terms to introduce a redundant term, escaping local minima
//     (rank +1), then immediately try to flip it away.
//
// Ported from matrix-multiplication-flip/f2/scheme.{h,cc} (the kFlip + Plus
// path only). The source was specialized to the matrix-multiplication tensor:
// each component was a flattened n x m matrix, so the bit width of a mode was a
// product n*m. Here the tensor is arbitrary, so the bit width of mode i is just
// its plain dimension dims_[i]. The reduce / flip-to-reducible actions (which
// needed a GF(2) linear solver) are not migrated.

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace upper_bound {

using Vec = uint64_t;                     // a mode component as a GF(2) vector
using RankOneTensor = std::array<Vec, 3>; // {a, b, c}

class FlipScheme {
public:
  // `dims` are the bit widths of the three modes (must be > 0 and < 64).
  // `terms` is the initial decomposition; zero terms (any component == 0) are
  // dropped.
  FlipScheme(const std::array<int, 3> &dims,
             const std::vector<RankOneTensor> &terms);

  int Rank() const { return static_cast<int>(a_to_bc_array_[0].size()); }

  // One flip attempt across a random mode/group. Returns true if a flip was
  // applied. The represented tensor is invariant under this operation.
  bool Step(std::mt19937_64 *rng);

  // Introduce a redundant term (rank +1) then flip-reduce, to escape a local
  // minimum. The represented tensor is invariant.
  void Plus(std::mt19937_64 *rng);

  std::vector<RankOneTensor> RankOneTensors() const;

  // Human-readable decomposition, e.g. "rank=7. (a0+a3)*(b1)*(c0+c2) + ...".
  std::string ToString() const;

  // Checks the three component views are mutually consistent (debug aid).
  void CheckConsistency() const;

  const std::array<int, 3> &dims() const { return dims_; }

private:
  void Add(const RankOneTensor &abc);
  void Remove(const RankOneTensor &abc);

  bool TryToFlipBC(int abc_idx, const std::vector<RankOneTensor> &tensors,
                   std::mt19937_64 *rng);
  // Both return the rank change.
  int FlipBC(int abc_idx, int flip_idx,
             const std::vector<RankOneTensor> &tensors);
  int FlipBC(int abc_idx, const std::vector<RankOneTensor> &tensors,
             int flip_idx);
  void Plus(int abc_idx, const RankOneTensor &x, const RankOneTensor &y);

  std::array<int, 3> dims_ = {0, 0, 0};
  // a_to_bc_array_[0] maps a -> (b, c); [1] maps b -> (c, a); [2] maps c ->
  // (a, b). The three views encode the same set of rank-one tensors.
  std::array<std::unordered_multimap<Vec, std::array<Vec, 2>>, 3>
      a_to_bc_array_;
};

} // namespace upper_bound
