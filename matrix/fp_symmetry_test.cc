#include "matrix/fp_symmetry.h"

#include <algorithm>
#include <array>
#include <random>
#include <set>

#include "gtest/gtest.h"

#include "core/constraints.h"
#include "core/gf.h"
#include "core/gf_vec.h"
#include "core/rank_lower_bound_flatten.h" // FlattenTensorA, CyclicTranspose
#include "core/tensor.h"
#include "matrix/tensor.h"

namespace matrix {
namespace {

// All vectors of 𝔽_P^NA in the framework's coordinate order.
template <int P, int NA> std::vector<GFVec<P, NA>> Domain() {
  using Vec = GFVec<P, NA>;
  using F = GF<P>;
  std::vector<Vec> out;
  const int total = static_cast<int>(IntPow(P, NA));
  for (int code = 0; code < total; ++code) {
    Vec v{};
    int c = code;
    for (int i = 0; i < NA; ++i) {
      v.Set(i, F{static_cast<uint8_t>(c % P)});
      c /= P;
    }
    out.push_back(v);
  }
  return out;
}

// |GL(2,3)| = 48, |GL(3,3)| = 11232, |GL(2,5)| = 480.
TEST(FpSymmetryGroupTest, GroupSizes) {
  {
    FpSymmetryGroup<3, 2, 3, 3> g; // non-cube: Store = GL(3,3), Query = GL(2,3)
    EXPECT_EQ(g.store.Size(), 11232);
    EXPECT_EQ(g.query.Size(), 48);
  }
  {
    FpSymmetryGroup<3, 2, 2, 2> g; // cubic: Query doubled by the transpose
    EXPECT_EQ(g.store.Size(), 48);
    EXPECT_EQ(g.query.Size(), 96);
  }
  {
    FpSymmetryGroup<5, 2, 2, 3> g;
    EXPECT_EQ(g.store.Size(), 480);
    EXPECT_EQ(g.query.Size(), 480);
  }
}

// Identity acts trivially; every element is a bijection on functionals whose
// ApplyInverse exactly inverts Apply.
template <int P, int N0, int N1, int N2> void CheckGroup(int store_samples) {
  constexpr int NA = N0 * N1;
  FpSymmetryGroup<P, N0, N1, N2> g;
  using Vec = GFVec<P, NA>;
  const std::vector<Vec> domain = Domain<P, NA>();

  for (const Vec &v : domain) {
    EXPECT_EQ(g.store.Apply(g.store.Identity(), v), v);
    EXPECT_EQ(g.query.Apply(g.query.Identity(), v), v);
  }

  std::mt19937_64 rng(3);
  std::vector<int> store_idx;
  for (int t = 0; t < g.store.Size(); ++t) store_idx.push_back(t);
  std::shuffle(store_idx.begin(), store_idx.end(), rng);
  store_idx.resize(std::min<int>(store_samples, store_idx.size()));
  for (int t : store_idx) {
    const auto store = g.store.At(t);
    std::set<Vec> image;
    for (const Vec &v : domain) image.insert(g.store.Apply(store, v));
    EXPECT_EQ(image.size(), domain.size()) << "store t=" << t;
    for (const Vec &v : domain) {
      EXPECT_EQ(g.store.ApplyInverse(store, g.store.Apply(store, v)), v);
    }
  }

  for (int s = 0; s < g.query.Size(); ++s) {
    const auto query = g.query.At(s);
    std::set<Vec> image;
    for (const Vec &v : domain) image.insert(g.query.Apply(query, v));
    EXPECT_EQ(image.size(), domain.size()) << "query s=" << s;
    for (const Vec &v : domain) {
      EXPECT_EQ(g.query.ApplyInverse(query, g.query.Apply(query, v)), v);
    }
  }
}

TEST(FpSymmetryGroupTest, ActionIsValid) {
  CheckGroup<3, 2, 2, 2>(48);
  CheckGroup<3, 2, 3, 3>(300);
  CheckGroup<5, 2, 2, 3>(100);
}

// Witness round trip through the certificate's fixed32 encoding.
TEST(FpSymmetryGroupTest, QueryElemPacksIntoUint32) {
  FpSymmetryGroup<3, 2, 2, 2> g;
  for (int s = 0; s < g.query.Size(); ++s) {
    const auto e = g.query.At(s);
    const auto back = FpSymmetryGroup<3, 2, 2, 2>::QueryElem(static_cast<uint32_t>(e));
    EXPECT_EQ(back.l, e.l);
    EXPECT_EQ(back.transpose, e.transpose);
  }
}

template <int P, int NA>
Constraints<P, NA> RandomConstraints(std::mt19937_64 &rng) {
  using Vec = GFVec<P, NA>;
  using F = GF<P>;
  std::uniform_int_distribution<int> dim_dist(0, NA);
  std::uniform_int_distribution<int> digit_dist(0, P - 1);
  Constraints<P, NA> rows;
  const int target_dim = dim_dist(rng);
  for (int i = 0; i < target_dim; ++i) {
    Vec v{};
    for (int j = 0; j < NA; ++j) v.Set(j, F{static_cast<uint8_t>(digit_dist(rng))});
    rows.push_back(v);
  }
  const int rank = GaussJordanRREF<P, NA>(&rows);
  rows.erase(rows.begin(), rows.end() - rank);
  return rows;
}

template <int P, int N0, int N1, int N2>
std::array<int, 3> ThreeRanks(const Tensor<P, N0 * N1, N1 * N2, N2 * N0> &t) {
  constexpr int NA = N0 * N1, NB = N1 * N2, NC = N2 * N0;
  const auto t1 = CyclicTranspose<P, NA, NB, NC>(t);
  const auto t2 = CyclicTranspose<P, NB, NC, NA>(t1);
  return {
      FlattenTensorA<P, NA, NB, NC>(t).Rank(),
      FlattenTensorA<P, NB, NC, NA>(t1).Rank(),
      FlattenTensorA<P, NC, NA, NB>(t2).Rank(),
  };
}

// Constraining T_mat by R and by (query·store)·R must give isomorphic tensors;
// check the three flattening ranks (as a multiset, since the cubic transpose
// swaps two modes). Store elements are sampled (GL(3,3) has 11,232).
template <int P, int N0, int N1, int N2>
void CheckConstraintSymmetryInvariance(std::mt19937_64 &rng, int trials, int store_samples) {
  constexpr int NA = N0 * N1, NB = N1 * N2, NC = N2 * N0;
  FpSymmetryGroup<P, N0, N1, N2> g;
  using Tn = Tensor<P, NA, NB, NC>;
  const Tn tensor = BuildMulTensor<P, N0, N1, N2>();
  std::uniform_int_distribution<int> store_dist(0, g.store.Size() - 1);

  for (int trial = 0; trial < trials; ++trial) {
    const Constraints<P, NA> constraints = RandomConstraints<P, NA>(rng);
    const Tn t0 = ApplyConstraintsToTensor<P, NA, NB, NC>(constraints, tensor);
    std::array<int, 3> r0 = ThreeRanks<P, N0, N1, N2>(t0);
    std::sort(r0.begin(), r0.end());

    for (int s = 0; s < g.query.Size(); ++s) {
      const auto query = g.query.At(s);
      for (int k = 0; k < store_samples; ++k) {
        const auto store = g.store.At(store_dist(rng));
        Constraints<P, NA> transformed;
        for (const auto &r : constraints) {
          transformed.push_back(g.query.Apply(query, g.store.Apply(store, r)));
        }
        const int transformed_dim = GaussJordanRREF<P, NA>(&transformed);
        EXPECT_EQ(transformed_dim, static_cast<int>(constraints.size()));

        const Tn t1 = ApplyConstraintsToTensor<P, NA, NB, NC>(transformed, tensor);
        std::array<int, 3> r1 = ThreeRanks<P, N0, N1, N2>(t1);
        std::sort(r1.begin(), r1.end());
        EXPECT_EQ(r0, r1) << "N0=" << N0 << " N1=" << N1 << " N2=" << N2 << " trial=" << trial
                          << " query.l=" << query.l << " query.t=" << query.transpose
                          << " store=" << store;
      }
    }
  }
}

TEST(FpSymmetryGroupTest, ConstraintCommutesWithSymmetry) {
  std::mt19937_64 rng;
  CheckConstraintSymmetryInvariance<3, 2, 2, 2>(rng, 20, 48); // cubic, all of GL(2,3)
  CheckConstraintSymmetryInvariance<3, 2, 3, 3>(rng, 20, 20); // the ⟨2,3,3⟩ target
  CheckConstraintSymmetryInvariance<5, 2, 2, 3>(rng, 5, 10);
}

} // namespace
} // namespace matrix
