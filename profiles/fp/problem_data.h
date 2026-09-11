#pragma once

#include "profiles/fp/geometry.h"
#include "subspace_bounds/subspace_bounds.h"
#include <set>
#include <string>
#include <unordered_map>

namespace profiles::fp {

using u16 = uint16_t;
struct U64Hash {
  size_t operator()(uint64_t k) const {
    k ^= k >> 31;
    k *= 0x7FB5D329728EA185ull;
    k ^= k >> 27;
    k *= 0x81DADEF4BC2DD44Dull;
    return static_cast<size_t>(k ^ (k >> 33));
  }
};

// Construct once, then borrow as const for the entire worker lifetime.
template <int P, int N0, int N1> struct ProblemData {
  using G = Geometry<P, N0, N1>;
  using Digits = typename G::Digits;
  using Span = typename G::Span;
  static constexpr int NA = G::NA, Q = G::Q, kMaxDim = G::kMaxDim;
  // All nonzero vectors of a span, as point indices (each point once).
  void SpanPoints(const Span &s, std::vector<int> *out) const {
    out->clear();
    const int total = [&] {
      int t = 1;
      for (int i = 0; i < s.dim; ++i)
        t *= P;
      return t;
    }();
    for (int c = 1; c < total; ++c) {
      Digits v{};
      int cc = c;
      for (int i = 0; i < s.dim; ++i) {
        const int coef = cc % P;
        cc /= P;
        if (coef)
          G::SubScaled(v, P - coef, s.rows[i]); // v += coef * row
      }
      const int idx = point_of_code_[G::Encode(v)];
      if (idx >= 0 && (out->empty() || out->back() != idx))
        out->push_back(idx);
    }
    std::sort(out->begin(), out->end());
    out->erase(std::unique(out->begin(), out->end()), out->end());
  }

  // ------------------------------------------------------------ setup
  std::vector<u16> point_code_; // canonical code of point i
  std::vector<Digits> point_digits_;
  std::vector<int> point_rank_;
  std::vector<int> point_of_code_; // code -> point index (-1 for 0)
  int npoints_ = 0;
  int first_of_rank_[N0 + 2] = {}; // first point index of each rank class

  std::unordered_map<uint64_t, uint8_t, U64Hash> table_; // key -> L
  std::vector<std::vector<u16>> perms_; // group as point permutations
  std::vector<std::vector<std::vector<uint32_t>>>
      transporter_; // [q][t] -> perm ids
  int tmax_ = 0;

  void BuildPoints() {
    point_of_code_.assign(Q, -1);
    std::vector<std::pair<int, u16>> reps; // (rank, code)
    for (uint32_t c = 1; c < static_cast<uint32_t>(Q); ++c) {
      const Digits d = G::Decode(c);
      if (G::Encode(G::Normalize(d)) != c)
        continue;
      reps.push_back({G::MatRank(d), static_cast<u16>(c)});
    }
    std::sort(reps.begin(), reps.end());
    for (int i = 0; i <= N0 + 1; ++i)
      first_of_rank_[i] = -1;
    for (const auto &[rk, c] : reps) {
      if (first_of_rank_[rk] < 0)
        first_of_rank_[rk] = point_code_.size();
      point_code_.push_back(c);
      point_digits_.push_back(G::Decode(c));
      point_rank_.push_back(rk);
    }
    npoints_ = point_code_.size();
    for (int i = 0; i < npoints_; ++i) {
      const Digits d = point_digits_[i];
      for (int s = 1; s < P; ++s) {
        Digits e;
        for (int j = 0; j < NA; ++j)
          e[j] = static_cast<uint8_t>(d[j] * s % P);
        point_of_code_[G::Encode(e)] = i;
      }
    }
    LOG(INFO) << "points: " << npoints_ << " (rank classes start at" << [&] {
      std::string s;
      for (int rk = 1; rk <= N0; ++rk)
        s += " " + std::to_string(first_of_rank_[rk]);
      return s;
    }() << ")";
  }

  void LoadBounds(const std::vector<SubspaceBound<NA>> &records) {
    size_t n = 0;
    for (const auto &record : records) {
      std::vector<Digits> rows;
      for (int i = 0; i < record.dim; ++i)
        rows.push_back(G::Decode(record.rows[i]));
      const Span s = G::Rref(rows);
      CHECK_EQ(s.dim, record.dim);
      table_[G::Key(s)] = record.lb;
      ++n;
    }
    LOG(INFO) << "table: " << n << " subspaces; L(0) = " << L0()
              << " (the problem's certified bound)";
  }
  int L0() const {
    Span empty;
    empty.dim = 0;
    return table_.at(G::Key(empty));
  }
  int L(const Span &s) const {
    if (s.dim == 0)
      return L0();
    auto it = table_.find(G::Key(s));
    CHECK(it != table_.end()) << "subspace missing from table";
    return it->second;
  }

  // GL_n(F_P) as row-major digit matrices.
  template <int N> static std::vector<std::array<int, N * N>> GeneralLinear() {
    std::vector<std::array<int, N * N>> out;
    int total = 1;
    for (int i = 0; i < N * N; ++i)
      total *= P;
    for (int c = 0; c < total; ++c) {
      std::array<int, N * N> m;
      int cc = c;
      for (int i = 0; i < N * N; ++i) {
        m[i] = cc % P;
        cc /= P;
      }
      // invertible iff rank N
      std::array<int, N * N> a = m;
      int r = 0;
      for (int col = 0; col < N; ++col) {
        int p = -1;
        for (int i = r; i < N; ++i) {
          if (a[i * N + col]) {
            p = i;
            break;
          }
        }
        if (p < 0)
          continue;
        for (int j = 0; j < N; ++j)
          std::swap(a[r * N + j], a[p * N + j]);
        const int inv = G::Inv(static_cast<uint8_t>(a[r * N + col]));
        for (int j = 0; j < N; ++j)
          a[r * N + j] = a[r * N + j] * inv % P;
        for (int i = 0; i < N; ++i) {
          if (i == r || !a[i * N + col])
            continue;
          const int f = a[i * N + col];
          for (int j = 0; j < N; ++j)
            a[i * N + j] = (a[i * N + j] + (P - f) * a[r * N + j]) % P;
        }
        ++r;
      }
      if (r == N)
        out.push_back(m);
    }
    return out;
  }

  void BuildGroup() {
    const auto gl0 = GeneralLinear<N0>();
    const auto gl1 = GeneralLinear<N1>();
    std::set<std::vector<u16>> seen;
    std::vector<u16> perm(npoints_);
    for (const auto &l : gl0) {
      for (const auto &rmat : gl1) {
        for (int p = 0; p < npoints_; ++p) {
          const Digits &u = point_digits_[p];
          Digits w{};
          for (int i = 0; i < N0; ++i) {
            for (int j = 0; j < N1; ++j) {
              int acc = 0;
              for (int a = 0; a < N0; ++a) {
                for (int b = 0; b < N1; ++b) {
                  acc += l[i * N0 + a] * u[a * N1 + b] * rmat[b * N1 + j];
                }
              }
              w[i * N1 + j] = static_cast<uint8_t>(acc % P);
            }
          }
          perm[p] = static_cast<u16>(point_of_code_[G::Encode(w)]);
        }
        if (seen.insert(perm).second)
          perms_.push_back(perm);
      }
    }
    LOG(INFO) << "group: |GL(" << N0 << ")| = " << gl0.size() << ", |GL(" << N1
              << ")| = " << gl1.size() << ", " << perms_.size()
              << " distinct point permutations";
    // Transporter lists for targets below the first point of each orbit's
    // class (min F is the smallest point of its own G-orbit).
    tmax_ = 0;
    for (int rk = 1; rk <= N0; ++rk) {
      if (first_of_rank_[rk] >= 0)
        tmax_ = std::max(tmax_, first_of_rank_[rk] + 1);
    }
    transporter_.assign(npoints_, std::vector<std::vector<uint32_t>>(tmax_));
    for (uint32_t h = 0; h < perms_.size(); ++h) {
      for (int q = 0; q < npoints_; ++q) {
        const int t = perms_[h][q];
        if (t < tmax_)
          transporter_[q][t].push_back(h);
      }
    }
  }
};

} // namespace profiles::fp
