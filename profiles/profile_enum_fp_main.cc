// Generic exhaustive profile enumerator for <N0,N1,*> matrix multiplication
// over a prime field F_P (the field-independent core of profile_enum_q02_n333_main.cc,
// without the <3,3,3>-specific sunflower reduction).
//
// Problem P(r): a multiset F of r nonzero first factors, taken projectively
// (a factor and its nonzero scalar multiples are the same point of
// PG(NA-1, P), NA = N0*N1), such that for every subspace S of A* = F_P^NA
//   m_F(S) := #{terms with factor in S} <= cap(S) := r - L(S),
// where L(S) is the certified lower bound of the tensor restricted to the
// annihilator of S (the table written by subspace_bounds_main, rows as base-P
// codes). Any r-term decomposition yields such an F (the capacity inequality,
// Lemma 2.1 of paper/main.tex), so "no solution" proves rank >= r + 1.
//
// Search: DFS over the points in a fixed order (rank-one points first), a
// non-decreasing sequence with repeats allowed up to cap(<u>), maintaining the
// lattice of spans of all sub-multisets (dims 1..NA-1) with m and cap; every
// touched span is checked, so pruning is exact (a violated capacity of a
// partial multiset stays violated, and a violation always shows on a span of
// chosen points). Dead points (in a saturated span) give a counting bound.
// Orderly generation: only lex-minimal sorted sequences under the group
// G = GL_{N0}(F_P) x GL_{N1}(F_P) acting by U -> L U R (as permutations of the
// points; scalars act trivially) are kept, which is prefix-closed. The
// lex-min test uses transporter lists: h(F) < F needs h(q) <= min F for some
// q in F, and min F is the first point of its G-orbit, so only targets up to
// the first point of each orbit are indexed.
//
//   profile_enum_fp --p=3 --bin=subspace_bounds_q03_n233.bin --r=14
//   profile_enum_fp --p=2 --bin=<F2 table> --r=15 --check_list=...   (validation)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gflags/gflags.h"
#include "ng-log/logging.h"

DEFINE_int32(p, 3, "field characteristic (2 or 3 are compiled in)");
DEFINE_int32(n0, 2, "rows of the first-factor matrices");
DEFINE_int32(n1, 3, "columns of the first-factor matrices");
DEFINE_string(bin, "", "subspace_bounds table for the problem");
DEFINE_int32(r, 14, "number of terms; capacities are r - L(S)");
DEFINE_uint64(budget, 1'000'000'000, "DFS node budget per job");
DEFINE_int32(max_solutions, 100, "stop after this many canonical profiles");
DEFINE_int32(log_seconds, 30, "progress log interval");
DEFINE_bool(symmetry, true, "orderly generation under G");
DEFINE_int32(threads, 1, "worker threads");
DEFINE_int32(split_depth, 0, "split the search into jobs by the first choices");
DEFINE_string(check_list, "", "comma-separated factor codes: check this list only");
DEFINE_string(ones_subset, "", "testing: restrict candidates to these point codes");

namespace {

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

std::vector<uint32_t> ParseList(const std::string &s) {
  std::vector<uint32_t> out;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t next = s.find(',', pos);
    if (next == std::string::npos) next = s.size();
    out.push_back(std::stoul(s.substr(pos, next - pos)));
    pos = next + 1;
  }
  return out;
}

template <int P, int N0, int N1> class Engine {
 public:
  static constexpr int NA = N0 * N1;
  static constexpr int Q = [] {
    int q = 1;
    for (int i = 0; i < NA; ++i) q *= P;
    return q;
  }();
  using Digits = std::array<uint8_t, NA>;

  // ------------------------------------------------------------ field
  static uint8_t Inv(uint8_t a) {
    for (int b = 1; b < P; ++b) {
      if (a * b % P == 1) return static_cast<uint8_t>(b);
    }
    CHECK(false) << "no inverse";
    return 0;
  }
  static Digits Decode(uint32_t code) {
    Digits d;
    for (int i = 0; i < NA; ++i) {
      d[i] = static_cast<uint8_t>(code % P);
      code /= P;
    }
    return d;
  }
  static uint32_t Encode(const Digits &d) {
    uint32_t code = 0;
    for (int i = NA - 1; i >= 0; --i) code = code * P + d[i];
    return code;
  }
  static bool IsZero(const Digits &d) {
    for (int i = 0; i < NA; ++i) {
      if (d[i]) return false;
    }
    return true;
  }
  // a - c*b
  static void SubScaled(Digits &a, int c, const Digits &b) {
    for (int i = 0; i < NA; ++i) a[i] = static_cast<uint8_t>((a[i] + (P - c) * b[i] % P) % P);
  }
  static Digits Normalize(Digits d) { // first nonzero digit -> 1
    for (int i = 0; i < NA; ++i) {
      if (d[i]) {
        const int s = Inv(d[i]);
        for (int j = 0; j < NA; ++j) d[j] = static_cast<uint8_t>(d[j] * s % P);
        return d;
      }
    }
    return d;
  }
  // rank of the N0 x N1 matrix with entry (i, j) = d[i*N1 + j]
  static int MatRank(const Digits &d) {
    std::array<std::array<int, N1>, N0> m;
    for (int i = 0; i < N0; ++i)
      for (int j = 0; j < N1; ++j) m[i][j] = d[i * N1 + j];
    int r = 0;
    for (int col = 0; col < N1 && r < N0; ++col) {
      int piv = -1;
      for (int i = r; i < N0; ++i) {
        if (m[i][col]) { piv = i; break; }
      }
      if (piv < 0) continue;
      std::swap(m[r], m[piv]);
      const int s = Inv(static_cast<uint8_t>(m[r][col]));
      for (int j = 0; j < N1; ++j) m[r][j] = m[r][j] * s % P;
      for (int i = 0; i < N0; ++i) {
        if (i == r || !m[i][col]) continue;
        const int f = m[i][col];
        for (int j = 0; j < N1; ++j) m[i][j] = (m[i][j] + (P - f) * m[r][j]) % P;
      }
      ++r;
    }
    return r;
  }

  // -------------------------------------------------------- subspaces
  static constexpr int kMaxDim = NA - 1; // dim NA is the whole space: cap = r
  struct Span {
    uint8_t dim = 0;
    Digits rows[NA];
    int piv[NA];
  };
  static constexpr int kRowBits = [] {
    int b = 0;
    while ((1u << b) <= static_cast<unsigned>(Q)) ++b;
    return b;
  }();
  static_assert(kRowBits * NA + 4 <= 64, "span key must fit in 64 bits");

  // RREF (pivot = first nonzero column, pivot entry 1, pivot columns cleared),
  // rows in ascending pivot order.
  static Span Rref(std::vector<Digits> rows) {
    int r = 0;
    Span s;
    for (int col = 0; col < NA; ++col) {
      int p = -1;
      for (int k = r; k < static_cast<int>(rows.size()); ++k) {
        if (rows[k][col]) { p = k; break; }
      }
      if (p < 0) continue;
      std::swap(rows[r], rows[p]);
      const int inv = Inv(rows[r][col]);
      for (int j = 0; j < NA; ++j) rows[r][j] = static_cast<uint8_t>(rows[r][j] * inv % P);
      for (int k = 0; k < static_cast<int>(rows.size()); ++k) {
        if (k != r && rows[k][col]) SubScaled(rows[k], rows[k][col], rows[r]);
      }
      s.piv[r] = col;
      ++r;
    }
    // Copy only now: later pivot columns keep eliminating earlier rows.
    for (int i = 0; i < r; ++i) s.rows[i] = rows[i];
    s.dim = static_cast<uint8_t>(r);
    return s;
  }
  static uint64_t Key(const Span &s) {
    uint64_t key = s.dim;
    for (int i = 0; i < s.dim; ++i) key = (key << kRowBits) | Encode(s.rows[i]);
    return key;
  }
  static bool InSpan(const Span &s, Digits v) {
    for (int i = 0; i < s.dim; ++i) {
      if (v[s.piv[i]]) SubScaled(v, v[s.piv[i]], s.rows[i]);
    }
    return IsZero(v);
  }
  static Span Join(const Span &s, const Digits &v) {
    std::vector<Digits> rows(s.rows, s.rows + s.dim);
    rows.push_back(v);
    return Rref(rows);
  }
  // All nonzero vectors of a span, as point indices (each point once).
  void SpanPoints(const Span &s, std::vector<int> *out) const {
    out->clear();
    const int total = [&] {
      int t = 1;
      for (int i = 0; i < s.dim; ++i) t *= P;
      return t;
    }();
    for (int c = 1; c < total; ++c) {
      Digits v{};
      int cc = c;
      for (int i = 0; i < s.dim; ++i) {
        const int coef = cc % P;
        cc /= P;
        if (coef) SubScaled(v, P - coef, s.rows[i]); // v += coef * row
      }
      const int idx = point_of_code_[Encode(v)];
      if (idx >= 0 && (out->empty() || out->back() != idx)) out->push_back(idx);
    }
    std::sort(out->begin(), out->end());
    out->erase(std::unique(out->begin(), out->end()), out->end());
  }

  // ------------------------------------------------------------ setup
  std::vector<u16> point_code_;      // canonical code of point i
  std::vector<Digits> point_digits_;
  std::vector<int> point_rank_;
  std::vector<int> point_of_code_;   // code -> point index (-1 for 0)
  int npoints_ = 0;
  int first_of_rank_[N0 + 2];        // first point index of each rank class

  std::unordered_map<uint64_t, uint8_t, U64Hash> table_; // key -> L
  std::vector<std::vector<u16>> perms_;                  // group as point permutations
  std::vector<std::vector<std::vector<uint32_t>>> transporter_; // [q][t] -> perm ids
  int tmax_ = 0;

  void BuildPoints() {
    point_of_code_.assign(Q, -1);
    std::vector<std::pair<int, u16>> reps; // (rank, code)
    for (uint32_t c = 1; c < static_cast<uint32_t>(Q); ++c) {
      const Digits d = Decode(c);
      if (Encode(Normalize(d)) != c) continue;
      reps.push_back({MatRank(d), static_cast<u16>(c)});
    }
    std::sort(reps.begin(), reps.end());
    for (int i = 0; i <= N0 + 1; ++i) first_of_rank_[i] = -1;
    for (const auto &[rk, c] : reps) {
      if (first_of_rank_[rk] < 0) first_of_rank_[rk] = point_code_.size();
      point_code_.push_back(c);
      point_digits_.push_back(Decode(c));
      point_rank_.push_back(rk);
    }
    npoints_ = point_code_.size();
    for (int i = 0; i < npoints_; ++i) {
      const Digits d = point_digits_[i];
      for (int s = 1; s < P; ++s) {
        Digits e;
        for (int j = 0; j < NA; ++j) e[j] = static_cast<uint8_t>(d[j] * s % P);
        point_of_code_[Encode(e)] = i;
      }
    }
    LOG(INFO) << "points: " << npoints_ << " (rank classes start at"
              << [&] {
                   std::string s;
                   for (int rk = 1; rk <= N0; ++rk) s += " " + std::to_string(first_of_rank_[rk]);
                   return s;
                 }()
              << ")";
  }

  void LoadTable(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    CHECK(f != nullptr) << "cannot open " << path;
    uint8_t d, lb;
    u16 raw[16];
    size_t n = 0;
    while (std::fread(&d, 1, 1, f) == 1) {
      CHECK_LE(d, NA);
      CHECK_EQ(std::fread(raw, 2, d, f), static_cast<size_t>(d));
      CHECK_EQ(std::fread(&lb, 1, 1, f), size_t{1});
      std::vector<Digits> rows;
      for (int i = 0; i < d; ++i) rows.push_back(Decode(raw[i]));
      const Span s = Rref(rows);
      CHECK_EQ(s.dim, d);
      table_[Key(s)] = lb;
      ++n;
    }
    std::fclose(f);
    LOG(INFO) << "table: " << n << " subspaces; L(0) = " << L0() << " (the problem's certified bound)";
    CHECK_GE(FLAGS_r, L0()) << "r below the certified rank: trivially infeasible";
  }
  int L0() const {
    Span empty;
    empty.dim = 0;
    return table_.at(Key(empty));
  }
  int L(const Span &s) const {
    if (s.dim == 0) return L0();
    auto it = table_.find(Key(s));
    CHECK(it != table_.end()) << "subspace missing from table";
    return it->second;
  }
  int Cap(const Span &s) const { return FLAGS_r - L(s); }

  // GL_n(F_P) as row-major digit matrices.
  template <int N> static std::vector<std::array<int, N * N>> GeneralLinear() {
    std::vector<std::array<int, N * N>> out;
    int total = 1;
    for (int i = 0; i < N * N; ++i) total *= P;
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
          if (a[i * N + col]) { p = i; break; }
        }
        if (p < 0) continue;
        for (int j = 0; j < N; ++j) std::swap(a[r * N + j], a[p * N + j]);
        const int inv = Inv(static_cast<uint8_t>(a[r * N + col]));
        for (int j = 0; j < N; ++j) a[r * N + j] = a[r * N + j] * inv % P;
        for (int i = 0; i < N; ++i) {
          if (i == r || !a[i * N + col]) continue;
          const int f = a[i * N + col];
          for (int j = 0; j < N; ++j) a[i * N + j] = (a[i * N + j] + (P - f) * a[r * N + j]) % P;
        }
        ++r;
      }
      if (r == N) out.push_back(m);
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
          perm[p] = static_cast<u16>(point_of_code_[Encode(w)]);
        }
        if (seen.insert(perm).second) perms_.push_back(perm);
      }
    }
    LOG(INFO) << "group: |GL(" << N0 << ")| = " << gl0.size() << ", |GL(" << N1
              << ")| = " << gl1.size() << ", " << perms_.size()
              << " distinct point permutations";
    // Transporter lists for targets below the first point of each orbit's
    // class (min F is the smallest point of its own G-orbit).
    tmax_ = 0;
    for (int rk = 1; rk <= N0; ++rk) {
      if (first_of_rank_[rk] >= 0) tmax_ = std::max(tmax_, first_of_rank_[rk] + 1);
    }
    transporter_.assign(npoints_, std::vector<std::vector<uint32_t>>(tmax_));
    for (uint32_t h = 0; h < perms_.size(); ++h) {
      for (int q = 0; q < npoints_; ++q) {
        const int t = perms_[h][q];
        if (t < tmax_) transporter_[q][t].push_back(h);
      }
    }
  }

  void SelfTest() {
    // L is constant on rank classes of points (a convention check between
    // the table's coordinates and our matrix layout), and G-invariant.
    for (int rk = 1; rk <= N0; ++rk) {
      const int a = first_of_rank_[rk];
      if (a < 0) continue;
      const int la = L(Rref({point_digits_[a]}));
      for (int i = a; i < npoints_ && point_rank_[i] == rk; ++i) {
        CHECK_EQ(L(Rref({point_digits_[i]})), la) << "point " << i;
      }
      LOG(INFO) << "  rank-" << rk << " points: L = " << la;
    }
    std::mt19937_64 rng(11);
    for (int t = 0; t < 3000; ++t) {
      const int k = 1 + rng() % (NA - 1);
      std::vector<Digits> gens, img;
      const auto &h = perms_[rng() % perms_.size()];
      for (int i = 0; i < k; ++i) {
        const int p = rng() % npoints_;
        gens.push_back(point_digits_[p]);
        img.push_back(point_digits_[h[p]]);
      }
      const Span s = Rref(gens), si = Rref(img);
      CHECK_EQ(s.dim, si.dim);
      if (s.dim == NA) continue;
      CHECK_EQ(L(s), L(si));
    }
    LOG(INFO) << "self-test passed";
  }

  // ------------------------------------------------------------ search
  struct Node {
    uint64_t key;
    uint8_t cap, m;
    Span span;
  };
  struct Undo {
    size_t first_new;
    std::vector<int> incremented;
  };

  std::vector<int> cand_;            // candidate point indices (ascending)
  std::vector<int> cap1_;            // cap of <point> per candidate position
  std::vector<Node> nodes_;
  std::unordered_map<uint64_t, int, U64Hash> index_;
  std::vector<int> chosen_;          // candidate positions, non-decreasing
  std::vector<int> mult_;            // multiplicity per candidate position
  int need_ = 0;
  uint64_t budget_ = 0, nodes_visited_ = 0;
  bool stop_ = false;
  std::set<std::vector<int>> solutions_;
  std::chrono::steady_clock::time_point last_log_;
  std::string last_violation_;
  std::vector<const std::vector<u16> *> group_; // perms used for lex-min (may be restricted)
  bool restricted_ = false;

  void SetupCandidates(const std::vector<uint32_t> &subset_codes) {
    cand_.clear();
    if (subset_codes.empty()) {
      for (int i = 0; i < npoints_; ++i) cand_.push_back(i);
      restricted_ = false;
    } else {
      std::set<int> s;
      for (uint32_t c : subset_codes) {
        CHECK_LT(c, static_cast<uint32_t>(Q));
        CHECK_GE(point_of_code_[c], 0);
        s.insert(point_of_code_[c]);
      }
      cand_.assign(s.begin(), s.end());
      restricted_ = true;
    }
    cap1_.clear();
    for (int i : cand_) cap1_.push_back(Cap(Rref({point_digits_[i]})));
    group_.clear();
    if (FLAGS_symmetry) {
      for (const auto &h : perms_) {
        if (restricted_) {
          bool ok = true;
          for (int i : cand_) {
            if (!std::binary_search(cand_.begin(), cand_.end(), static_cast<int>(h[i]))) {
              ok = false;
              break;
            }
          }
          if (!ok) continue;
        }
        group_.push_back(&h);
      }
      LOG(INFO) << "candidates: " << cand_.size() << ", symmetry group used: " << group_.size();
    }
  }

  void Reset() {
    nodes_.clear();
    index_.clear();
    chosen_.clear();
    mult_.assign(cand_.size(), 0);
  }

  void Violation(const Span &s, int m, int cap) {
    last_violation_ = "dim " + std::to_string(s.dim) + " span{";
    for (int i = 0; i < s.dim; ++i) {
      last_violation_ += (i ? "," : "") + std::to_string(Encode(s.rows[i]));
    }
    last_violation_ += "} m=" + std::to_string(m) + " cap=" + std::to_string(cap) +
                       " (L=" + std::to_string(FLAGS_r - cap) + ")";
  }

  bool Create(const Span &s, Undo *undo) {
    (void)undo;
    const uint64_t key = Key(s);
    if (index_.count(key)) return true;
    Node nd;
    nd.key = key;
    nd.span = s;
    nd.cap = static_cast<uint8_t>(std::max(0, Cap(s)));
    nd.m = 0;
    for (int pos : chosen_) {
      if (InSpan(s, point_digits_[cand_[pos]])) ++nd.m;
    }
    index_.emplace(key, static_cast<int>(nodes_.size()));
    nodes_.push_back(nd);
    if (nd.m > nd.cap) {
      Violation(s, nd.m, nd.cap);
      return false;
    }
    return true;
  }

  bool Add(int pos, Undo *undo) {
    undo->first_new = nodes_.size();
    undo->incremented.clear();
    chosen_.push_back(pos);
    ++mult_[pos];
    const Digits &u = point_digits_[cand_[pos]];
    const size_t n0 = undo->first_new;
    bool singleton_seen = false;
    for (size_t i = 0; i < n0; ++i) {
      Node &nd = nodes_[i];
      if (InSpan(nd.span, u)) {
        if (nd.span.dim == 1) singleton_seen = true;
        ++nd.m;
        undo->incremented.push_back(static_cast<int>(i));
        if (nd.m > nd.cap) {
          Violation(nd.span, nd.m, nd.cap);
          return false;
        }
      } else if (nd.span.dim < kMaxDim) {
        if (!Create(Join(nd.span, u), undo)) return false;
      }
    }
    if (!singleton_seen && !Create(Rref({u}), undo)) return false;
    return true;
  }

  void UndoAdd(const Undo &undo) {
    --mult_[chosen_.back()];
    chosen_.pop_back();
    for (int i : undo.incremented) --nodes_[i].m;
    for (size_t j = undo.first_new; j < nodes_.size(); ++j) index_.erase(nodes_[j].key);
    nodes_.resize(undo.first_new);
  }

  bool EnoughLive(int idx, int need) {
    std::vector<char> dead(cand_.size(), 0);
    std::vector<int> pts;
    for (const Node &nd : nodes_) {
      if (nd.m < nd.cap) continue;
      SpanPoints(nd.span, &pts);
      for (int p : pts) {
        const auto it = std::lower_bound(cand_.begin(), cand_.end(), p);
        if (it != cand_.end() && *it == p) dead[it - cand_.begin()] = 1;
      }
    }
    int live = 0;
    for (size_t i = idx; i < cand_.size(); ++i) {
      if (!dead[i]) {
        live += cap1_[i] - mult_[i];
        if (live >= need) return true;
      }
    }
    return live >= need;
  }

  // Is chosen_ (as sorted point list) plus point pos lex-minimal under G?
  bool LexMinWith(int pos) {
    if (group_.empty()) return true;
    const size_t k = chosen_.size() + 1;
    int p[64], img[64];
    for (size_t i = 0; i + 1 < k; ++i) p[i] = cand_[chosen_[i]];
    p[k - 1] = cand_[pos];
    const int p1 = p[0];
    auto test = [&](const std::vector<u16> &h) {
      for (size_t i = 0; i < k; ++i) img[i] = h[p[i]];
      std::sort(img, img + k);
      for (size_t i = 0; i < k; ++i) {
        if (img[i] < p[i]) return false;
        if (img[i] > p[i]) return true;
      }
      return true;
    };
    if (restricted_ || p1 >= tmax_) {
      for (const auto *h : group_) {
        if (!test(*h)) return false;
      }
      return true;
    }
    // h(F) < F requires h(q) <= p1 for some q in F.
    int last_q = -1;
    for (size_t i = 0; i < k; ++i) {
      const int q = p[i];
      if (q == last_q) continue;
      last_q = q;
      for (int t = 0; t <= p1; ++t) {
        for (uint32_t h : transporter_[q][t]) {
          if (!test(perms_[h])) return false;
        }
      }
    }
    return true;
  }

  void Record() {
    std::vector<int> sol;
    for (int pos : chosen_) sol.push_back(cand_[pos]);
    std::sort(sol.begin(), sol.end());
    std::vector<int> best = sol;
    if (FLAGS_symmetry || true) {
      std::vector<int> img(sol.size());
      for (const auto &h : perms_) {
        for (size_t i = 0; i < sol.size(); ++i) img[i] = h[sol[i]];
        std::sort(img.begin(), img.end());
        if (img < best) best = img;
      }
    }
    if (solutions_.insert(best).second) {
      std::string s;
      for (int p : best) s += " " + std::to_string(point_code_[p]);
      LOG(INFO) << "PROFILE #" << solutions_.size() << " (point codes):" << s;
      if (static_cast<int>(solutions_.size()) >= FLAGS_max_solutions) stop_ = true;
    }
  }

  void Dfs(int idx, int need) {
    if (stop_) return;
    if (need == 0) {
      Record();
      return;
    }
    if (idx >= static_cast<int>(cand_.size())) return;
    if (!EnoughLive(idx, need)) return;
    if (mult_[idx] < cap1_[idx]) {
      ++nodes_visited_;
      if (nodes_visited_ > budget_) {
        stop_ = true;
        return;
      }
      MaybeLog(idx);
      if (LexMinWith(idx)) {
        Undo undo;
        if (Add(idx, &undo)) Dfs(idx, need - 1); // stay: repeats allowed
        UndoAdd(undo);
      }
      if (stop_) return;
    }
    Dfs(idx + 1, need);
  }

  void MaybeLog(int idx) {
    if ((nodes_visited_ & 0x3FF) != 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_).count() <
        FLAGS_log_seconds) {
      return;
    }
    last_log_ = now;
    std::string prefix;
    for (int pos : chosen_) prefix += " " + std::to_string(point_code_[cand_[pos]]);
    LOG(INFO) << "  ... nodes=" << nodes_visited_ << " depth=" << chosen_.size() << " idx=" << idx
              << " lattice=" << nodes_.size() << " prefix:" << prefix;
  }

  std::string RunFrom(const std::vector<int> &prefix, int start, uint64_t budget) {
    Reset();
    budget_ = budget;
    nodes_visited_ = 0;
    solutions_.clear();
    stop_ = false;
    last_log_ = std::chrono::steady_clock::now();
    std::vector<Undo> undos(prefix.size());
    size_t added = 0;
    bool ok = true;
    for (int pos : prefix) {
      if (!Add(pos, &undos[added++])) { ok = false; break; }
    }
    if (ok) Dfs(start, need_ - static_cast<int>(prefix.size()));
    while (added > 0) UndoAdd(undos[--added]);
    if (stop_ && solutions_.empty()) return "OVER BUDGET";
    if (!solutions_.empty()) return "FEASIBLE";
    return "INFEASIBLE";
  }

  void PrefixDfs(int idx, int depth, std::vector<int> *prefix,
                 std::vector<std::pair<std::vector<int>, int>> *out) {
    if (static_cast<int>(prefix->size()) == depth) {
      out->push_back({*prefix, idx});
      return;
    }
    const int need = need_ - static_cast<int>(prefix->size());
    if (idx >= static_cast<int>(cand_.size())) return;
    if (!EnoughLive(idx, need)) return;
    if (mult_[idx] < cap1_[idx] && LexMinWith(idx)) {
      Undo undo;
      if (Add(idx, &undo)) {
        prefix->push_back(idx);
        PrefixDfs(idx, depth, prefix, out);
        prefix->pop_back();
      }
      UndoAdd(undo);
    }
    PrefixDfs(idx + 1, depth, prefix, out);
  }
  std::vector<std::pair<std::vector<int>, int>> Prefixes(int depth) {
    Reset();
    std::vector<std::pair<std::vector<int>, int>> out;
    if (depth <= 0 || need_ <= depth) {
      out.push_back({{}, 0});
      return out;
    }
    std::vector<int> prefix;
    PrefixDfs(0, depth, &prefix, &out);
    return out;
  }

  bool Check(const std::vector<uint32_t> &codes) {
    Reset();
    for (uint32_t c : codes) {
      CHECK_LT(c, static_cast<uint32_t>(Q));
      const int p = point_of_code_[c];
      CHECK_GE(p, 0) << "zero factor";
      const auto it = std::lower_bound(cand_.begin(), cand_.end(), p);
      CHECK(it != cand_.end() && *it == p);
      Undo undo;
      if (!Add(it - cand_.begin(), &undo)) {
        LOG(ERROR) << "violation after adding code " << c << ": " << last_violation_;
        return false;
      }
    }
    LOG(INFO) << "list of " << codes.size() << " passes all capacities at r=" << FLAGS_r
              << " (lattice nodes: " << nodes_.size() << ")";
    return true;
  }
};

template <int P, int N0, int N1> int RunEngine() {
  using E = Engine<P, N0, N1>;
  auto engine = std::make_unique<E>();
  engine->BuildPoints();
  engine->LoadTable(FLAGS_bin);
  engine->BuildGroup();
  engine->SelfTest();
  engine->need_ = FLAGS_r;
  engine->SetupCandidates(ParseList(FLAGS_ones_subset));

  if (!FLAGS_check_list.empty()) {
    const bool ok = engine->Check(ParseList(FLAGS_check_list));
    std::printf("check_list: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
  }

  const auto prefixes = engine->Prefixes(FLAGS_split_depth);
  LOG(INFO) << prefixes.size() << " job(s)";
  std::mutex mu;
  std::atomic<size_t> next_job{0};
  uint64_t total_nodes = 0;
  int over_budget = 0;
  std::set<std::vector<int>> all_solutions;
  const auto t_start = std::chrono::steady_clock::now();
  auto worker = [&](E *e) {
    while (true) {
      const size_t k = next_job.fetch_add(1);
      if (k >= prefixes.size()) break;
      const std::string verdict = e->RunFrom(prefixes[k].first, prefixes[k].second, FLAGS_budget);
      std::lock_guard<std::mutex> lock(mu);
      total_nodes += e->nodes_visited_;
      if (verdict == "OVER BUDGET") ++over_budget;
      for (const auto &s : e->solutions_) all_solutions.insert(s);
      if (prefixes.size() > 1 && k % 100 == 0) {
        LOG(INFO) << "  progress: " << k << "/" << prefixes.size() << " jobs, nodes=" << total_nodes
                  << ", "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count()
                  << "s";
      }
    }
  };
  {
    const int nthreads = std::max(1, std::min<int>(FLAGS_threads, prefixes.size()));
    std::vector<std::unique_ptr<E>> engines;
    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t) {
      // Each worker needs its own search state but shares the tables: copy
      // the engine (tables are copied too; fine for these sizes).
      engines.push_back(std::make_unique<E>(*engine));
      pool.emplace_back(worker, engines.back().get());
    }
    worker(engine.get());
    for (auto &th : pool) th.join();
  }
  std::string verdict = "INFEASIBLE";
  if (!all_solutions.empty()) verdict = "FEASIBLE";
  else if (over_budget > 0) verdict = "OVER BUDGET";
  std::printf("r=%d: %s jobs=%zu nodes=%llu profiles=%zu over_budget_jobs=%d time=%.1fs\n",
              FLAGS_r, verdict.c_str(), prefixes.size(),
              static_cast<unsigned long long>(total_nodes), all_solutions.size(), over_budget,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
  for (const auto &sol : all_solutions) {
    std::string s;
    for (int p : sol) s += " " + std::to_string(engine->point_code_[p]);
    std::printf("  profile:%s\n", s.c_str());
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);
  CHECK(!FLAGS_bin.empty()) << "--bin is required";
  const int key = FLAGS_p * 100 + FLAGS_n0 * 10 + FLAGS_n1;
  switch (key) {
    case 323: return RunEngine<3, 2, 3>();
    case 223: return RunEngine<2, 2, 3>();
    case 232: return RunEngine<2, 3, 2>();
    case 222: return RunEngine<2, 2, 2>();
    default: LOG(FATAL) << "unsupported (p, n0, n1) = (" << FLAGS_p << ", " << FLAGS_n0 << ", " << FLAGS_n1 << ")";
  }
}
