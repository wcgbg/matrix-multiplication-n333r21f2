// Exhaustive decision of Problem P (paper/main.tex, Section 2: the reduction
// to a finite enumeration) for 20-term
// decompositions of <3,3,3> over F_2, using the structure forced by the
// certified capacities (Proposition 2.2 and Lemmas 2.3 and 2.4 of the paper;
// certificate of 2026-09-02):
//
//   * every 2-space has cap <= 2, so the 20 first factors form a cap set (no
//     dependent triple); and a 2-space has cap 1 exactly when it contains no
//     rank-one matrix, so any two chosen factors of rank >= 2 differ by a
//     rank-one matrix. Those differences are closed under addition, hence
//     lie in one rank-one block B = {x c^T : c} or its transpose: the
//     rank->=2 factors W sit in ONE coset U0 + B, |W| <= 8 (the optional
//     excess lemma behind --prop3 also gives 2 n2 + 3 n3 >= 11, i.e.
//     |W| >= 4; the notes do not use it);
//   * the other n1 = 20 - |W| >= 12 factors are rank-one matrices a b^T.
//
// Outer loop: (coset of the fixed block B0 = {e1 c^T}, W) up to the
// stabiliser of B0 in the symmetry group G = {U -> A U M, U -> A U^T M}
// (|G| = 56,448; all 14 blocks form one G-orbit, so fixing B0 loses nothing).
// Inner loop: exact DFS over the 49 rank-one matrices that maintains the
// lattice of spans of all subsets of the chosen set (dims <= kLatticeMaxDim)
// and checks m(S) <= cap(S) = 20 - L(S) on every touched span, plus the
// optional contextual caps m(B_x) <= 7 and m(S_{x,c}) <= 12 (--prop3), and the
// dim-7/8 capacities at the leaves. Pruning is sound (a violated capacity of a
// partial set stays violated) and the check is complete (if m(S) > cap(S)
// then S' = span(chosen ∩ S) ⊆ S has m(S') = m(S) > cap(S) >= cap(S')).
//
// Verdict per outer case: INFEASIBLE / FEASIBLE (profiles printed) / OVER
// BUDGET. Diagnostic tool, not part of the verifier trust base.
//
//   bazel run --config=opt //profiles:profile_enum_q02_n333_main -- \
//       --bin=/abs/path/subspace_bounds_q02_n333.bin [--list_only] [--cases=3,7]

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "gflags/gflags.h"
#include "ng-log/logging.h"

DEFINE_string(bin, "", "subspace_bounds_q02_n333.bin (from subspace_bounds_main)");
DEFINE_int32(min_w, 4, "smallest |W| (rank->=2 factors) to consider");
DEFINE_int32(max_w, 8, "largest |W| to consider");
DEFINE_uint64(budget, 1'000'000'000, "DFS node budget per outer case");
DEFINE_string(cases, "", "comma-separated outer case indices to run (default all)");
DEFINE_bool(list_only, false, "only enumerate and print the outer cases");
DEFINE_int32(max_solutions, 50, "stop an outer case after this many profiles");
DEFINE_int32(log_seconds, 10, "progress log interval");
DEFINE_int32(r, 20, "number of terms; capacities are r - L(S)");
DEFINE_bool(prop3, true,
            "use the excess lemma (needs r = 20; not needed for the result): "
            "contextual caps m(B_x) <= 7, "
            "m(S_{x,c}) <= 12 and the filter 2 n2 + 3 n3 >= 11");
DEFINE_string(check_list, "",
              "comma-separated matrices (9-bit ints): only check this list "
              "against every capacity and report the first violation");
DEFINE_bool(symmetry, true,
            "orderly generation: keep only rank-one sets that are lex-minimal "
            "in their orbit under the stabiliser of W");
DEFINE_string(ones_subset, "",
              "testing: restrict the rank-one candidates to this comma-separated "
              "subset (the symmetry group is restricted to its setwise stabiliser)");
DEFINE_int32(threads, 1, "worker threads");
DEFINE_int32(split_depth, 0,
             "split each outer case into jobs by its first split_depth rank-one "
             "choices (0 = one job per case) so one case can use many threads");

namespace {

using u16 = uint16_t;

// Comma-separated unsigned integers (std::stoul; a malformed item throws).
std::vector<unsigned long> ParseUnsignedList(const std::string &csv) {
  std::vector<unsigned long> values;
  size_t pos = 0;
  while (pos < csv.size()) {
    size_t next = csv.find(',', pos);
    if (next == std::string::npos) next = csv.size();
    values.push_back(std::stoul(csv.substr(pos, next - pos)));
    pos = next + 1;
  }
  return values;
}
using u128 = __uint128_t;

struct U128Hash {
  size_t operator()(u128 k) const {
    uint64_t lo = static_cast<uint64_t>(k), hi = static_cast<uint64_t>(k >> 64);
    uint64_t h = lo * 0x9E3779B97F4A7C15ull ^ (hi + 0x7F4A7C159E3779B9ull);
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    return static_cast<size_t>(h ^ (h >> 32));
  }
};

// ---------------------------------------------------------------- matrices
// A 3x3 matrix over F_2 is 9 bits, bit 3*i + j = entry (row i, column j).

int Rank(u16 c) {
  int rows[3] = {c & 7, (c >> 3) & 7, (c >> 6) & 7};
  int r = 0;
  for (int bit = 0; bit < 3; ++bit) {
    int piv = -1;
    for (int k = r; k < 3; ++k) {
      if (rows[k] >> bit & 1) { piv = k; break; }
    }
    if (piv < 0) continue;
    std::swap(rows[r], rows[piv]);
    for (int k = 0; k < 3; ++k) {
      if (k != r && (rows[k] >> bit & 1)) rows[k] ^= rows[r];
    }
    ++r;
  }
  return r;
}

u16 Transpose(u16 u) {
  u16 t = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (u >> (3 * i + j) & 1) t |= u16{1} << (3 * j + i);
    }
  }
  return t;
}

u16 Mul(u16 a, u16 b) { // (a b)_{ij} = sum_k a_{ik} b_{kj}
  u16 c = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int s = 0;
      for (int k = 0; k < 3; ++k) s ^= (a >> (3 * i + k) & 1) & (b >> (3 * k + j) & 1);
      if (s) c |= u16{1} << (3 * i + j);
    }
  }
  return c;
}

int RowVec(int x, u16 u) { // x^T U as a 3-bit column-index vector
  int v = 0;
  for (int j = 0; j < 3; ++j) {
    int s = 0;
    for (int i = 0; i < 3; ++i) s ^= (x >> i & 1) & (u >> (3 * i + j) & 1);
    if (s) v |= 1 << j;
  }
  return v;
}

int ColVec(int y, u16 u) { // U y as a 3-bit row-index vector
  int v = 0;
  for (int i = 0; i < 3; ++i) {
    int s = 0;
    for (int j = 0; j < 3; ++j) s ^= (y >> j & 1) & (u >> (3 * i + j) & 1);
    if (s) v |= 1 << i;
  }
  return v;
}

// ------------------------------------------------------------- subspaces
// RREF with pivot = highest set bit, rows sorted descending; the same
// canonical form is applied to the table rows on load, so keys match.

int Rref(std::vector<u16> &rows) {
  int n = rows.size();
  int r = 0;
  for (int bit = 8; bit >= 0; --bit) {
    int piv = -1;
    for (int k = r; k < n; ++k) {
      if (rows[k] >> bit & 1) { piv = k; break; }
    }
    if (piv < 0) continue;
    std::swap(rows[r], rows[piv]);
    for (int k = 0; k < n; ++k) {
      if (k != r && (rows[k] >> bit & 1)) rows[k] ^= rows[r];
    }
    ++r;
  }
  rows.resize(r);
  std::sort(rows.begin(), rows.end(), std::greater<u16>());
  return r;
}

u128 KeyOfRref(const u16 *rows, int dim) {
  u128 key = dim;
  for (int i = 0; i < dim; ++i) key = (key << 9) | rows[i];
  return key;
}

bool InSpan(const u16 *rows, int dim, u16 v) {
  for (int i = 0; i < dim; ++i) {
    const int piv = 31 - __builtin_clz(rows[i]);
    if (v >> piv & 1) v ^= rows[i];
  }
  return v == 0;
}

struct CapTable {
  std::unordered_map<u128, uint8_t, U128Hash> lb; // key -> L(S)
  int L(u128 key) const {
    auto it = lb.find(key);
    CHECK(it != lb.end()) << "subspace missing from the table";
    return it->second;
  }
  int Get(u128 key) const { return FLAGS_r - L(key); } // the capacity
  int LOfGens(std::vector<u16> gens) const {
    const int d = Rref(gens);
    if (d == 0) return FLAGS_r;
    return L(KeyOfRref(gens.data(), d));
  }
};

void LoadTable(const std::string &path, CapTable *t) {
  FILE *f = std::fopen(path.c_str(), "rb");
  CHECK(f != nullptr) << "cannot open " << path;
  t->lb.reserve(8'400'000);
  uint8_t d, lb;
  u16 raw[9];
  size_t n = 0;
  while (std::fread(&d, 1, 1, f) == 1) {
    CHECK_EQ(std::fread(raw, 2, d, f), static_cast<size_t>(d));
    CHECK_EQ(std::fread(&lb, 1, 1, f), size_t{1});
    std::vector<u16> rows(raw, raw + d);
    const int dim = Rref(rows);
    CHECK_EQ(dim, d);
    if (dim == 0) continue;
    t->lb[KeyOfRref(rows.data(), dim)] = lb;
    ++n;
  }
  std::fclose(f);
  CHECK_EQ(n, size_t{8'283'457}) << "unexpected table size";
  LOG(INFO) << "capacity table: " << n << " subspaces";
}

// ------------------------------------------------------------- symmetry
using Perm = std::array<u16, 512>;

std::vector<Perm> BuildGroup() {
  std::vector<u16> gl;
  for (int a = 1; a < 512; ++a) {
    if (Rank(a) == 3) gl.push_back(a);
  }
  CHECK_EQ(gl.size(), size_t{168});
  std::vector<Perm> g;
  g.reserve(2 * 168 * 168);
  for (int t = 0; t < 2; ++t) {
    for (u16 a : gl) {
      for (u16 m : gl) {
        Perm p;
        for (int u = 0; u < 512; ++u) {
          const u16 v = t ? Transpose(static_cast<u16>(u)) : static_cast<u16>(u);
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
  for (size_t i = 0; i < s.size(); ++i) out[i] = p[s[i]];
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<u16> Canonical(const std::vector<Perm> &group, const std::vector<u16> &s) {
  std::vector<u16> best = Image(group[0], s);
  for (size_t i = 1; i < group.size(); ++i) {
    std::vector<u16> img = Image(group[i], s);
    if (img < best) best = std::move(img);
  }
  return best;
}

// ------------------------------------------------------------- the search
constexpr int kLatticeMaxDim = 6;
constexpr int kBlockCap = 7;  // m(B_x) <= 7, --prop3 excess lemma, r = 20 only
constexpr int kLineCap = 12;  // m(S_{x,c}) <= 12, --prop3 excess lemma, r = 20 only

struct Node {
  u128 key;
  uint8_t dim, cap, m;
  u16 rows[kLatticeMaxDim];
};

struct Undo {
  size_t first_new;
  std::vector<int> incremented;
};

class Search {
 public:
  Search(const CapTable &table, const std::vector<Perm> &group)
      : table_(table), group_(group) {
    for (int u = 1; u < 512; ++u) {
      for (int x = 1; x < 8; ++x) {
        rowvec_[x][u] = RowVec(x, u);
        colvec_[x][u] = ColVec(x, u);
      }
      if (Rank(u) == 1) ones_.push_back(u);
    }
    CHECK_EQ(ones_.size(), size_t{49});
    if (!FLAGS_ones_subset.empty()) {
      std::vector<u16> subset;
      for (unsigned long v : ParseUnsignedList(FLAGS_ones_subset)) {
        subset.push_back(static_cast<u16>(v));
      }
      std::sort(subset.begin(), subset.end());
      for (u16 v : subset) CHECK_EQ(Rank(v), 1) << v;
      ones_ = subset;
    }
    index_.reserve(1 << 16);
  }

  // Runs one outer case. Returns "INFEASIBLE", "FEASIBLE" or "OVER BUDGET".
  std::string Run(const std::vector<u16> &w, uint64_t budget) {
    if (!Prepare(w)) return "INFEASIBLE";
    return RunFrom({}, 0, budget);
  }

  // Runs the part of an outer case whose first rank-one choices are exactly
  // `prefix` (increasing), continuing from candidate index `start`.
  // Prepare(w) must have been called.
  std::string RunFrom(const std::vector<u16> &prefix, int start, uint64_t budget) {
    CHECK_EQ(chosen_.size(), base_depth_);
    budget_ = budget;
    nodes_visited_ = 0;
    solutions_.clear();
    stop_ = false;
    last_log_ = std::chrono::steady_clock::now();
    std::vector<Undo> undos(prefix.size());
    size_t added = 0;
    bool ok = true;
    for (u16 u : prefix) {
      if (!Add(u, &undos[added++])) { ok = false; break; }
    }
    if (ok) Dfs(start, need_ - static_cast<int>(prefix.size()));
    while (added > 0) UndoAdd(undos[--added]);
    if (stop_ && solutions_.empty()) return "OVER BUDGET";
    if (!solutions_.empty()) return "FEASIBLE";
    return "INFEASIBLE";
  }

  // The lex-minimal, capacity-feasible rank-one prefixes of length `depth`
  // (with the candidate index to continue from); together they partition the
  // search of the case. Prepare(w) must have been called.
  std::vector<std::pair<std::vector<u16>, int>> Prefixes(int depth) {
    CHECK_EQ(chosen_.size(), base_depth_);
    std::vector<std::pair<std::vector<u16>, int>> out;
    if (depth <= 0 || need_ <= depth) {
      out.push_back({{}, 0});
      return out;
    }
    std::vector<u16> prefix;
    PrefixDfs(0, depth, &prefix, &out);
    return out;
  }

  int need() const { return need_; }

  // Loads W: resets the state, computes Stab(W) and adds W's elements.
  // Returns false (with last_violation_ set) if W alone violates a capacity.
  bool Prepare(const std::vector<u16> &w) {
    Reset();
    need_ = FLAGS_r - static_cast<int>(w.size());
    last_log_ = std::chrono::steady_clock::now();
    // Stabiliser of W (setwise) in G: the symmetries available to the inner
    // search. Any solution (W, P) maps to the solution (W, h(P)), so it is
    // enough to enumerate the lex-minimal P of each orbit; lex-minimality is
    // prefix-closed for sets generated in increasing order, so non-minimal
    // prefixes can be pruned.
    stab_w_.clear();
    if (FLAGS_symmetry) {
      std::vector<u16> ws(w);
      std::sort(ws.begin(), ws.end());
      const bool restricted = ones_.size() != 49;
      for (const Perm &p : group_) {
        if (Image(p, ws) != ws) continue;
        if (restricted && Image(p, ones_) != ones_) continue;
        stab_w_.push_back(&p);
      }
      LOG(INFO) << "  |Stab(W)| = " << stab_w_.size();
    }
    for (u16 u : w) {
      Undo undo;
      if (!Add(u, &undo)) {
        LOG(INFO) << "W itself violates a capacity: " << last_violation_;
        return false;
      }
      // (never undone: W stays for the whole case)
    }
    base_depth_ = chosen_.size();
    return true;
  }

  const std::string &last_violation() const { return last_violation_; }

  // Checks one explicit list against every capacity (no search).
  bool Check(const std::vector<u16> &list) {
    Reset();
    for (u16 u : list) {
      Undo undo;
      if (!Add(u, &undo)) {
        LOG(ERROR) << "violation after adding " << u << ": " << last_violation_;
        return false;
      }
    }
    if (!LeafOk()) {
      LOG(ERROR) << "violation at dim 7/8: " << last_violation_;
      return false;
    }
    LOG(INFO) << "list of " << list.size() << " passes all capacities at r=" << FLAGS_r
              << " (lattice nodes: " << nodes_.size() << ")";
    return true;
  }

  uint64_t nodes_visited() const { return nodes_visited_; }
  const std::set<std::vector<u16>> &solutions() const { return solutions_; }

 private:
  void Reset() {
    nodes_.clear();
    index_.clear();
    chosen_.clear();
    for (int x = 0; x < 8; ++x) {
      cnt_row_[x] = cnt_col_[x] = 0;
      for (int c = 0; c < 8; ++c) cnt_line_row_[x][c] = cnt_line_col_[x][c] = 0;
    }
  }

  // Adds u to the chosen set, updating the lattice and the contextual
  // counters. Returns false on any violated capacity; the caller must Undo
  // in either case (the partial update is recorded in *undo).
  bool Add(u16 u, Undo *undo) {
    undo->first_new = nodes_.size();
    undo->incremented.clear();
    chosen_.push_back(u);
    bool ok = true;
    for (int x = 1; x < 8; ++x) {
      const int rv = rowvec_[x][u], cv = colvec_[x][u];
      if (rv == 0 && ++cnt_row_[x] > kBlockCap && FLAGS_prop3) {
        ok = false;
        last_violation_ = "contextual m(B_x) > 7, x=" + std::to_string(x);
      }
      if (cv == 0 && ++cnt_col_[x] > kBlockCap && FLAGS_prop3) {
        ok = false;
        last_violation_ = "contextual m(B^col_y) > 7, y=" + std::to_string(x);
      }
      for (int c = 1; c < 8; ++c) {
        if ((rv == 0 || rv == c) && ++cnt_line_row_[x][c] > kLineCap && FLAGS_prop3) {
          ok = false;
          last_violation_ = "contextual m(S_{x,c}) > 12";
        }
        if ((cv == 0 || cv == c) && ++cnt_line_col_[x][c] > kLineCap && FLAGS_prop3) {
          ok = false;
          last_violation_ = "contextual m(S^col_{y,z}) > 12";
        }
      }
    }
    if (!ok) return false;
    // Singleton span.
    {
      u16 r[1] = {u};
      if (!Create(r, 1, undo)) return false;
    }
    const size_t n0 = undo->first_new;
    for (size_t i = 0; i < n0; ++i) {
      Node &nd = nodes_[i];
      if (InSpan(nd.rows, nd.dim, u)) {
        ++nd.m;
        undo->incremented.push_back(static_cast<int>(i));
        if (nd.m > nd.cap) {
          Violation(nd.rows, nd.dim, nd.m, nd.cap);
          return false;
        }
      } else if (nd.dim < kLatticeMaxDim) {
        std::vector<u16> rows(nd.rows, nd.rows + nd.dim);
        rows.push_back(u);
        const int d = Rref(rows);
        CHECK_EQ(d, nd.dim + 1);
        if (!Create(rows.data(), d, undo)) return false;
      }
    }
    return true;
  }

  // Creates the lattice node for the RREF `rows` unless present; returns
  // false if its capacity is violated.
  bool Create(const u16 *rows, int dim, Undo *undo) {
    const u128 key = KeyOfRref(rows, dim);
    if (index_.count(key)) return true;
    Node nd;
    nd.key = key;
    nd.dim = dim;
    nd.cap = table_.Get(key);
    nd.m = 0;
    for (int i = 0; i < dim; ++i) nd.rows[i] = rows[i];
    for (u16 c : chosen_) {
      if (InSpan(nd.rows, dim, c)) ++nd.m;
    }
    index_.emplace(key, static_cast<int>(nodes_.size()));
    nodes_.push_back(nd);
    (void)undo;
    if (nd.m > nd.cap) {
      Violation(nd.rows, dim, nd.m, nd.cap);
      return false;
    }
    return true;
  }

  void Violation(const u16 *rows, int dim, int m, int cap) {
    last_violation_ = "dim " + std::to_string(dim) + " span{";
    for (int i = 0; i < dim; ++i) last_violation_ += (i ? "," : "") + std::to_string(rows[i]);
    last_violation_ += "} m=" + std::to_string(m) + " cap=" + std::to_string(cap) +
                       " (L=" + std::to_string(FLAGS_r - cap) + ")";
  }

  void UndoAdd(const Undo &undo) {
    const u16 u = chosen_.back();
    chosen_.pop_back();
    for (int x = 1; x < 8; ++x) {
      const int rv = rowvec_[x][u], cv = colvec_[x][u];
      if (rv == 0) --cnt_row_[x];
      if (cv == 0) --cnt_col_[x];
      for (int c = 1; c < 8; ++c) {
        if (rv == 0 || rv == c) --cnt_line_row_[x][c];
        if (cv == 0 || cv == c) --cnt_line_col_[x][c];
      }
    }
    for (int i : undo.incremented) --nodes_[i].m;
    for (size_t j = undo.first_new; j < nodes_.size(); ++j) index_.erase(nodes_[j].key);
    nodes_.resize(undo.first_new);
  }

  // Leaf: extend the lattice to dims 7 and 8 (temporarily) and check them.
  bool LeafOk() {
    std::unordered_map<u128, char, U128Hash> seen;
    std::vector<std::vector<u16>> frontier;
    for (const Node &nd : nodes_) {
      if (nd.dim == kLatticeMaxDim) frontier.emplace_back(nd.rows, nd.rows + nd.dim);
    }
    for (int dim = kLatticeMaxDim + 1; dim <= 8; ++dim) {
      std::vector<std::vector<u16>> next;
      for (const auto &base : frontier) {
        for (u16 c : chosen_) {
          if (InSpan(base.data(), base.size(), c)) continue;
          std::vector<u16> rows = base;
          rows.push_back(c);
          Rref(rows);
          const u128 key = KeyOfRref(rows.data(), rows.size());
          if (!seen.emplace(key, 1).second) continue;
          int m = 0;
          for (u16 v : chosen_) {
            if (InSpan(rows.data(), rows.size(), v)) ++m;
          }
          const int cap = table_.Get(key);
          if (m > cap) {
            Violation(rows.data(), rows.size(), m, cap);
            return false;
          }
          next.push_back(std::move(rows));
        }
      }
      frontier = std::move(next);
    }
    return true;
  }

  void Dfs(int idx, int need) {
    if (stop_) return;
    if (need == 0) {
      if (LeafOk()) {
        std::vector<u16> sol(chosen_);
        std::sort(sol.begin(), sol.end());
        const auto canon = Canonical(group_, sol);
        if (solutions_.insert(canon).second) {
          std::string s;
          for (u16 v : canon) s += " " + std::to_string(v);
          LOG(INFO) << "PROFILE #" << solutions_.size() << ":" << s;
          if (static_cast<int>(solutions_.size()) >= FLAGS_max_solutions) stop_ = true;
        }
      }
      return;
    }
    if (static_cast<int>(ones_.size()) - idx < need) return;
    if (!EnoughLive(idx, need)) return;
    // Include ones_[idx].
    {
      Undo undo;
      ++nodes_visited_;
      if (nodes_visited_ > budget_) {
        stop_ = true;
        return;
      }
      MaybeLog(idx);
      if (LexMinWith(ones_[idx])) {
        if (Add(ones_[idx], &undo)) Dfs(idx + 1, need - 1);
        UndoAdd(undo); // Add always pushes, so the undo is always due
      }
      if (stop_) return;
    }
    Dfs(idx + 1, need);
  }

  void PrefixDfs(int idx, int depth, std::vector<u16> *prefix,
                 std::vector<std::pair<std::vector<u16>, int>> *out) {
    if (static_cast<int>(prefix->size()) == depth) {
      out->push_back({*prefix, idx});
      return;
    }
    const int need = need_ - static_cast<int>(prefix->size());
    if (static_cast<int>(ones_.size()) - idx < need) return;
    if (!EnoughLive(idx, need)) return;
    if (LexMinWith(ones_[idx])) {
      Undo undo;
      if (Add(ones_[idx], &undo)) {
        prefix->push_back(ones_[idx]);
        PrefixDfs(idx + 1, depth, prefix, out);
        prefix->pop_back();
      }
      UndoAdd(undo);
    }
    PrefixDfs(idx + 1, depth, prefix, out);
  }

  // Every element of a saturated span (m == cap) is dead: adding it would
  // push that span over its capacity. Prune when the live candidates beyond
  // idx cannot supply `need` more elements.
  bool EnoughLive(int idx, int need) {
    uint64_t dead[8] = {};
    for (const Node &nd : nodes_) {
      if (nd.m < nd.cap) continue;
      const uint32_t lim = 1u << nd.dim;
      for (uint32_t mask = 1; mask < lim; ++mask) {
        u16 v = 0;
        for (int i = 0; i < nd.dim; ++i) {
          if (mask >> i & 1) v ^= nd.rows[i];
        }
        dead[v >> 6] |= uint64_t{1} << (v & 63);
      }
    }
    int live = 0;
    for (size_t i = idx; i < ones_.size(); ++i) {
      const u16 v = ones_[i];
      if (!(dead[v >> 6] >> (v & 63) & 1) && ++live >= need) return true;
    }
    return live >= need;
  }

  // Is the rank-one part of the chosen set plus u (which exceeds all of it)
  // lex-minimal in its Stab(W)-orbit? Elements are compared as sorted lists.
  bool LexMinWith(u16 u) {
    if (stab_w_.empty()) return true;
    const size_t n0 = base_depth_;
    const size_t k = chosen_.size() - n0 + 1;
    u16 p[32], img[32];
    for (size_t i = 0; i + 1 < k; ++i) p[i] = chosen_[n0 + i];
    p[k - 1] = u; // chosen_ is increasing beyond base_depth_, so p is sorted
    for (const Perm *h : stab_w_) {
      // Quick test on the minimum of the image.
      u16 mn = 0x3FF;
      for (size_t i = 0; i < k; ++i) {
        const u16 v = (*h)[p[i]];
        if (v < mn) mn = v;
      }
      if (mn > p[0]) continue;
      if (mn < p[0]) return false;
      for (size_t i = 0; i < k; ++i) img[i] = (*h)[p[i]];
      std::sort(img, img + k);
      for (size_t i = 0; i < k; ++i) {
        if (img[i] < p[i]) return false;
        if (img[i] > p[i]) break;
      }
    }
    return true;
  }

  void MaybeLog(int idx) {
    if ((nodes_visited_ & 0xFFF) != 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_).count() <
        FLAGS_log_seconds) {
      return;
    }
    last_log_ = now;
    std::string prefix;
    for (size_t i = base_depth_; i < chosen_.size(); ++i) {
      prefix += " " + std::to_string(chosen_[i]);
    }
    LOG(INFO) << "  ... nodes=" << nodes_visited_ << " depth=" << chosen_.size() - base_depth_
              << " idx=" << idx << " lattice=" << nodes_.size() << " prefix:" << prefix;
  }

  const CapTable &table_;
  const std::vector<Perm> &group_;
  std::vector<u16> ones_;
  int rowvec_[8][512], colvec_[8][512];
  std::vector<Node> nodes_;
  std::unordered_map<u128, int, U128Hash> index_;
  std::vector<u16> chosen_;
  int cnt_row_[8], cnt_col_[8], cnt_line_row_[8][8], cnt_line_col_[8][8];
  size_t base_depth_ = 0;
  int need_ = 0;
  uint64_t budget_ = 0, nodes_visited_ = 0;
  bool stop_ = false;
  std::set<std::vector<u16>> solutions_;
  std::chrono::steady_clock::time_point last_log_;
  std::string last_violation_;
  std::vector<const Perm *> stab_w_;
};

// --------------------------------------------------------- outer cases
struct OuterCase {
  std::vector<u16> w; // canonical under the stabiliser of B0
  int n2, n3;
};

std::vector<OuterCase> EnumerateOuter(const std::vector<Perm> &stab) {
  std::set<std::vector<u16>> seen;
  std::vector<OuterCase> cases;
  if (FLAGS_min_w == 0) cases.push_back({{}, 0, 0}); // all r factors rank-one
  for (int cid = 1; cid < 64; ++cid) { // coset (rows 2-3 of U); cid 0 is B0 itself
    std::vector<u16> cands;
    for (int r1 = 0; r1 < 8; ++r1) {
      const u16 u = static_cast<u16>(cid << 3 | r1);
      if (Rank(u) >= 2) cands.push_back(u);
    }
    const int n = cands.size();
    for (uint32_t mask = 1; mask < (1u << n); ++mask) {
      const int k = __builtin_popcount(mask);
      if (k < FLAGS_min_w || k > FLAGS_max_w) continue;
      std::vector<u16> w;
      int n2 = 0, n3 = 0;
      for (int i = 0; i < n; ++i) {
        if (mask >> i & 1) {
          w.push_back(cands[i]);
          (Rank(cands[i]) == 2 ? n2 : n3)++;
        }
      }
      if (FLAGS_prop3 && 2 * n2 + 3 * n3 < 11) continue; // --prop3 excess lemma
      const auto canon = Canonical(stab, w);
      if (seen.insert(canon).second) cases.push_back({canon, n2, n3});
    }
  }
  std::sort(cases.begin(), cases.end(), [](const OuterCase &a, const OuterCase &b) {
    if (a.w.size() != b.w.size()) return a.w.size() < b.w.size();
    return a.w < b.w;
  });
  return cases;
}

void SelfTest(const CapTable &table, const std::vector<Perm> &group) {
  // Known capacities: rank-one block (L = 15), B_x (L = 9), S_{x,c} (L = 6),
  // singletons (L = 19).
  CHECK_EQ(table.LOfGens({0x1, 0x2, 0x4}), 15);
  CHECK_EQ(table.LOfGens({0x1, 0x8, 0x40}), 15);
  CHECK_EQ(table.LOfGens({0x8, 0x10, 0x20, 0x40, 0x80, 0x100}), 9);
  CHECK_EQ(table.LOfGens({0x1, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100}), 6);
  CHECK_EQ(table.LOfGens({0x1}), 19);
  CHECK_EQ(table.LOfGens({0x1FF}), 19);
  // Dim-2 dichotomy: L = 19 iff no rank-one element.
  int checked = 0;
  for (int a = 1; a < 512 && checked < 3000; a += 7) {
    for (int b = a + 1; b < 512 && checked < 3000; b += 11) {
      const bool has_one = Rank(a) == 1 || Rank(b) == 1 || Rank(a ^ b) == 1;
      CHECK_EQ(table.LOfGens({static_cast<u16>(a), static_cast<u16>(b)}), has_one ? 18 : 19);
      ++checked;
    }
  }
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
    for (int i = 0; i < k; ++i) gens.push_back(1 + rng() % 511);
    const Perm &p = group[rng() % group.size()];
    std::vector<u16> img;
    for (u16 v : gens) img.push_back(p[v]);
    CHECK_EQ(table.LOfGens(gens), table.LOfGens(img));
  }
  LOG(INFO) << "self-test passed";
}

// " v1 v2 ..." (a space before every value): the format of every printed list.
std::string SpaceJoined(const std::vector<u16> &v) {
  std::string s;
  for (u16 x : v) s += " " + std::to_string(x);
  return s;
}

// --check_list: one explicit list against every capacity. Returns the exit code.
int RunCheckList(const CapTable &table, const std::vector<Perm> &group,
                 const std::vector<u16> &list) {
  Search search(table, group);
  const bool ok = search.Check(list);
  std::printf("check_list: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}

// Stabiliser of B0 = span{1, 2, 4} (matrices supported on row 1).
std::vector<Perm> StabiliserOfB0(const std::vector<Perm> &group) {
  std::vector<Perm> stab;
  for (const Perm &p : group) {
    if (p[1] < 8 && p[2] < 8 && p[4] < 8) stab.push_back(p);
  }
  CHECK_EQ(stab.size(), size_t{4032});
  return stab;
}

// " |W|=k:n" for every k in [--min_w, --max_w].
std::string OuterCaseHistogram(const std::vector<OuterCase> &cases) {
  std::array<int, 9> hist{};
  for (const auto &c : cases) ++hist[c.w.size()];
  std::string h;
  for (int k = FLAGS_min_w; k <= FLAGS_max_w; ++k) {
    h += " |W|=" + std::to_string(k) + ":" + std::to_string(hist[k]);
  }
  return h;
}

// --list_only.
void PrintCaseList(const std::vector<OuterCase> &cases) {
  for (size_t i = 0; i < cases.size(); ++i) {
    std::printf("case %zu |W|=%zu n2=%d n3=%d:%s\n", i, cases[i].w.size(), cases[i].n2,
                cases[i].n3, SpaceJoined(cases[i].w).c_str());
  }
}

// --cases, or every case; each index must exist.
std::vector<size_t> SelectCases(size_t num_cases) {
  std::vector<size_t> selected;
  if (FLAGS_cases.empty()) {
    for (size_t i = 0; i < num_cases; ++i) selected.push_back(i);
  } else {
    for (unsigned long i : ParseUnsignedList(FLAGS_cases)) selected.push_back(i);
  }
  for (size_t i : selected) CHECK_LT(i, num_cases);
  return selected;
}

// Jobs: (case, forced rank-one prefix, continuation index). A case whose W
// already violates a capacity gets no job and is INFEASIBLE outright.
struct Job {
  size_t sel; // index into `selected`
  std::vector<u16> prefix;
  int start;
};
struct Result {
  bool w_violates = false;
  int jobs = 0, done = 0, over_budget = 0;
  uint64_t nodes = 0;
  std::set<std::vector<u16>> profiles;
  double secs = 0;
};

// The jobs of the selected cases (the prefixes at --split_depth), recording the
// per-case job counts and W violations in *results.
std::vector<Job> BuildJobs(const CapTable &table, const std::vector<Perm> &group,
                           const std::vector<OuterCase> &cases,
                           const std::vector<size_t> &selected,
                           std::vector<Result> *results) {
  std::vector<Job> jobs;
  Search search(table, group);
  for (size_t j = 0; j < selected.size(); ++j) {
    const auto &c = cases[selected[j]];
    const std::string s = SpaceJoined(c.w);
    if (!search.Prepare(c.w)) {
      (*results)[j].w_violates = true;
      LOG(INFO) << "case " << selected[j] << " |W|=" << c.w.size() << " W:" << s
                << " -> INFEASIBLE (W violates " << search.last_violation() << ")";
      continue;
    }
    const auto prefixes = search.Prefixes(FLAGS_split_depth);
    for (const auto &p : prefixes) jobs.push_back({j, p.first, p.second});
    (*results)[j].jobs = prefixes.size();
    LOG(INFO) << "case " << selected[j] << " |W|=" << c.w.size() << " n2=" << c.n2
              << " n3=" << c.n3 << " W:" << s << " -> " << prefixes.size() << " job(s)";
  }
  return jobs;
}

// The worker pool: every worker owns one Search, re-prepares it when a job's
// case changes, and records each job's result under the mutex.
struct JobPool {
  const CapTable &table;
  const std::vector<Perm> &group;
  const std::vector<OuterCase> &cases;
  const std::vector<size_t> &selected;
  const std::vector<Job> &jobs;
  std::vector<Result> *results;
  std::mutex mu;
  std::atomic<size_t> next_job{0};
  const std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

  void Record(size_t k, const Job &job, const std::string &verdict, double secs,
              const Search &search) {
    std::lock_guard<std::mutex> lock(mu);
    Result &r = (*results)[job.sel];
    ++r.done;
    r.nodes += search.nodes_visited();
    r.secs += secs;
    if (verdict == "OVER BUDGET") ++r.over_budget;
    for (const auto &sol : search.solutions()) r.profiles.insert(sol);
    if (r.done == r.jobs) {
      LOG(INFO) << "case " << selected[job.sel] << " complete: nodes=" << r.nodes
                << " profiles=" << r.profiles.size() << " over_budget_jobs=" << r.over_budget
                << " cpu=" << r.secs << "s";
    } else if (jobs.size() > 1 && (k % 64 == 0)) {
      LOG(INFO) << "  progress: " << k << "/" << jobs.size() << " jobs dispatched, "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                       .count()
                << "s elapsed";
    }
  }

  void Worker() {
    Search search(table, group);
    size_t prepared = SIZE_MAX;
    while (true) {
      const size_t k = next_job.fetch_add(1);
      if (k >= jobs.size()) break;
      const Job &job = jobs[k];
      if (prepared != job.sel) {
        CHECK(search.Prepare(cases[selected[job.sel]].w));
        prepared = job.sel;
      }
      const auto t0 = std::chrono::steady_clock::now();
      const std::string verdict = search.RunFrom(job.prefix, job.start, FLAGS_budget);
      const double secs =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      Record(k, job, verdict, secs, search);
    }
  }

  // nthreads workers including the calling thread.
  void Run(int nthreads) {
    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t) pool.emplace_back([this] { Worker(); });
    Worker();
    for (auto &th : pool) th.join();
  }
};

// The per-case verdict lines with their profiles on stdout, then the summary
// (stdout flushed first, then the LOG line, then the printed line).
int PrintSummary(const std::vector<OuterCase> &cases, const std::vector<size_t> &selected,
                 const std::vector<Result> &results) {
  int feasible = 0, infeasible = 0, over = 0;
  for (size_t j = 0; j < selected.size(); ++j) {
    const Result &r = results[j];
    std::string verdict = "INFEASIBLE";
    if (!r.profiles.empty()) verdict = "FEASIBLE";
    else if (r.over_budget > 0) verdict = "OVER BUDGET";
    std::printf("case %zu |W|=%zu: %s jobs=%d nodes=%llu profiles=%zu cpu=%.1fs%s\n",
                selected[j], cases[selected[j]].w.size(), verdict.c_str(), r.jobs,
                static_cast<unsigned long long>(r.nodes), r.profiles.size(), r.secs,
                r.w_violates ? " (W violates)" : "");
    for (const auto &sol : r.profiles) {
      std::printf("  profile:%s\n", SpaceJoined(sol).c_str());
    }
    if (verdict == "FEASIBLE") ++feasible;
    else if (verdict == "INFEASIBLE") ++infeasible;
    else ++over;
  }
  std::fflush(stdout);
  LOG(INFO) << "summary: feasible=" << feasible << " infeasible=" << infeasible
            << " over_budget=" << over << " of " << selected.size();
  std::printf("summary: feasible=%d infeasible=%d over_budget=%d of %zu\n", feasible,
              infeasible, over, selected.size());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);
  CHECK(!FLAGS_bin.empty()) << "--bin is required";
  if (FLAGS_r != 20 && FLAGS_prop3) {
    LOG(WARNING) << "the --prop3 excess lemma is specific to r = 20; disabling it";
    FLAGS_prop3 = false;
  }
  LOG(INFO) << "r=" << FLAGS_r << " prop3=" << (FLAGS_prop3 ? "on" : "off")
            << " |W| in [" << FLAGS_min_w << "," << FLAGS_max_w << "]";

  CapTable table;
  LoadTable(FLAGS_bin, &table);
  const std::vector<Perm> group = BuildGroup();
  CHECK_EQ(group.size(), size_t{56'448});
  SelfTest(table, group);

  if (!FLAGS_check_list.empty()) {
    std::vector<u16> list;
    for (unsigned long v : ParseUnsignedList(FLAGS_check_list)) {
      list.push_back(static_cast<u16>(v));
    }
    return RunCheckList(table, group, list);
  }

  const std::vector<Perm> stab = StabiliserOfB0(group);
  const std::vector<OuterCase> cases = EnumerateOuter(stab);
  LOG(INFO) << "outer cases (up to the stabiliser of B0): " << cases.size()
            << OuterCaseHistogram(cases);
  if (FLAGS_list_only) {
    PrintCaseList(cases);
    return 0;
  }

  const std::vector<size_t> selected = SelectCases(cases.size());
  std::vector<Result> results(selected.size());
  const std::vector<Job> jobs = BuildJobs(table, group, cases, selected, &results);
  LOG(INFO) << jobs.size() << " jobs over " << selected.size() << " cases";

  JobPool pool{table, group, cases, selected, jobs, &results};
  pool.Run(std::max(1, std::min<int>(FLAGS_threads, jobs.size())));
  return PrintSummary(cases, selected, results);
}
