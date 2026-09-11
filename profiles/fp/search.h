#pragma once

#include "profiles/result.h"

// Generic exhaustive profile enumerator for <N0,N1,*> matrix multiplication
// over a prime field F_P (the field-independent core of
// profile_enum_q02_n333_main.cc, without the <3,3,3>-specific sunflower
// reduction).
//
// Problem P(r): a multiset F of r nonzero first factors, taken projectively
// (a factor and its nonzero scalar multiples are the same point of
// PG(NA-1, P), NA = N0*N1), such that for every subspace S of A* = F_P^NA
//   m_F(S) := #{terms with factor in S} <= cap(S) := r - L(S),
// where L(S) is the certified lower bound of the tensor restricted to the
// annihilator of S (expanded from the certificate in memory, rows as base-P
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

#include "profiles/fp/problem_data.h"
#include "profiles/options.h"
#include <chrono>
#include <set>
#include <string>
#include <unordered_map>

namespace profiles::fp {

template <int P, int N0, int N1> class Search {
public:
  using G = Geometry<P, N0, N1>;
  using Digits = typename G::Digits;
  using Span = typename G::Span;
  static constexpr int NA = G::NA, Q = G::Q, kMaxDim = G::kMaxDim;
  Search(const ProblemData<P, N0, N1> &data, const profiles::Options &options)
      : data_(data), options_(options), need_(options.r) {
    ValidateOptions(options);
  }
  uint64_t nodes_visited() const { return nodes_visited_; }
  const auto &solutions() const { return solutions_; }
  const auto &data() const { return data_; }
  int Cap(const Span &s) const { return options_.r - data_.L(s); }

private:
  const ProblemData<P, N0, N1> &data_;
  const profiles::Options options_;
  // ------------------------------------------------------------ search
  struct Node {
    uint64_t key = 0;
    uint8_t cap = 0;
    uint8_t m = 0;
    Span span;
  };
  struct Undo {
    size_t first_new = 0;
    std::vector<int> incremented;
  };

  std::vector<int> cand_; // candidate point indices (ascending)
  std::vector<int> cap1_; // cap of <point> per candidate position
  std::vector<Node> nodes_;
  std::unordered_map<uint64_t, int, U64Hash> index_;
  std::vector<int> chosen_; // candidate positions, non-decreasing
  std::vector<int> mult_;   // multiplicity per candidate position
  int need_ = 0;
  uint64_t budget_ = 0;
  uint64_t nodes_visited_ = 0;
  StopReason stop_reason_ = StopReason::kExhausted;
  std::set<std::vector<int>> solutions_;
  std::chrono::steady_clock::time_point last_log_;
  std::string last_violation_;
  std::vector<const std::vector<u16> *>
      group_; // perms used for lex-min (may be restricted)
  bool restricted_ = false;

public:
  void SetupCandidates(const std::vector<uint32_t> &subset_codes) {
    ValidateCodes(subset_codes, Q);
    cand_.clear();
    if (subset_codes.empty()) {
      for (int i = 0; i < data_.npoints_; ++i)
        cand_.push_back(i);
      restricted_ = false;
    } else {
      std::set<int> s;
      for (uint32_t c : subset_codes) {
        CHECK_LT(c, static_cast<uint32_t>(Q));
        CHECK_GE(data_.point_of_code_[c], 0);
        s.insert(data_.point_of_code_[c]);
      }
      cand_.assign(s.begin(), s.end());
      restricted_ = true;
    }
    cap1_.clear();
    for (int i : cand_)
      cap1_.push_back(Cap(G::Rref({data_.point_digits_[i]})));
    group_.clear();
    if (options_.symmetry) {
      for (const auto &h : data_.perms_) {
        if (restricted_) {
          bool ok = true;
          for (int i : cand_) {
            if (!std::binary_search(cand_.begin(), cand_.end(),
                                    static_cast<int>(h[i]))) {
              ok = false;
              break;
            }
          }
          if (!ok)
            continue;
        }
        group_.push_back(&h);
      }
      LOG(INFO) << "candidates: " << cand_.size()
                << ", symmetry group used: " << group_.size();
    }
  }

private:
  void Reset() {
    nodes_.clear();
    index_.clear();
    chosen_.clear();
    mult_.assign(cand_.size(), 0);
  }

  void Violation(const Span &s, int m, int cap) {
    last_violation_ = "dim " + std::to_string(s.dim) + " span{";
    for (int i = 0; i < s.dim; ++i) {
      last_violation_ += (i ? "," : "") + std::to_string(G::Encode(s.rows[i]));
    }
    last_violation_ += "} m=" + std::to_string(m) +
                       " cap=" + std::to_string(cap) +
                       " (L=" + std::to_string(options_.r - cap) + ")";
  }

  bool Create(const Span &s, Undo *undo) {
    (void)undo;
    const uint64_t key = G::Key(s);
    if (index_.count(key))
      return true;
    Node nd;
    nd.key = key;
    nd.span = s;
    nd.cap = static_cast<uint8_t>(std::max(0, Cap(s)));
    nd.m = 0;
    for (int pos : chosen_) {
      if (G::InSpan(s, data_.point_digits_[cand_[pos]]))
        ++nd.m;
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
    const Digits &u = data_.point_digits_[cand_[pos]];
    const size_t n0 = undo->first_new;
    bool singleton_seen = false;
    for (size_t i = 0; i < n0; ++i) {
      Node &nd = nodes_[i];
      if (G::InSpan(nd.span, u)) {
        if (nd.span.dim == 1)
          singleton_seen = true;
        ++nd.m;
        undo->incremented.push_back(static_cast<int>(i));
        if (nd.m > nd.cap) {
          Violation(nd.span, nd.m, nd.cap);
          return false;
        }
      } else if (nd.span.dim < kMaxDim) {
        if (!Create(G::Join(nd.span, u), undo))
          return false;
      }
    }
    if (!singleton_seen && !Create(G::Rref({u}), undo))
      return false;
    return true;
  }

  void UndoAdd(const Undo &undo) {
    --mult_[chosen_.back()];
    chosen_.pop_back();
    for (int i : undo.incremented)
      --nodes_[i].m;
    for (size_t j = undo.first_new; j < nodes_.size(); ++j)
      index_.erase(nodes_[j].key);
    nodes_.resize(undo.first_new);
  }

  bool EnoughLive(int idx, int need) {
    std::vector<char> dead(cand_.size(), 0);
    std::vector<int> pts;
    for (const Node &nd : nodes_) {
      if (nd.m < nd.cap)
        continue;
      data_.SpanPoints(nd.span, &pts);
      for (int p : pts) {
        const auto it = std::lower_bound(cand_.begin(), cand_.end(), p);
        if (it != cand_.end() && *it == p)
          dead[it - cand_.begin()] = 1;
      }
    }
    int live = 0;
    for (size_t i = idx; i < cand_.size(); ++i) {
      if (!dead[i]) {
        live += cap1_[i] - mult_[i];
        if (live >= need)
          return true;
      }
    }
    return live >= need;
  }

  // Is chosen_ (as sorted point list) plus point pos lex-minimal under G?
  bool LexMinWith(int pos) {
    if (group_.empty())
      return true;
    const size_t k = chosen_.size() + 1;
    int p[64] = {};
    int img[64] = {};
    for (size_t i = 0; i + 1 < k; ++i)
      p[i] = cand_[chosen_[i]];
    p[k - 1] = cand_[pos];
    const int p1 = p[0];
    auto test = [&](const std::vector<u16> &h) {
      for (size_t i = 0; i < k; ++i)
        img[i] = h[p[i]];
      std::sort(img, img + k);
      for (size_t i = 0; i < k; ++i) {
        if (img[i] < p[i])
          return false;
        if (img[i] > p[i])
          return true;
      }
      return true;
    };
    if (restricted_ || p1 >= data_.tmax_) {
      for (const auto *h : group_) {
        if (!test(*h))
          return false;
      }
      return true;
    }
    // h(F) < F requires h(q) <= p1 for some q in F.
    int last_q = -1;
    for (size_t i = 0; i < k; ++i) {
      const int q = p[i];
      if (q == last_q)
        continue;
      last_q = q;
      for (int t = 0; t <= p1; ++t) {
        for (uint32_t h : data_.transporter_[q][t]) {
          if (!test(data_.perms_[h]))
            return false;
        }
      }
    }
    return true;
  }

  void Record() {
    std::vector<int> sol;
    for (int pos : chosen_)
      sol.push_back(cand_[pos]);
    std::sort(sol.begin(), sol.end());
    std::vector<int> best = sol;
    // Output is normalized even when prefix symmetry pruning is disabled.
    std::vector<int> img(sol.size());
    for (const auto &h : data_.perms_) {
      for (size_t i = 0; i < sol.size(); ++i)
        img[i] = h[sol[i]];
      std::sort(img.begin(), img.end());
      if (img < best)
        best = img;
    }
    if (solutions_.insert(best).second) {
      std::string s;
      for (int p : best)
        s += " " + std::to_string(data_.point_code_[p]);
      LOG(INFO) << "PROFILE #" << solutions_.size() << " (point codes):" << s;
      if (static_cast<int>(solutions_.size()) >= options_.max_solutions)
        stop_reason_ = StopReason::kSolutionLimit;
    }
  }

  void Dfs(int idx, int need) {
    if (stop_reason_ != StopReason::kExhausted)
      return;
    if (need == 0) {
      Record();
      return;
    }
    if (idx >= static_cast<int>(cand_.size()))
      return;
    if (!EnoughLive(idx, need))
      return;
    if (mult_[idx] < cap1_[idx]) {
      ++nodes_visited_;
      if (nodes_visited_ > budget_) {
        stop_reason_ = StopReason::kNodeBudget;
        return;
      }
      MaybeLog(idx);
      if (LexMinWith(idx)) {
        Undo undo;
        if (Add(idx, &undo))
          Dfs(idx, need - 1); // stay: repeats allowed
        UndoAdd(undo);
      }
      if (stop_reason_ != StopReason::kExhausted)
        return;
    }
    Dfs(idx + 1, need);
  }

  void MaybeLog(int idx) {
    if ((nodes_visited_ & 0x3FF) != 0)
      return;
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_)
            .count() < options_.log_seconds) {
      return;
    }
    last_log_ = now;
    std::string prefix;
    for (int pos : chosen_)
      prefix += " " + std::to_string(data_.point_code_[cand_[pos]]);
    LOG(INFO) << "  ... nodes=" << nodes_visited_ << " depth=" << chosen_.size()
              << " idx=" << idx << " lattice=" << nodes_.size()
              << " prefix:" << prefix;
  }

public:
  SearchResult<int> RunFrom(const std::vector<int> &prefix, int start,
                            uint64_t budget) {
    Reset();
    budget_ = budget;
    nodes_visited_ = 0;
    solutions_.clear();
    stop_reason_ = StopReason::kExhausted;
    last_log_ = std::chrono::steady_clock::now();
    std::vector<Undo> undos(prefix.size());
    size_t added = 0;
    bool ok = true;
    for (int pos : prefix) {
      if (!Add(pos, &undos[added++])) {
        ok = false;
        break;
      }
    }
    if (ok)
      Dfs(start, need_ - static_cast<int>(prefix.size()));
    while (added > 0)
      UndoAdd(undos[--added]);
    return {stop_reason_, nodes_visited_, solutions_};
  }

private:
  void PrefixDfs(int idx, int depth, std::vector<int> *prefix,
                 std::vector<std::pair<std::vector<int>, int>> *out) {
    if (static_cast<int>(prefix->size()) == depth) {
      out->push_back({*prefix, idx});
      return;
    }
    const int need = need_ - static_cast<int>(prefix->size());
    if (idx >= static_cast<int>(cand_.size()))
      return;
    if (!EnoughLive(idx, need))
      return;
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

public:
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

public:
  bool Check(const std::vector<uint32_t> &codes) {
    Reset();
    for (uint32_t c : codes) {
      CHECK_LT(c, static_cast<uint32_t>(Q));
      const int p = data_.point_of_code_[c];
      CHECK_GE(p, 0) << "zero factor";
      const auto it = std::lower_bound(cand_.begin(), cand_.end(), p);
      CHECK(it != cand_.end() && *it == p);
      Undo undo;
      if (!Add(it - cand_.begin(), &undo)) {
        LOG(ERROR) << "violation after adding code " << c << ": "
                   << last_violation_;
        return false;
      }
    }
    LOG(INFO) << "list of " << codes.size()
              << " passes all capacities at r=" << options_.r
              << " (lattice nodes: " << nodes_.size() << ")";
    return true;
  }
};

} // namespace profiles::fp
