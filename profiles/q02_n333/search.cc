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
// stabilizer of B0 in the symmetry group G = {U -> A U M, U -> A U^T M}
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
#include "profiles/q02_n333/search.h"
#include <algorithm>

namespace profiles::q02_n333 {

Search::Search(const CapTable &table, const std::vector<Perm> &group,
               const Options &options)
    : table_(table), options_(options), group_(group) {
  profiles::ValidateOptions(options, 32);
  ValidateCodes(options.ones_subset, 512);
  ValidateCodes(options.check_list, 512);
  for (int u = 1; u < 512; ++u) {
    for (int x = 1; x < 8; ++x) {
      rowvec_[x][u] = RowVec(x, u);
      colvec_[x][u] = ColVec(x, u);
    }
    if (Rank(u) == 1)
      ones_.push_back(u);
  }
  CHECK_EQ(ones_.size(), size_t{49});
  if (!options_.ones_subset.empty()) {
    std::vector<u16> subset;
    for (unsigned long v : options_.ones_subset) {
      subset.push_back(static_cast<u16>(v));
    }
    std::sort(subset.begin(), subset.end());
    for (u16 v : subset)
      CHECK_EQ(Rank(v), 1) << v;
    ones_ = subset;
  }
  index_.reserve(1 << 16);
}

SearchResult<u16> Search::Run(const std::vector<u16> &w, uint64_t budget) {
  if (!Prepare(w))
    return {};
  return RunFrom({}, 0, budget);
}

SearchResult<u16> Search::RunFrom(const std::vector<u16> &prefix, int start,
                                  uint64_t budget) {
  CHECK_EQ(chosen_.size(), base_depth_);
  budget_ = budget;
  nodes_visited_ = 0;
  solutions_.clear();
  stop_reason_ = StopReason::kExhausted;
  last_log_ = std::chrono::steady_clock::now();
  std::vector<Undo> undos(prefix.size());
  size_t added = 0;
  bool ok = true;
  for (u16 u : prefix) {
    if (!Add(u, &undos[added++])) {
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

std::vector<std::pair<std::vector<u16>, int>> Search::Prefixes(int depth) {
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

int Search::need() const { return need_; }

bool Search::Prepare(const std::vector<u16> &w) {
  Reset();
  need_ = options_.r - static_cast<int>(w.size());
  last_log_ = std::chrono::steady_clock::now();
  // Stabilizer of W (setwise) in G: the symmetries available to the inner
  // search. Any solution (W, P) maps to the solution (W, h(P)), so it is
  // enough to enumerate the lex-minimal P of each orbit; lex-minimality is
  // prefix-closed for sets generated in increasing order, so non-minimal
  // prefixes can be pruned.
  stab_w_.clear();
  if (options_.symmetry) {
    std::vector<u16> ws(w);
    std::sort(ws.begin(), ws.end());
    const bool restricted = ones_.size() != 49;
    for (const Perm &p : group_) {
      if (Image(p, ws) != ws)
        continue;
      if (restricted && Image(p, ones_) != ones_)
        continue;
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

const std::string &Search::last_violation() const { return last_violation_; }

bool Search::Check(const std::vector<u16> &list) {
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
  LOG(INFO) << "list of " << list.size()
            << " passes all capacities at r=" << options_.r
            << " (lattice nodes: " << nodes_.size() << ")";
  return true;
}

uint64_t Search::nodes_visited() const { return nodes_visited_; }

const std::set<std::vector<u16>> &Search::solutions() const {
  return solutions_;
}

void Search::Dfs(int idx, int need) {
  if (stop_reason_ != StopReason::kExhausted)
    return;
  if (need == 0) {
    if (LeafOk()) {
      std::vector<u16> sol(chosen_);
      std::sort(sol.begin(), sol.end());
      const auto canon = Canonical(group_, sol);
      if (solutions_.insert(canon).second) {
        std::string s;
        for (u16 v : canon)
          s += " " + std::to_string(v);
        LOG(INFO) << "PROFILE #" << solutions_.size() << ":" << s;
        if (static_cast<int>(solutions_.size()) >= options_.max_solutions)
          stop_reason_ = StopReason::kSolutionLimit;
      }
    }
    return;
  }
  if (static_cast<int>(ones_.size()) - idx < need)
    return;
  if (!EnoughLive(idx, need))
    return;
  // Include ones_[idx].
  {
    Undo undo;
    ++nodes_visited_;
    if (nodes_visited_ > budget_) {
      stop_reason_ = StopReason::kNodeBudget;
      return;
    }
    MaybeLog(idx);
    if (LexMinWith(ones_[idx])) {
      if (Add(ones_[idx], &undo))
        Dfs(idx + 1, need - 1);
      UndoAdd(undo); // Add always pushes, so the undo is always due
    }
    if (stop_reason_ != StopReason::kExhausted)
      return;
  }
  Dfs(idx + 1, need);
}

void Search::PrefixDfs(int idx, int depth, std::vector<u16> *prefix,
                       std::vector<std::pair<std::vector<u16>, int>> *out) {
  if (static_cast<int>(prefix->size()) == depth) {
    out->push_back({*prefix, idx});
    return;
  }
  const int need = need_ - static_cast<int>(prefix->size());
  if (static_cast<int>(ones_.size()) - idx < need)
    return;
  if (!EnoughLive(idx, need))
    return;
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

bool Search::LexMinWith(u16 u) {
  if (stab_w_.empty())
    return true;
  const size_t n0 = base_depth_;
  const size_t k = chosen_.size() - n0 + 1;
  u16 p[32] = {};
  u16 img[32] = {};
  for (size_t i = 0; i + 1 < k; ++i)
    p[i] = chosen_[n0 + i];
  p[k - 1] = u; // chosen_ is increasing beyond base_depth_, so p is sorted
  for (const Perm *h : stab_w_) {
    // Quick test on the minimum of the image.
    u16 mn = 0x3FF;
    for (size_t i = 0; i < k; ++i) {
      const u16 v = (*h)[p[i]];
      if (v < mn)
        mn = v;
    }
    if (mn > p[0])
      continue;
    if (mn < p[0])
      return false;
    for (size_t i = 0; i < k; ++i)
      img[i] = (*h)[p[i]];
    std::sort(img, img + k);
    for (size_t i = 0; i < k; ++i) {
      if (img[i] < p[i])
        return false;
      if (img[i] > p[i])
        break;
    }
  }
  return true;
}

void Search::MaybeLog(int idx) {
  if ((nodes_visited_ & 0xFFF) != 0)
    return;
  const auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_)
          .count() < options_.log_seconds) {
    return;
  }
  last_log_ = now;
  std::string prefix;
  for (size_t i = base_depth_; i < chosen_.size(); ++i) {
    prefix += " " + std::to_string(chosen_[i]);
  }
  LOG(INFO) << "  ... nodes=" << nodes_visited_
            << " depth=" << chosen_.size() - base_depth_ << " idx=" << idx
            << " lattice=" << nodes_.size() << " prefix:" << prefix;
}

} // namespace profiles::q02_n333
