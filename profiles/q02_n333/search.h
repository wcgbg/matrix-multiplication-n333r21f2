#pragma once

#include "profiles/result.h"

#include "profiles/q02_n333/algebra.h"
#include "profiles/q02_n333/options.h"
#include "profiles/q02_n333/symmetry.h"
#include <chrono>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace profiles::q02_n333 {

constexpr int kLatticeMaxDim = 6;
constexpr int kBlockCap = 7; // m(B_x) <= 7, --prop3 excess lemma, r = 20 only
constexpr int kLineCap =
    12; // m(S_{x,c}) <= 12, --prop3 excess lemma, r = 20 only

struct Node {
  u128 key = 0;
  uint8_t dim = 0;
  uint8_t cap = 0;
  uint8_t m = 0;
  u16 rows[kLatticeMaxDim] = {};
};

struct Undo {
  size_t first_new = 0;
  std::vector<int> incremented;
};

class Search {
public:
  Search(const CapTable &table, const std::vector<Perm> &group,
         const Options &options);

  // Runs one outer case, reporting solutions and the precise stop reason.
  SearchResult<u16> Run(const std::vector<u16> &w, uint64_t budget);

  // Runs the part of an outer case whose first rank-one choices are exactly
  // `prefix` (increasing), continuing from candidate index `start`.
  // Prepare(w) must have been called.
  SearchResult<u16> RunFrom(const std::vector<u16> &prefix, int start,
                            uint64_t budget);

  // The lex-minimal, capacity-feasible rank-one prefixes of length `depth`
  // (with the candidate index to continue from); together they partition the
  // search of the case. Prepare(w) must have been called.
  std::vector<std::pair<std::vector<u16>, int>> Prefixes(int depth);

  int need() const;

  // Loads W: resets the state, computes Stab(W) and adds W's elements.
  // Returns false (with last_violation_ set) if W alone violates a capacity.
  bool Prepare(const std::vector<u16> &w);

  const std::string &last_violation() const;

  // Checks one explicit list against every capacity (no search).
  bool Check(const std::vector<u16> &list);

  uint64_t nodes_visited() const;
  const std::set<std::vector<u16>> &solutions() const;

private:
  void Reset();

  // Adds u to the chosen set, updating the lattice and the contextual
  // counters. Returns false on any violated capacity; the caller must Undo
  // in either case (the partial update is recorded in *undo).
  bool Add(u16 u, Undo *undo);

  // Creates the lattice node for the RREF `rows` unless present; returns
  // false if its capacity is violated.
  bool Create(const u16 *rows, int dim, Undo *undo);

  void Violation(const u16 *rows, int dim, int m, int cap);

  void UndoAdd(const Undo &undo);

  // Leaf: extend the lattice to dims 7 and 8 (temporarily) and check them.
  bool LeafOk();

  void Dfs(int idx, int need);

  void PrefixDfs(int idx, int depth, std::vector<u16> *prefix,
                 std::vector<std::pair<std::vector<u16>, int>> *out);

  // Every element of a saturated span (m == cap) is dead: adding it would
  // push that span over its capacity. Prune when the live candidates beyond
  // idx cannot supply `need` more elements.
  bool EnoughLive(int idx, int need);

  // Is the rank-one part of the chosen set plus u (which exceeds all of it)
  // lex-minimal in its Stab(W)-orbit? Elements are compared as sorted lists.
  bool LexMinWith(u16 u);

  void MaybeLog(int idx);

  const CapTable &table_;
  const Options options_;
  const std::vector<Perm> &group_;
  std::vector<u16> ones_;
  int rowvec_[8][512] = {};
  int colvec_[8][512] = {};
  std::vector<Node> nodes_;
  std::unordered_map<u128, int, U128Hash> index_;
  std::vector<u16> chosen_;
  int cnt_row_[8] = {};
  int cnt_col_[8] = {};
  int cnt_line_row_[8][8] = {};
  int cnt_line_col_[8][8] = {};
  size_t base_depth_ = 0;
  int need_ = 0;
  uint64_t budget_ = 0;
  uint64_t nodes_visited_ = 0;
  StopReason stop_reason_ = StopReason::kExhausted;
  std::set<std::vector<u16>> solutions_;
  std::chrono::steady_clock::time_point last_log_;
  std::string last_violation_;
  std::vector<const Perm *> stab_w_;
};

} // namespace profiles::q02_n333
