#include "profiles/q02_n333/search.h"
#include <algorithm>

namespace profiles::q02_n333 {

void Search::Reset() {
  nodes_.clear();
  index_.clear();
  chosen_.clear();
  for (int x = 0; x < 8; ++x) {
    cnt_row_[x] = cnt_col_[x] = 0;
    for (int c = 0; c < 8; ++c)
      cnt_line_row_[x][c] = cnt_line_col_[x][c] = 0;
  }
}

bool Search::Add(u16 u, Undo *undo) {
  undo->first_new = nodes_.size();
  undo->incremented.clear();
  chosen_.push_back(u);
  bool ok = true;
  for (int x = 1; x < 8; ++x) {
    const int rv = rowvec_[x][u];
    const int cv = colvec_[x][u];
    if (rv == 0 && ++cnt_row_[x] > kBlockCap && options_.prop3) {
      ok = false;
      last_violation_ = "contextual m(B_x) > 7, x=" + std::to_string(x);
    }
    if (cv == 0 && ++cnt_col_[x] > kBlockCap && options_.prop3) {
      ok = false;
      last_violation_ = "contextual m(B^col_y) > 7, y=" + std::to_string(x);
    }
    for (int c = 1; c < 8; ++c) {
      if ((rv == 0 || rv == c) && ++cnt_line_row_[x][c] > kLineCap &&
          options_.prop3) {
        ok = false;
        last_violation_ = "contextual m(S_{x,c}) > 12";
      }
      if ((cv == 0 || cv == c) && ++cnt_line_col_[x][c] > kLineCap &&
          options_.prop3) {
        ok = false;
        last_violation_ = "contextual m(S^col_{y,z}) > 12";
      }
    }
  }
  if (!ok)
    return false;
  // Singleton span.
  {
    u16 r[1] = {u};
    if (!Create(r, 1, undo))
      return false;
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
      if (!Create(rows.data(), d, undo))
        return false;
    }
  }
  return true;
}

bool Search::Create(const u16 *rows, int dim, Undo *undo) {
  const u128 key = KeyOfRref(rows, dim);
  if (index_.count(key))
    return true;
  Node nd;
  nd.key = key;
  nd.dim = dim;
  nd.cap = table_.Get(key);
  nd.m = 0;
  for (int i = 0; i < dim; ++i)
    nd.rows[i] = rows[i];
  for (u16 c : chosen_) {
    if (InSpan(nd.rows, dim, c))
      ++nd.m;
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

void Search::Violation(const u16 *rows, int dim, int m, int cap) {
  last_violation_ = "dim " + std::to_string(dim) + " span{";
  for (int i = 0; i < dim; ++i)
    last_violation_ += (i ? "," : "") + std::to_string(rows[i]);
  last_violation_ += "} m=" + std::to_string(m) +
                     " cap=" + std::to_string(cap) +
                     " (L=" + std::to_string(options_.r - cap) + ")";
}

void Search::UndoAdd(const Undo &undo) {
  const u16 u = chosen_.back();
  chosen_.pop_back();
  for (int x = 1; x < 8; ++x) {
    const int rv = rowvec_[x][u];
    const int cv = colvec_[x][u];
    if (rv == 0)
      --cnt_row_[x];
    if (cv == 0)
      --cnt_col_[x];
    for (int c = 1; c < 8; ++c) {
      if (rv == 0 || rv == c)
        --cnt_line_row_[x][c];
      if (cv == 0 || cv == c)
        --cnt_line_col_[x][c];
    }
  }
  for (int i : undo.incremented)
    --nodes_[i].m;
  for (size_t j = undo.first_new; j < nodes_.size(); ++j)
    index_.erase(nodes_[j].key);
  nodes_.resize(undo.first_new);
}

bool Search::LeafOk() {
  std::unordered_map<u128, char, U128Hash> seen;
  std::vector<std::vector<u16>> frontier;
  for (const Node &nd : nodes_) {
    if (nd.dim == kLatticeMaxDim)
      frontier.emplace_back(nd.rows, nd.rows + nd.dim);
  }
  for (int dim = kLatticeMaxDim + 1; dim <= 8; ++dim) {
    std::vector<std::vector<u16>> next;
    for (const auto &base : frontier) {
      for (u16 c : chosen_) {
        if (InSpan(base.data(), base.size(), c))
          continue;
        std::vector<u16> rows = base;
        rows.push_back(c);
        Rref(rows);
        const u128 key = KeyOfRref(rows.data(), rows.size());
        if (!seen.emplace(key, 1).second)
          continue;
        int m = 0;
        for (u16 v : chosen_) {
          if (InSpan(rows.data(), rows.size(), v))
            ++m;
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

bool Search::EnoughLive(int idx, int need) {
  uint64_t dead[8] = {};
  for (const Node &nd : nodes_) {
    if (nd.m < nd.cap)
      continue;
    const uint32_t lim = 1u << nd.dim;
    for (uint32_t mask = 1; mask < lim; ++mask) {
      u16 v = 0;
      for (int i = 0; i < nd.dim; ++i) {
        if (mask >> i & 1)
          v ^= nd.rows[i];
      }
      dead[v >> 6] |= uint64_t{1} << (v & 63);
    }
  }
  int live = 0;
  for (size_t i = idx; i < ones_.size(); ++i) {
    const u16 v = ones_[i];
    if (!(dead[v >> 6] >> (v & 63) & 1) && ++live >= need)
      return true;
  }
  return live >= need;
}

} // namespace profiles::q02_n333
