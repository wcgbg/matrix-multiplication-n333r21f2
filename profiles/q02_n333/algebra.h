#pragma once

#include "ng-log/logging.h"
#include "subspace_bounds/subspace_bounds.h"
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace profiles::q02_n333 {

// A 3x3 F_2 matrix is nine bits: bit 3*i+j is entry (i, j).
using u16 = uint16_t;
using u128 = __uint128_t;

struct U128Hash {
  size_t operator()(u128 k) const {
    const uint64_t lo = static_cast<uint64_t>(k);
    const uint64_t hi = static_cast<uint64_t>(k >> 64);
    uint64_t h = lo * 0x9E3779B97F4A7C15ull ^ (hi + 0x7F4A7C159E3779B9ull);
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    return static_cast<size_t>(h ^ (h >> 32));
  }
};

int Rank(u16 c);
u16 Transpose(u16 u);
u16 Mul(u16 a, u16 b);
int RowVec(int x, u16 u);
int ColVec(int y, u16 u);
// Pivot on the highest set bit; rows are sorted in descending order.
// Apply this convention to expanded records as well, so lookup keys match.
inline int Rref(std::vector<u16> &rows) {
  int n = rows.size();
  int r = 0;
  for (int bit = 8; bit >= 0; --bit) {
    int piv = -1;
    for (int k = r; k < n; ++k) {
      if (rows[k] >> bit & 1) {
        piv = k;
        break;
      }
    }
    if (piv < 0)
      continue;
    std::swap(rows[r], rows[piv]);
    for (int k = 0; k < n; ++k) {
      if (k != r && (rows[k] >> bit & 1))
        rows[k] ^= rows[r];
    }
    ++r;
  }
  rows.resize(r);
  std::sort(rows.begin(), rows.end(), std::greater<u16>());
  return r;
}
inline u128 KeyOfRref(const u16 *rows, int dim) {
  u128 key = dim;
  for (int i = 0; i < dim; ++i)
    key = (key << 9) | rows[i];
  return key;
}
inline bool InSpan(const u16 *rows, int dim, u16 v) {
  for (int i = 0; i < dim; ++i) {
    const int piv = 31 - __builtin_clz(rows[i]);
    if (v >> piv & 1)
      v ^= rows[i];
  }
  return v == 0;
}

struct CapTable {
  explicit CapTable(int rank) : rank_(rank) {}
  const int rank_;
  std::unordered_map<u128, uint8_t, U128Hash> lb; // key -> L(S)
  int L(u128 key) const {
    auto it = lb.find(key);
    CHECK(it != lb.end()) << "subspace missing from the table";
    return it->second;
  }
  int Get(u128 key) const { return rank_ - L(key); } // the capacity
  int LOfGens(std::vector<u16> gens) const {
    const int d = Rref(gens);
    if (d == 0)
      return rank_;
    return L(KeyOfRref(gens.data(), d));
  }
};

void LoadBounds(const std::vector<SubspaceBound<9>> &records, CapTable *table);

} // namespace profiles::q02_n333
