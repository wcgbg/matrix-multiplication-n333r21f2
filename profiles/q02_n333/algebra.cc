#include "profiles/q02_n333/algebra.h"
#include <algorithm>

namespace profiles::q02_n333 {

int Rank(u16 c) {
  int rows[3] = {c & 7, (c >> 3) & 7, (c >> 6) & 7};
  int r = 0;
  for (int bit = 0; bit < 3; ++bit) {
    int piv = -1;
    for (int k = r; k < 3; ++k) {
      if (rows[k] >> bit & 1) {
        piv = k;
        break;
      }
    }
    if (piv < 0)
      continue;
    std::swap(rows[r], rows[piv]);
    for (int k = 0; k < 3; ++k) {
      if (k != r && (rows[k] >> bit & 1))
        rows[k] ^= rows[r];
    }
    ++r;
  }
  return r;
}

u16 Transpose(u16 u) {
  u16 t = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (u >> (3 * i + j) & 1)
        t |= u16{1} << (3 * j + i);
    }
  }
  return t;
}

u16 Mul(u16 a, u16 b) { // (a b)_{ij} = sum_k a_{ik} b_{kj}
  u16 c = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int s = 0;
      for (int k = 0; k < 3; ++k)
        s ^= (a >> (3 * i + k) & 1) & (b >> (3 * k + j) & 1);
      if (s)
        c |= u16{1} << (3 * i + j);
    }
  }
  return c;
}

int RowVec(int x, u16 u) { // x^T U as a 3-bit column-index vector
  int v = 0;
  for (int j = 0; j < 3; ++j) {
    int s = 0;
    for (int i = 0; i < 3; ++i)
      s ^= (x >> i & 1) & (u >> (3 * i + j) & 1);
    if (s)
      v |= 1 << j;
  }
  return v;
}

int ColVec(int y, u16 u) { // U y as a 3-bit row-index vector
  int v = 0;
  for (int i = 0; i < 3; ++i) {
    int s = 0;
    for (int j = 0; j < 3; ++j)
      s ^= (y >> j & 1) & (u >> (3 * i + j) & 1);
    if (s)
      v |= 1 << i;
  }
  return v;
}

void LoadBounds(const std::vector<SubspaceBound<9>> &records, CapTable *t) {
  t->lb.reserve(8'400'000);
  size_t n = 0;
  for (const auto &record : records) {
    std::vector<u16> rows(record.rows.begin(),
                          record.rows.begin() + record.dim);
    const int dim = Rref(rows);
    CHECK_EQ(dim, record.dim);
    if (dim == 0)
      continue;
    t->lb[KeyOfRref(rows.data(), dim)] = record.lb;
    ++n;
  }
  CHECK_EQ(n, size_t{8'283'457}) << "unexpected table size";
  LOG(INFO) << "capacity table: " << n << " subspaces";
}

} // namespace profiles::q02_n333
