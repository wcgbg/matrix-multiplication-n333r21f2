#pragma once

// Gauss-Jordan elimination over the prime field 𝔽_P.
//
// Two entry points live here:
//   - GaussJordanEliminationF2<N> for the P == 2 bit-packed hot path
//     (operates on the BitVec<N> inside each GFVec<2, N>);
//   - GaussJordanEliminationFq<P, N> for odd P, operating on
//     the packed-digit GFVec<P, N> rows.
//
// Both produce full reduced row-echelon form (RREF):
//   - column order reversed (highest-coordinate pivot first);
//   - row order reversed (zero rows park at the front), so erasing the
//     leading-zero rows gives the canonical key the OrbitMap stores;
//   - each pivot row is normalized to leading 1 — for F_q this is a scalar
//     multiplication by the pivot's inverse; for F₂ the pivot is already 1;
//   - entries above and below each pivot are eliminated (full RREF, not just
//     row echelon).
//
// Returns the number of nonzero rows (the rank). core/constraints.h provides
// a dispatching wrapper (GaussJordanRREF) that picks the right entry point.

#include <vector>

#include "core/bit_vec.h"
#include "core/gf.h"
#include "core/gf_vec.h"

// F₂ Gauss-Jordan elimination on the bit-packed F₂ row
// representation. Accesses each row's underlying BitVec via `.data` so the
// algorithm operates directly on the GFVec<2, N> wrapper, preserving the
// XOR/popcount hot path.
//
// Outputs are full RREF in the column-reversed convention:
//   - row order reversed (zero rows park at the front),
//   - each pivot row is leading-1 (already true in F₂),
//   - above-and-below elimination (full Jordan).
// Returns the number of nonzero rows (the rank).
template <int N>
int GaussJordanEliminationF2(std::vector<GFVec<2, N>> *matrix) {
  using BV = BitVec<N>;
  const int size = static_cast<int>(matrix->size());
  if (size == 0 || N <= 0) {
    return 0;
  }
  GFVec<2, N> *const data = matrix->data();

  int rank = 0;
  int current_row = size - 1;
  int pivot_col = N - 1;

  while (current_row >= 0 && pivot_col >= 0) {
    const BV col_mask = static_cast<BV>(BV{1} << pivot_col);
    int pivot_row = -1;
    for (int r = current_row; r >= 0; --r) {
      if (data[r].data & col_mask) {
        pivot_row = r;
        break;
      }
    }
    if (pivot_row == -1) {
      --pivot_col;
      continue;
    }

    if (pivot_row != current_row) {
      std::swap(data[current_row], data[pivot_row]);
    }

    for (int r = 0; r < size; ++r) {
      if (r != current_row && (data[r].data & col_mask)) {
        data[r].data = static_cast<BV>(data[r].data ^ data[current_row].data);
      }
    }

    ++rank;
    --current_row;
    --pivot_col;
  }

  return rank;
}

template <int P, int N>
int GaussJordanEliminationFq(std::vector<GFVec<P, N>> *matrix) {
  static_assert(P != 2, "GaussJordanEliminationFq is for odd primes; use "
                        "GaussJordanEliminationF2 for the F₂ BitVec hot path");
  using Vec = GFVec<P, N>;
  using Field = GF<P>;

  const int size = static_cast<int>(matrix->size());
  if (size == 0 || N <= 0) {
    return 0;
  }
  Vec *const data = matrix->data();

  int rank = 0;
  int current_row = size - 1;
  int pivot_col = N - 1;

  const Field zero = Field::Zero();
  const Field one = Field::One();

  while (current_row >= 0 && pivot_col >= 0) {
    // Find a pivot row: any row in [0, current_row] with a nonzero entry in
    // the pivot column.
    int pivot_row = -1;
    for (int r = current_row; r >= 0; --r) {
      if (data[r][pivot_col] != zero) {
        pivot_row = r;
        break;
      }
    }
    if (pivot_row == -1) {
      --pivot_col;
      continue;
    }

    if (pivot_row != current_row) {
      std::swap(data[current_row], data[pivot_row]);
    }

    // Normalize the pivot row to leading 1 by scaling by the inverse of the
    // pivot. Cheap: a single field-scalar multiplication.
    const Field pivot = data[current_row][pivot_col];
    if (pivot != one) {
      data[current_row] = pivot.Inverse() * data[current_row];
    }

    // Eliminate above and below the pivot: for each other row r with a
    // nonzero entry c in this column, do row[r] -= c * row[current_row].
    for (int r = 0; r < size; ++r) {
      if (r == current_row) {
        continue;
      }
      const Field c = data[r][pivot_col];
      if (c == zero) {
        continue;
      }
      data[r] = data[r] - (c * data[current_row]);
    }

    ++rank;
    --current_row;
    --pivot_col;
  }

  return rank;
}

template <int P, int N>
bool IsLinearIndependentFq(std::vector<GFVec<P, N>> matrix) {
  return GaussJordanEliminationFq<P, N>(&matrix) ==
         static_cast<int>(matrix.size());
}
