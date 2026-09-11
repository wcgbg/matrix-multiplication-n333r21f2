#include "dynamic_matrix.h"

#include <sstream>

#include "ng-log/logging.h"

template <int P>
DynamicMatrix<P>::DynamicMatrix(int n0, int n1)
    : n0_(n0), n1_(n1), data_(n0 * n1, Field::Zero()) {}

template <int P> void DynamicMatrix<P>::ResizeRows(int n0) {
  n0_ = n0;
  data_.resize(n0 * n1_);
  // Newly-added entries default-initialize to the field zero. std::vector's
  // `resize(new_size)` value-initializes trailing GF<P> elements (value = 0
  // = Field::Zero()), so no extra fill is needed.
}

template <int P> std::string DynamicMatrix<P>::ToString() const {
  std::ostringstream oss;
  oss << '[';
  for (int i0 = 0; i0 < n0_; ++i0) {
    if (i0 > 0) {
      oss << ';';
    }
    for (int i1 = 0; i1 < n1_; ++i1) {
      if (i1 > 0) {
        oss << ',';
      }
      oss << (*this)(i0, i1).ToString();
    }
  }
  oss << ']';
  return oss.str();
}

template <int P>
DynamicMatrix<P> DynamicMatrix<P>::Plus(const DynamicMatrix<P> &other) const {
  CHECK_EQ(n0_, other.n0_) << "Plus: row count mismatch";
  CHECK_EQ(n1_, other.n1_) << "Plus: column count mismatch";
  DynamicMatrix<P> result(n0_, n1_);
  for (int i0 = 0; i0 < n0_; ++i0) {
    for (int i1 = 0; i1 < n1_; ++i1) {
      result(i0, i1) = (*this)(i0, i1) + other(i0, i1);
    }
  }
  return result;
}

template <int P> bool DynamicMatrix<P>::IsZero() const {
  const Field zero = Field::Zero();
  for (int i0 = 0; i0 < n0_; ++i0) {
    for (int i1 = 0; i1 < n1_; ++i1) {
      if ((*this)(i0, i1) != zero) {
        return false;
      }
    }
  }
  return true;
}

template <int P> int DynamicMatrix<P>::Rank() const {
  if (n0_ == 0 || n1_ == 0) {
    return 0;
  }
  // Copy to row-major 2D structure for Gauss-Jordan elimination.
  std::vector<std::vector<Field>> rows(n0_,
                                       std::vector<Field>(n1_, Field::Zero()));
  for (int i0 = 0; i0 < n0_; ++i0) {
    for (int i1 = 0; i1 < n1_; ++i1) {
      rows[i0][i1] = (*this)(i0, i1);
    }
  }

  const Field zero = Field::Zero();
  const Field one = Field::One();

  int rank = 0;
  int pivot_col = 0;

  for (int row = 0; row < n0_ && pivot_col < n1_; ++row) {
    int pivot_row = -1;
    for (int r = row; r < n0_; ++r) {
      if (rows[r][pivot_col] != zero) {
        pivot_row = r;
        break;
      }
    }

    if (pivot_row == -1) {
      ++pivot_col;
      --row;
      continue;
    }

    if (pivot_row != row) {
      std::swap(rows[row], rows[pivot_row]);
    }

    // Normalize the pivot row to leading 1. For P = 2 the pivot is
    // already 1 and this scaling is a no-op; for odd P we
    // scale by the inverse.
    if constexpr (P != 2) {
      const Field pivot = rows[row][pivot_col];
      if (pivot != one) {
        const Field inv = pivot.Inverse();
        for (int c = 0; c < n1_; ++c) {
          rows[row][c] = inv * rows[row][c];
        }
      }
    }

    // Eliminate above and below the pivot.
    for (int r = 0; r < n0_; ++r) {
      if (r == row) {
        continue;
      }
      const Field c = rows[r][pivot_col];
      if (c == zero) {
        continue;
      }
      for (int col = 0; col < n1_; ++col) {
        rows[r][col] = rows[r][col] - c * rows[row][col];
      }
    }

    ++rank;
    ++pivot_col;
  }

  return rank;
}

template class DynamicMatrix<2>;
template class DynamicMatrix<3>;
template class DynamicMatrix<5>;
template class DynamicMatrix<7>;
template class DynamicMatrix<11>;
template class DynamicMatrix<13>;
