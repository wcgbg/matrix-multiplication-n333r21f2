#pragma once

// Constraints on the A factor.
//
// A constraint is a linear functional in the dual space (𝔽_q^NA)*, where
// q = P prime. For the F₂ specialization P == 2 we
// represent it as a BitVec (an NA-bit unsigned integer); the bit-packed
// XOR/popcount hot path remains. For odd P the row type is a
// packed GFVec<P, NA> (one byte per F_q element, each holding the F_q
// index in [0, q); see core/gf.h for the index ↔ digit-vector
// bijection).
//
// A set of constraints is a std::vector<row>, normalised to (column-reversed)
// RREF so the vector itself is a canonical form for the subspace it spans.

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "ng-log/logging.h"

#include "core/bit_vec.h"
#include "core/gf_vec.h"
#include "core/math_utils.h"
#include "core/tensor.h"

// A constraint is one row of an NA-coordinate constraint set over 𝔽_q
// (q = P). The row type is GFVec<P, NA> in both regimes:
//   - P == 2: the partial specialization holds a bit-packed
//     BitVec<NA> in `.data` (preserving the XOR/popcount hot path);
//   - odd P: the primary template holds a packed
//     std::array<GF<P>, NA> of F_q indices (sizeof == NA).
// Both layouts are trivially copyable, byte-blob-hashable, and serialised
// raw into the certificate proto's `constraints` field.

template <int P, int NA>
using Constraints = std::vector<GFVec<P, NA>>;

// Column-reversed RREF dispatcher: routes to the F_2 bit-packed version for
// P = 2 and to the F_q mod-q version otherwise. Both implementations
// follow the same convention (column reversed, zero rows park at the front,
// pivot normalized to 1) so the F_2 result is bitwise-identical to today.
template <int P, int N> int GaussJordanRREF(Constraints<P, N> *m) {
  if constexpr (P == 2) {
    return GaussJordanEliminationF2<N>(m);
  } else {
    return GaussJordanEliminationFq<P, N>(m);
  }
}

// Independence check, P-aware.
template <int P, int N>
bool IsLinearIndependentRREF(Constraints<P, N> m) {
  return GaussJordanRREF<P, N>(&m) == static_cast<int>(m.size());
}

// Impose each constraint (a linear functional on the A-space, constrained to
// equal zero) on the A-mode of `tensor` by substitution, returning the reduced
// tensor.
//
// `constraints` is assumed to be in the column-reversed RREF canonical form
// that the Constraints type guarantees: every nonzero row's pivot is its most
// significant set element, that pivot is unique across rows and cleared from
// every other row, and the remaining ("free") coordinates are never themselves
// pivots. The row's constraint then reads a_pivot = −Σ row[i]·a_i over the
// free coordinates, so we fold the pivot's A-slice into each free coordinate's
// slice and zero the pivot slice. Because free coordinates are never pivots,
// one pass over the rows suffices and their order does not matter.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
Tensor<P, NA, NB, NC>
ApplyConstraintsToTensor(const Constraints<P, NA> &constraints,
                         const Tensor<P, NA, NB, NC> &tensor) {
  Tensor<P, NA, NB, NC> result = tensor;
  if constexpr (P == 2) {
    // F₂ fast path: GF<2,1> addition is XOR, so `+=` compiles to the same
    // bitwise operation as the historical `^=` over uint8_t. The row's
    // underlying BitVec lives at `.data`; pull it into a local for the
    // bit-twiddling.
    using BV = BitVec<NA>;
    for (const GFVec<P, NA> &row_v : constraints) {
      const BV row = row_v.data;
      if (row == 0)
        continue;
      const int pivot = std::bit_width(row) - 1;
      const BV free = static_cast<BV>(row ^ (BV{1} << pivot));
      for (std::size_t i = 0; i < NA; ++i) {
        if (!((free >> i) & 1))
          continue;
        for (std::size_t j = 0; j < NB; ++j) {
          for (std::size_t k = 0; k < NC; ++k) {
            result[i][j][k] += result[pivot][j][k];
          }
        }
      }
      for (std::size_t j = 0; j < NB; ++j) {
        for (std::size_t k = 0; k < NC; ++k) {
          result[pivot][j][k] = GF<P>::Zero();
        }
      }
    }
  } else {
    // F_q path: full-RREF canonical row has pivot coefficient 1, so the
    // substitution equation reads a_pivot = -Σ_{i ≠ pivot} row[i]·a_i.
    // Folding a_pivot into the tensor adds -row[i]·result[pivot] to result[i]
    // for each free coordinate i, then zeros result[pivot].
    for (const auto &row : constraints) {
      if (row.IsZero())
        continue;
      const int pivot = row.LeadingNonzeroIdx();
      CHECK((row[pivot] == GF<P>::One()))
          << "ApplyConstraintsToTensor: row pivot is not normalized to 1; "
             "input is not in canonical RREF";
      for (std::size_t i = 0; i < NA; ++i) {
        if (static_cast<int>(i) == pivot)
          continue;
        const GF<P> c = row[i];
        if (c == GF<P>::Zero())
          continue;
        for (std::size_t j = 0; j < NB; ++j) {
          for (std::size_t k = 0; k < NC; ++k) {
            result[i][j][k] = result[i][j][k] - c * result[pivot][j][k];
          }
        }
      }
      for (std::size_t j = 0; j < NB; ++j) {
        for (std::size_t k = 0; k < NC; ++k) {
          result[pivot][j][k] = GF<P>::Zero();
        }
      }
    }
  }
  return result;
}

// Print a single constraint (linear functional) as a string of length N, most
// significant coordinate first. For P = 2 it's a 0/1 string; otherwise each
// coordinate is a single character (P must be ≤ 16).
template <int P, int N>
std::string ConstraintToString(GFVec<P, N> v) {
  std::array<char, 16> digits = {'0', '1', '2', '3', '4', '5', '6', '7',
                                 '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
  if constexpr (P == 2) {
    std::string s(N, '0');
    for (int i = 0; i < N; ++i) {
      if ((v.data >> i) & 1)
        s[N - 1 - i] = '1';
    }
    return s;
  } else {
    static_assert(P <= digits.size());
    std::string s(N, '0');
    for (int i = 0; i < N; ++i) {
      s[N - 1 - i] = digits.at(v.data[i].value);
    }
    return s;
  }
}

// Print a constraint set as its per-functional bit strings, comma-joined, e.g.
// "00001,00011". "EMPTY" for the zero subspace. Mirrors the matrix-mult
// proof_verifier/constraints.h ConstraintsToString verbose field.
template <int P, int NA>
std::string ConstraintsToString(const Constraints<P, NA> &constraints) {
  if (constraints.empty()) {
    return "EMPTY";
  }
  std::string result;
  for (const GFVec<P, NA> v : constraints) {
    if (!result.empty()) {
      result += ',';
    }
    result += ConstraintToString<P, NA>(v);
  }
  return result;
}

// Serialise a constraint set to a raw little-endian byte blob (the proto's
// `ConstrainedTensor.constraints` field). Ports proof_verifier/constraints.h's
// ConstraintsToCompactString / ConstraintsFromCompactString, with the matrix
// data type swapped for the NA-wide row. The byte layout matches the buffer
// ConstraintsHash reads, so this is the single home for the on-disk encoding;
// the search and verifier binaries decode it with ConstraintsFromBytes.
template <int P, int NA>
std::string ConstraintsToBytes(const Constraints<P, NA> &constraints) {
  using Row = GFVec<P, NA>;
  static_assert(std::endian::native == std::endian::little);
  if (constraints.empty()) {
    return {};
  }
  return std::string(reinterpret_cast<const char *>(constraints.data()),
                     constraints.size() * sizeof(Row));
}

template <int P, int NA>
Constraints<P, NA> ConstraintsFromBytes(const std::string &bytes) {
  using Row = GFVec<P, NA>;
  static_assert(std::endian::native == std::endian::little);
  CHECK_EQ(bytes.size() % sizeof(Row), 0u);
  if (bytes.empty()) {
    return {};
  }
  Constraints<P, NA> constraints(bytes.size() / sizeof(Row));
  std::memcpy(constraints.data(), bytes.data(), bytes.size());
  return constraints;
}

// A fast hash for constraint sequences (drops in for boost::unordered).
// Reads the underlying row buffer 8 bytes at a time. Ported verbatim from
// proof_verifier/constraints.h (only the template parameters changed from a
// matrix data type to (NA, P)).
template <int P, int NA> struct ConstraintsHash {
  size_t operator()(const Constraints<P, NA> &r) const noexcept {
    using Row = GFVec<P, NA>;
    const uint8_t *data = reinterpret_cast<const uint8_t *>(r.data());
    size_t bytes = r.size() * sizeof(Row);
    uint64_t h = bytes;
    while (bytes >= 8) {
      uint64_t chunk;
      std::memcpy(&chunk, data, 8);
      h = (chunk + (h << 6)) + (0x9e3779b9 + (h >> 2));
      data += 8;
      bytes -= 8;
    }
    if (bytes > 0) {
      uint64_t chunk = 0;
      std::memcpy(&chunk, data, bytes);
      h = (chunk + (h << 6)) + (0x9e3779b9 + (h >> 2));
    }
    return h;
  }
};
