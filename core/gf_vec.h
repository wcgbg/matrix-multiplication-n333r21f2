#pragma once

// GFVec<P, N>: a row vector in 𝔽_P^N (P prime).
//
// Stored as std::array<GF<P>, N>: one GF value per element. Since
// sizeof(GF<P>) == 1 and GF is trivially copyable, sizeof(GFVec<P, N>)
// == N — byte-blob hashing (ConstraintsHash) and on-disk serialisation read
// the buffer uniformly, identical to the legacy uint8_t-backed layout.
//
// For P == 2 the partial specialization below replaces the
// std::array<GF, N> backing with a single bit-packed BitVec<N> (sizeof ==
// sizeof(BitVec<N>)). Generic code keeps the same surface (operator[],
// IsZero, LeadingNonzeroIdx, +/-/*); F₂-specific hot paths reach into
// `.data` to do raw integer XOR/popcount.
//
// Fieldent-level arithmetic uses GF<P>'s operators.

#include <array>
#include <bit>
#include <compare>
#include <cstdint>

#include "core/bit_vec.h"
#include "core/gf.h"

template <int P, int N> struct GFVec {
  static_assert(N >= 0, "N must be non-negative");
  static_assert(P >= 2 && P < 256, "P must satisfy 2 <= P < 256");

  using Field = GF<P>;

  // One GF value per element. Byte layout is identical to the legacy
  // std::array<uint8_t, N> form because GF is layout-compatible with uint8_t.
  std::array<Field, static_cast<std::size_t>(N)> data{};

  constexpr Field operator[](int i) const { return data[i]; }
  constexpr Field &operator[](int i) { return data[i]; }

  constexpr void Set(int i, Field v) { data[i] = v; }

  constexpr bool operator==(const GFVec &) const = default;

  // Ordering matches the F_2 BitVec<N> convention and the integer
  // encoding i ↦ Σ data[k] · q^k: coordinate N-1 is the "most significant"
  // element (compared first). Iterating an integer counter from 0 to q^N − 1
  // visits GFVecs in this total order.
  constexpr auto operator<=>(const GFVec &other) const {
    for (int i = N - 1; i >= 0; --i) {
      if (auto c = data[i].value <=> other.data[i].value; c != 0) {
        return c;
      }
    }
    return std::strong_ordering::equal;
  }

  constexpr bool IsZero() const {
    for (std::size_t i = 0; i < data.size(); ++i) {
      if (data[i].value != 0) {
        return false;
      }
    }
    return true;
  }

  // Index of the highest-order nonzero element, or -1 if the vector is zero.
  constexpr int LeadingNonzeroIdx() const {
    for (int i = N - 1; i >= 0; --i) {
      if (data[i].value != 0) {
        return i;
      }
    }
    return -1;
  }

  // The value of the highest-order nonzero element, or zero if the vector is
  // zero.
  constexpr Field LeadingNonzero() const {
    const int idx = LeadingNonzeroIdx();
    if (idx < 0)
      return Field::Zero();
    return data[idx];
  }

  // Fieldent-wise field addition.
  friend constexpr GFVec operator+(GFVec a, const GFVec &b) {
    for (int i = 0; i < N; ++i) {
      a.data[i] += b.data[i];
    }
    return a;
  }

  friend constexpr GFVec operator-(GFVec a, const GFVec &b) {
    for (int i = 0; i < N; ++i) {
      a.data[i] -= b.data[i];
    }
    return a;
  }

  // Negation (per-element).
  friend constexpr GFVec operator-(GFVec a) {
    for (int i = 0; i < N; ++i) {
      a.data[i] = -a.data[i];
    }
    return a;
  }

  // Scalar multiplication.
  friend constexpr GFVec operator*(Field s, GFVec a) {
    for (int i = 0; i < N; ++i) {
      a.data[i] = s * a.data[i];
    }
    return a;
  }

  constexpr GFVec &operator+=(const GFVec &b) {
    *this = *this + b;
    return *this;
  }
  constexpr GFVec &operator-=(const GFVec &b) {
    *this = *this - b;
    return *this;
  }
  constexpr GFVec &operator*=(Field s) {
    *this = s * *this;
    return *this;
  }
};

// Partial specialization for P == 2: the F₂ hot path.
// Stores a single bit-packed BitVec<N>; the generic API (operator[], IsZero,
// LeadingNonzeroIdx, +/-/*) is preserved so templated code reads the same
// against both specializations. F₂-specific branches read/write `.data`
// directly for raw integer arithmetic (XOR/popcount/shift).
template <int N> struct GFVec<2, N> {
  static_assert(N >= 0, "N must be non-negative");

  using Field = GF<2>;

  // Public, aggregate-style: `GFVec<2, N>{bv}` brace-inits `data = bv`.
  // No converting constructor — keeping the type an aggregate makes the
  // "is this an F_q operator or an integer operator?" reading unambiguous at
  // every call site that touches `.data`.
  BitVec<N> data{};

  // Per-element read: bit `i` of `data`, returned by value (bits aren't
  // addressable, so `v[i]` cannot be an lvalue — no caller mutates via the
  // subscript in an F₂ branch).
  constexpr Field operator[](int i) const {
    return Field{static_cast<uint8_t>((data >> i) & 1)};
  }

  // Set element `i` to `v` (used by non-F₂ branches; provided for parity).
  constexpr void Set(int i, Field v) {
    const BitVec<N> mask = static_cast<BitVec<N>>(BitVec<N>{1} << i);
    if (v.value & 1) {
      data = static_cast<BitVec<N>>(data | mask);
    } else {
      data = static_cast<BitVec<N>>(data & static_cast<BitVec<N>>(~mask));
    }
  }

  constexpr bool operator==(const GFVec &) const = default;
  constexpr auto operator<=>(const GFVec &) const = default;

  constexpr bool IsZero() const { return data == 0; }

  // Index of the highest set bit, or -1 if zero. Matches the generic
  // template's "highest-order nonzero" semantics.
  constexpr int LeadingNonzeroIdx() const {
    return data == 0 ? -1 : (std::bit_width(data) - 1);
  }

  // The value of the highest-order nonzero element: 1 over F₂ (or 0 if zero).
  constexpr Field LeadingNonzero() const {
    return IsZero() ? Field::Zero() : Field::One();
  }

  // F₂ field ops collapse to XOR on `data`. Kept for API parity with the
  // primary template.
  friend constexpr GFVec operator+(GFVec a, const GFVec &b) {
    a.data = static_cast<BitVec<N>>(a.data ^ b.data);
    return a;
  }
  friend constexpr GFVec operator-(GFVec a, const GFVec &b) {
    a.data = static_cast<BitVec<N>>(a.data ^ b.data);
    return a;
  }
  friend constexpr GFVec operator-(GFVec a) { return a; }
  friend constexpr GFVec operator*(Field s, GFVec a) {
    if ((s.value & 1) == 0) {
      a.data = BitVec<N>{0};
    }
    return a;
  }
  constexpr GFVec &operator+=(const GFVec &b) {
    data = static_cast<BitVec<N>>(data ^ b.data);
    return *this;
  }
  constexpr GFVec &operator-=(const GFVec &b) {
    data = static_cast<BitVec<N>>(data ^ b.data);
    return *this;
  }
  constexpr GFVec &operator*=(Field s) {
    if ((s.value & 1) == 0) {
      data = BitVec<N>{0};
    }
    return *this;
  }
};

// Layout check: GFVec is just N bytes of data, no padding.
static_assert(sizeof(GFVec<3, 1>) == 1);
static_assert(sizeof(GFVec<3, 8>) == 8);
static_assert(sizeof(GFVec<5, 16>) == 16);

// On-disk format invariant: the F₂ specialization is exactly one BitVec<N>,
// so ConstraintsToBytes/FromBytes (which rely on sizeof(row)) match the
// legacy bit-packed-integer layout.
static_assert(sizeof(GFVec<2, 4>) == sizeof(BitVec<4>));
static_assert(sizeof(GFVec<2, 8>) == sizeof(BitVec<8>));
static_assert(sizeof(GFVec<2, 9>) == sizeof(BitVec<9>));
static_assert(sizeof(GFVec<2, 16>) == sizeof(BitVec<16>));
static_assert(sizeof(GFVec<2, 17>) == sizeof(BitVec<17>));
static_assert(sizeof(GFVec<2, 32>) == sizeof(BitVec<32>));
static_assert(sizeof(GFVec<2, 33>) == sizeof(BitVec<33>));

// Decode an integer in [0, q^N) into an GFVec, where q = P. The convention
// matches operator<=> above: data[i] = (n / q^i) mod q, so iterating the
// integer counter visits GFVecs in `<` order.
template <int P, int N>
constexpr GFVec<P, N> DecodeGFVec(uint64_t i) {
  if constexpr (P == 2) {
    return GFVec<2, N>{static_cast<BitVec<N>>(i)};
  } else {
    constexpr uint64_t kQ = GF<P>::kQ;
    GFVec<P, N> v{};
    for (std::size_t k = 0; k < v.data.size(); ++k) {
      v.data[k] = GF<P>{static_cast<uint8_t>(i % kQ)};
      i /= kQ;
    }
    return v;
  }
}

// Encode an GFVec to its integer in [0, q^N): Σ data[k] · q^k, where q = P.
template <int P, int N>
constexpr uint64_t EncodeGFVec(const GFVec<P, N> &v) {
  if constexpr (P == 2) {
    return static_cast<uint64_t>(v.data);
  } else {
    constexpr uint64_t kQ = GF<P>::kQ;
    uint64_t r = 0;
    for (int k = static_cast<int>(v.data.size()) - 1; k >= 0; --k) {
      r = r * kQ + static_cast<uint64_t>(v.data[k].value);
    }
    return r;
  }
}
