#pragma once

// The prime field 𝔽_P (P prime).
//
// GF<P> is BOTH:
//   - a value type: one byte of storage, the element in [0, P).
//   - the field's static-ops home: kQ and static Add/Sub/Mul/Neg/Inverse live
//     on the type itself.
//
// Direct mod-P arithmetic (no tables); P == 2 collapses to XOR/AND. Inverse via
// Fermat.
//
// Layout: sizeof(GF<P>) == 1, trivially copyable, layout-compatible with
// uint8_t. This lets std::array<GF, N> share its byte representation with
// std::array<uint8_t, N> — load-bearing for the memcpy paths in
// core/constraints.h.

#include <cstdint>
#include <string>
#include <type_traits>

#include "ng-log/logging.h"

namespace intpow_detail {
// Non-constexpr on purpose: reaching this call during constant evaluation makes
// the enclosing expression non-constant, so a compile-time IntPow overflow is a
// required diagnostic (build error) with a clear "non-constexpr function"
// message. The runtime overflow path uses LOG(FATAL) instead. Keep it
// non-constexpr.
inline void Overflow() {}
} // namespace intpow_detail

// base^exp over uint64. Overflow is treated as an error:
//   - constant-evaluated (static_assert, constexpr init, template args): a hard
//     compile error, via the non-constexpr intpow_detail::Overflow().
//   - at runtime: LOG(FATAL). No caller may rely on a saturated return value;
//     a caller that legitimately probes a possibly-huge search-space size must
//     bound the exponent itself before calling (see
//     RankLowerBoundForcedProductA, which caps its iteration count without ever
//     overflowing).
constexpr uint64_t IntPow(int base, int exp) {
  uint64_t r = 1;
  for (int i = 0; i < exp; ++i) {
    if (r > (UINT64_MAX / static_cast<uint64_t>(base))) {
      if (std::is_constant_evaluated()) {
        intpow_detail::Overflow();
      } else {
        LOG(FATAL) << "IntPow overflow: " << base << "^" << exp
                   << " does not fit in uint64";
      }
    }
    r *= static_cast<uint64_t>(base);
  }
  return r;
}

namespace gf_internal {

// Compile-time primality check.
constexpr bool IsPrime(int p) {
  if (p < 2)
    return false;
  if (p == 2)
    return true;
  if (p % 2 == 0)
    return false;
  for (int d = 3; d * d <= p; d += 2) {
    if (p % d == 0)
      return false;
  }
  return true;
}

// Scalar add / sub / mul / neg mod P.
template <int P> constexpr uint8_t AddModP(uint8_t a, uint8_t b) {
  if constexpr (P == 2) {
    return static_cast<uint8_t>(a ^ b);
  } else {
    const int s = static_cast<int>(a) + static_cast<int>(b);
    return static_cast<uint8_t>(s >= P ? s - P : s);
  }
}

template <int P> constexpr uint8_t SubModP(uint8_t a, uint8_t b) {
  if constexpr (P == 2) {
    return static_cast<uint8_t>(a ^ b);
  } else {
    const int s = static_cast<int>(a) - static_cast<int>(b);
    return static_cast<uint8_t>(s < 0 ? s + P : s);
  }
}

template <int P> constexpr uint8_t MulModP(uint8_t a, uint8_t b) {
  if constexpr (P == 2) {
    return static_cast<uint8_t>(a & b);
  } else {
    return static_cast<uint8_t>((static_cast<int>(a) * static_cast<int>(b)) %
                                P);
  }
}

template <int P> constexpr uint8_t NegModP(uint8_t a) {
  if constexpr (P == 2) {
    return a;
  } else {
    return static_cast<uint8_t>(a == 0 ? 0 : P - a);
  }
}

} // namespace gf_internal

template <int P> struct GF {
  static_assert(P >= 2 && P < 256, "P must satisfy 2 <= P < 256");
  static_assert(gf_internal::IsPrime(P), "P must be prime");

  uint8_t value = 0;
  static constexpr int kQ = P;

  constexpr GF() = default;
  constexpr GF(uint8_t v) : value(v) {}
  constexpr explicit operator uint8_t() const { return value; }

  static constexpr GF Zero() { return GF{static_cast<uint8_t>(0)}; }
  static constexpr GF One() { return GF{static_cast<uint8_t>(1)}; }

  constexpr bool operator==(const GF &) const = default;

  static constexpr GF Add(GF a, GF b) {
    return GF{gf_internal::AddModP<P>(a.value, b.value)};
  }
  static constexpr GF Sub(GF a, GF b) {
    return GF{gf_internal::SubModP<P>(a.value, b.value)};
  }
  static constexpr GF Mul(GF a, GF b) {
    return GF{gf_internal::MulModP<P>(a.value, b.value)};
  }
  static constexpr GF Neg(GF a) { return GF{gf_internal::NegModP<P>(a.value)}; }

  // Multiplicative inverse via Fermat: a^(P−2) = a⁻¹. Inverse(0) returns 0
  // (ill-defined but does not produce surprising values; callers must not
  // invert 0).
  static constexpr GF Inverse(GF a) {
    if (a.value == 0) {
      return Zero();
    }
    if constexpr (P == 2) {
      return One();
    } else {
      int result = 1;
      int base = a.value;
      int e = P - 2;
      while (e > 0) {
        if (e & 1) {
          result = (result * base) % P;
        }
        base = (base * base) % P;
        e >>= 1;
      }
      return GF{static_cast<uint8_t>(result)};
    }
  }

  // a^exp via binary exponentiation. Pow(0, 0) = 1 by convention.
  static constexpr GF Pow(GF a, int exp) {
    GF result = One();
    GF base = a;
    while (exp > 0) {
      if (exp & 1) {
        result = Mul(result, base);
      }
      base = Mul(base, base);
      exp >>= 1;
    }
    return result;
  }

  friend constexpr GF operator+(GF a, GF b) { return Add(a, b); }
  friend constexpr GF operator-(GF a, GF b) { return Sub(a, b); }
  friend constexpr GF operator*(GF a, GF b) { return Mul(a, b); }
  friend constexpr GF operator-(GF a) { return Neg(a); }
  constexpr GF Inverse() const { return Inverse(*this); }

  constexpr GF &operator+=(GF b) {
    value = Add(*this, b).value;
    return *this;
  }
  constexpr GF &operator-=(GF b) {
    value = Sub(*this, b).value;
    return *this;
  }
  constexpr GF &operator*=(GF b) {
    value = Mul(*this, b).value;
    return *this;
  }

  // Decimal representation of the element — multi-character for P > 10
  // (e.g., GF<13>{12} → "12") so values stay unambiguous.
  std::string ToString() const {
    return std::to_string(static_cast<int>(value));
  }
};

// Layout guarantees the rest of the codebase relies on.
static_assert(sizeof(GF<2>) == 1);
static_assert(sizeof(GF<3>) == 1);
static_assert(std::is_trivially_copyable_v<GF<2>>);
static_assert(std::is_trivially_copyable_v<GF<3>>);
