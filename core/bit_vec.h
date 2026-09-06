#pragma once

// BitVec<N>: the smallest unsigned integer type holding at least N bits.
// Constraints on A live in (𝔽₂^N)*, which we identify with N-bit integers.
//
// We use uint8_t / uint16_t / uint32_t / uint64_t depending on N, because:
//   - smaller types compress constraint vectors and the orbit-map keys, and
//   - the DFS path bitmask in backtracking must fit in uint32_t (so N ≤ 31 is
//     a hard ceiling of the substitution method).
//
// For matrix multiplication N = n0 * n1, the number of entries of an A-side
// matrix.

#include <cstdint>
#include <type_traits>

template <std::size_t N>
using BitVec = std::conditional_t<
    (N <= 8), uint8_t,
    std::conditional_t<(N <= 16), uint16_t,
                       std::conditional_t<(N <= 32), uint32_t, uint64_t>>>;

static_assert(std::is_same_v<BitVec<4>, uint8_t>);
static_assert(std::is_same_v<BitVec<8>, uint8_t>);
static_assert(std::is_same_v<BitVec<9>, uint16_t>);
static_assert(std::is_same_v<BitVec<16>, uint16_t>);
static_assert(std::is_same_v<BitVec<17>, uint32_t>);
static_assert(std::is_same_v<BitVec<32>, uint32_t>);
static_assert(std::is_same_v<BitVec<33>, uint64_t>);

// Low N bits set.
template <std::size_t N>
inline constexpr BitVec<N> kBitVecAllOnes = static_cast<BitVec<N>>(
    static_cast<BitVec<N>>(~BitVec<N>{0}) >> (sizeof(BitVec<N>) * 8 - N));

// All-ones spans the full type when N is exactly the type width.
static_assert(kBitVecAllOnes<8> == uint8_t{0xFF});
static_assert(kBitVecAllOnes<16> == uint16_t{0xFFFF});
static_assert(kBitVecAllOnes<32> == uint32_t{0xFFFFFFFF});
static_assert(kBitVecAllOnes<64> == uint64_t{0xFFFFFFFFFFFFFFFF});
// And only the low N bits when N is below the width.
static_assert(kBitVecAllOnes<1> == uint8_t{0x01});
static_assert(kBitVecAllOnes<4> == uint8_t{0x0F});
static_assert(kBitVecAllOnes<9> == uint16_t{0x01FF});
static_assert(kBitVecAllOnes<31> == uint32_t{0x7FFFFFFF});
static_assert(kBitVecAllOnes<33> == uint64_t{0x1FFFFFFFF});
