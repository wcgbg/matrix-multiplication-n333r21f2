#pragma once

// A 3-tensor with compile-time dimensions over 𝔽_q (q = P), with one
// GF<P> coefficient per cell.

#include <array>
#include <cstddef>

#include "core/gf.h"

template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
using Tensor = std::array<std::array<std::array<GF<P>, NC>, NB>, NA>;
