#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>

#include "core/gf.h"
#include "core/tensor.h"

// Convert a tensor to its sparse string representation, e.g.,
// a0*b1*c3 + a1*b0*c2 + ... (F2), or 2*a0*b1*c3 + ... (F3,F4,F5).
// If the tensor is all zeros, return "0".
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
std::string TensorToSparseString(const Tensor<P, NA, NB, NC> &tensor) {
  std::string result;
  for (std::size_t i = 0; i < NA; ++i) {
    for (std::size_t j = 0; j < NB; ++j) {
      for (std::size_t k = 0; k < NC; ++k) {
        const uint8_t coeff = tensor[i][j][k].value;
        if (coeff == 0) {
          continue;
        }
        if (!result.empty()) {
          result += " + ";
        }
        if (coeff != 1) {
          result += std::to_string(static_cast<int>(coeff));
          result += '*';
        }
        result += 'a';
        result += std::to_string(i);
        result += "*b";
        result += std::to_string(j);
        result += "*c";
        result += std::to_string(k);
      }
    }
  }
  if (result.empty()) {
    return "0";
  }
  return result;
}

// `text` is like a0*b1*c1 + a1*b0*c1 (F2), or 2*a0*b1*c1 + ... (F3,F4,F5).
// The string "0" (or an empty/whitespace-only string) produces the zero tensor.
template <int P, std::size_t NA, std::size_t NB, std::size_t NC>
Tensor<P, NA, NB, NC> SparseStringToTensor(const std::string &text) {
  Tensor<P, NA, NB, NC> tensor{};
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t plus = text.find('+', pos);
    const size_t end = (plus == std::string::npos) ? text.size() : plus;
    // Parse the single term in [pos, end), ignoring whitespace.
    int coeff = 1;
    bool has_coeff = false;
    int idx[3] = {-1, -1, -1}; // indices for a, b, c.
    for (size_t i = pos; i < end;) {
      const char c = text[i];
      if (std::isspace(static_cast<unsigned char>(c)) || c == '*') {
        ++i;
      } else if (std::isdigit(static_cast<unsigned char>(c))) {
        int value = 0;
        while (i < end && std::isdigit(static_cast<unsigned char>(text[i]))) {
          value = value * 10 + (text[i] - '0');
          ++i;
        }
        coeff = value;
        has_coeff = true;
      } else { // 'a', 'b', or 'c' followed by an index.
        const int which = c - 'a';
        ++i;
        int value = 0;
        while (i < end && std::isdigit(static_cast<unsigned char>(text[i]))) {
          value = value * 10 + (text[i] - '0');
          ++i;
        }
        idx[which] = value;
      }
    }
    pos = (plus == std::string::npos) ? text.size() : plus + 1;
    // A "0" term, or an empty term, contributes nothing.
    if (idx[0] < 0 || idx[1] < 0 || idx[2] < 0) {
      continue;
    }
    if (!has_coeff) {
      coeff = 1;
    }
    tensor[idx[0]][idx[1]][idx[2]] = GF<P>{static_cast<uint8_t>(coeff)};
  }
  return tensor;
}
