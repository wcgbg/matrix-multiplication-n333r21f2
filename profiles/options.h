#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "ng-log/logging.h"

namespace profiles {

// Values are supplied by the CLI; these defaults match the generic executable.
struct Options {
  int r = 14;
  uint64_t budget = 1'000'000'000;
  int max_solutions = 100;
  int log_seconds = 30;
  bool symmetry = true;
  int threads = 1;
  int split_depth = 0;
  std::vector<uint32_t> check_list;
  std::vector<uint32_t> ones_subset;
};

// Validate before narrowing codes, indexing fixed scratch arrays, or searching.
void ValidateOptions(const Options &options, int max_rank = 64);
void ValidateCodes(const std::vector<uint32_t> &codes, uint64_t limit);

template <class T = uint32_t> std::vector<T> ParseList(const std::string &csv) {
  std::vector<T> out;
  if (csv.empty())
    return out;
  size_t pos = 0;
  do {
    size_t next = csv.find(',', pos);
    if (next == std::string::npos)
      next = csv.size();
    uint64_t value = 0;
    const auto parsed =
        std::from_chars(csv.data() + pos, csv.data() + next, value);
    CHECK(parsed.ec == std::errc{}) << "invalid unsigned integer list: " << csv;
    CHECK_EQ(parsed.ptr, csv.data() + next)
        << "invalid unsigned integer list: " << csv;
    CHECK_LE(value, std::numeric_limits<T>::max())
        << "invalid unsigned integer list: " << csv;
    out.push_back(static_cast<T>(value));
    if (next == csv.size())
      break;
    pos = next + 1;
  } while (true);
  return out;
}

} // namespace profiles
