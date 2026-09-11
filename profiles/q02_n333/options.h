#pragma once

#include "profiles/options.h"

#include <cstddef>

namespace profiles::q02_n333 {

struct Options : profiles::Options {
  Options() {
    r = 20;
    budget = 100'000'000;
    split_depth = 4;
    max_solutions = 50;
    log_seconds = 10;
  }
  int min_w = 0;
  int max_w = 8;
  bool prop3 = false;
  bool experimental = false;
  bool list_only = false;
  std::vector<size_t> cases;
};

void ValidateBinaryOptions(const Options &options);

} // namespace profiles::q02_n333
