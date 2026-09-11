#include "profiles/options.h"

#include <set>

#include "ng-log/logging.h"

namespace profiles {

void ValidateOptions(const Options &options, int max_rank) {
  if (!options.check_list.empty())
    max_rank = 255;
  CHECK_GE(options.r, 1) << "--r must be positive";
  CHECK_LE(options.r, max_rank) << "--r exceeds the supported rank";
  CHECK_GT(options.max_solutions, 0) << "solution limit must be positive";
  CHECK_GT(options.log_seconds, 0) << "log interval must be positive";
  CHECK_GT(options.threads, 0) << "thread count must be positive";
  CHECK_GE(options.split_depth, 0) << "--split_depth must be nonnegative";
  if (options.check_list.empty())
    CHECK_LE(options.split_depth, options.r) << "--split_depth exceeds r";
  CHECK_LE(options.check_list.size(), static_cast<size_t>(options.r))
      << "--check_list has more than r factors";
  CHECK_EQ(
      std::set<uint32_t>(options.ones_subset.begin(), options.ones_subset.end())
          .size(),
      options.ones_subset.size())
      << "--ones_subset must not contain duplicate codes";
}

void ValidateCodes(const std::vector<uint32_t> &codes, uint64_t limit) {
  for (uint32_t code : codes) {
    CHECK_GT(code, 0) << "factor code must be nonzero";
    CHECK_LT(code, limit) << "factor code is out of range";
  }
}

} // namespace profiles
