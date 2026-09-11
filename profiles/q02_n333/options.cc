#include "profiles/q02_n333/options.h"

#include <set>

#include "ng-log/logging.h"

namespace profiles::q02_n333 {

void ValidateBinaryOptions(const Options &options) {
  profiles::ValidateOptions(options, 32);
  ValidateCodes(options.check_list, 512);
  ValidateCodes(options.ones_subset, 512);
  CHECK_GE(options.min_w, 0) << "--min_w must be nonnegative";
  CHECK_LE(options.max_w, 8) << "--max_w must not exceed 8";
  CHECK_LE(options.min_w, options.max_w) << "--min_w must not exceed --max_w";
  CHECK_EQ(std::set<size_t>(options.cases.begin(), options.cases.end()).size(),
           options.cases.size())
      << "--cases must not contain duplicate indices";
  if (options.prop3)
    CHECK_EQ(options.r, 20) << "--prop3 is valid only at r=20";
  if (options.check_list.empty() && !options.experimental) {
    CHECK(options.r == 20 && !options.prop3 && options.min_w == 0 &&
          options.max_w == 8 && options.cases.empty() &&
          options.ones_subset.empty())
        << "restricted or non-paper enumeration requires --experimental=true; "
           "it cannot establish a full profile classification";
  }
}

} // namespace profiles::q02_n333
