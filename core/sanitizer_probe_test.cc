#include "gtest/gtest.h"
#include <climits>

#ifndef __has_feature
#define __has_feature(x) 0
#endif
#if !defined(__SANITIZE_ADDRESS__) && !__has_feature(address_sanitizer)
#error                                                                         \
    "Run this manual test with --config=debug (ASan instrumentation required)"
#endif

namespace {
// These deliberate errors execute only in death-test child processes.
TEST(SanitizerProbeDeathTest, DetectsHeapBufferOverflow) {
  EXPECT_DEATH(
      {
        int *values = new int[1];
        volatile int index = 1;
        values[index] = 7;
        delete[] values;
      },
      "AddressSanitizer: heap-buffer-overflow");
}

TEST(SanitizerProbeDeathTest, StopsOnUndefinedSignedOverflow) {
  EXPECT_DEATH(
      {
        volatile int maximum = INT_MAX;
        volatile int one = 1;
        volatile int result = maximum + one;
        (void)result;
      },
      "runtime error: signed integer overflow");
}
} // namespace
