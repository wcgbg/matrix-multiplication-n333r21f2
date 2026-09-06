load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "flip_scheme",
    srcs = ["flip_scheme.cc"],
    hdrs = ["flip_scheme.h"],
    deps = [
        "@ng-log//:ng-log",
    ],
)

cc_library(
    name = "rank_upper_bound",
    hdrs = ["rank_upper_bound.h"],
    deps = [
        ":flip_scheme",
        "//core:tensor",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "rank_upper_bound_test",
    size = "small",
    srcs = ["rank_upper_bound_test.cc"],
    deps = [
        ":rank_upper_bound",
        "//core:tensor",
        "@googletest//:gtest_main",
    ],
)

# The rank_upper_bound_main binary (the second pipeline stage) lives here, next
# to the flip-graph search it drives. F_2-only. Problem selection comes from the
# //matrix:chosen_problem header via -DCP_* build defines.
cc_binary(
    name = "rank_upper_bound_main",
    srcs = ["rank_upper_bound_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":rank_upper_bound",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:proto_io",
        "//core:tensor",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)
