# Expand certified orbit bounds into subspace tables and check monotonicity.
# These tools are outside the certificate verifier trust boundary.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "subspace_bounds",
    hdrs = ["subspace_bounds.h"],
    deps = [
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:gf",
        "//core:gf_vec",
        "//subspace_bounds/search:orbit_map",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_binary(
    name = "check_table_main",
    srcs = ["check_table_main.cc"],
    deps = [
        ":subspace_bounds",
        "//core:bit_vec",
        "//core:constraints",
        "//core:gf",
        "//core:gf_vec",
        "//core:proto_io",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "subspace_bounds_test",
    srcs = ["subspace_bounds_test.cc"],
    deps = [
        ":subspace_bounds",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//matrix:problem",
        "@googletest//:gtest_main",
    ],
)
