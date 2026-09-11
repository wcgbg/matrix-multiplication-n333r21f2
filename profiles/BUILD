# Exhaustive capacity-constrained profile enumeration.
# These tools are outside the certificate verifier trust boundary.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "options",
    srcs = ["options.cc"],
    hdrs = [
        "options.h",
        "result.h",
    ],
    deps = ["@ng-log//:ng-log"],
)

cc_library(
    name = "q02_n333",
    srcs = [
        "q02_n333/algebra.cc",
        "q02_n333/options.cc",
        "q02_n333/run.cc",
        "q02_n333/search.cc",
        "q02_n333/search_lattice.cc",
        "q02_n333/self_test.cc",
        "q02_n333/symmetry.cc",
    ],
    hdrs = [
        "q02_n333/algebra.h",
        "q02_n333/options.h",
        "q02_n333/run.h",
        "q02_n333/search.h",
        "q02_n333/self_test.h",
        "q02_n333/symmetry.h",
    ],
    deps = [
        ":options",
        "//core:certificate_cc_proto",
        "//matrix:problem",
        "//subspace_bounds",
        "@ng-log//:ng-log",
    ],
)

cc_library(
    name = "fp",
    hdrs = [
        "fp/geometry.h",
        "fp/problem_data.h",
        "fp/run.h",
        "fp/search.h",
        "fp/self_test.h",
    ],
    deps = [
        ":options",
        "//subspace_bounds",
        "@ng-log//:ng-log",
    ],
)

cc_binary(
    name = "profile_enum_q02_n333_main",
    srcs = ["profile_enum_q02_n333_main.cc"],
    deps = [
        ":options",
        ":q02_n333",
        "//core:proto_io",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
    ],
)

cc_binary(
    name = "profile_enum_fp_main",
    srcs = ["profile_enum_fp_main.cc"],
    deps = [
        ":fp",
        ":options",
        "//core:proto_io",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
    ],
)

cc_test(
    name = "fp_search_test",
    srcs = ["fp/search_test.cc"],
    deps = [
        ":fp",
        ":options",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "q02_n333_search_test",
    srcs = ["q02_n333/search_test.cc"],
    deps = [
        ":q02_n333",
        "@googletest//:gtest_main",
    ],
)
