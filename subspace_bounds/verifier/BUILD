# The rank-lower-bound verifier — the trusted checker. Together with //core
# (trusted primitives) and //matrix (tensor + symmetry definitions, including
# //matrix:chosen_problem), this is the complete trust base for the project's rank
# lower bounds. See README.md, "What to trust". It does NOT depend on //subspace_bounds/search or
# //subspace_bounds/upper_bound.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "backtracking_verifier",
    hdrs = ["backtracking_verifier.h"],
    deps = [
        "//core:backtracking_proof",
        "//core:bit_vec",
        "//core:constraints",
        "@boost.unordered",
        "@ng-log//:ng-log",
    ],
)

cc_library(
    name = "verifier",
    hdrs = ["verifier.h"],
    deps = [
        ":backtracking_verifier",
        "//core:backtracking_proof",
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:rank_lower_bound_flatten",
        "//core:rank_lower_bound_forced_product",
        "//core:rank_lower_bound_rank_one_span",
        "//core:tensor",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_binary(
    name = "verifier_main",
    srcs = ["verifier_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":verifier",
        "//core:backtracking_proof",
        "//core:certificate_cc_proto",
        "//core:proto_io",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
    ],
)

# Generates certificates at runtime (via //subspace_bounds/search) and verifies them, so the
# test — but not the trusted verifier library above — depends on //subspace_bounds/search.
cc_test(
    name = "verifier_test",
    srcs = ["verifier_test.cc"],
    deps = [
        ":verifier",
        "//core:backtracking_proof",
        "//core:certificate_cc_proto",
        "//matrix:problem",
        "//subspace_bounds/search:orbit_enumerator_slow",
        "//subspace_bounds/search:rank_lower_bound_computer",
        "@googletest//:gtest_main",
        "@onetbb//:tbb",
    ],
)
