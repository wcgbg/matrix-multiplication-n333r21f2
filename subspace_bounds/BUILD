# Expand certified orbit bounds into subspace tables and check monotonicity.
# These tools are outside the certificate verifier trust boundary.
load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "subspace_bounds_main",
    srcs = ["subspace_bounds_main.cc"],
    deps = [
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:gf",
        "//core:gf_vec",
        "//core:proto_io",
        "//matrix:chosen_problem",
        "//subspace_bounds/search:orbit_map",
        "@gflags",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_binary(
    name = "check_table_main",
    srcs = ["check_table_main.cc"],
    deps = [
        "//core:bit_vec",
        "//core:constraints",
        "//core:gf",
        "//core:gf_vec",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)
