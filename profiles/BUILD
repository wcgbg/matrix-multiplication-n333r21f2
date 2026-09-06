# Exhaustive capacity-constrained profile enumeration using subspace tables.
# These tools are outside the certificate verifier trust boundary.
load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "profile_enum_q02_n333_main",
    srcs = ["profile_enum_q02_n333_main.cc"],
    deps = [
        "@gflags",
        "@ng-log//:ng-log",
    ],
)

cc_binary(
    name = "profile_enum_fp_main",
    srcs = ["profile_enum_fp_main.cc"],
    deps = [
        "@gflags",
        "@ng-log//:ng-log",
    ],
)
