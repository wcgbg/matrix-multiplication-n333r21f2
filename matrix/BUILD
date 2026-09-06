# The <N0,N1,N2> matrix-multiplication tensor over a prime field and its
# A-side symmetry group (lookup tables for F_2, arithmetic for odd primes), plus
# the build-time problem selection. Part of the verifier trust base.
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
    deps = [
        "//core:tensor",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":tensor",
        "//core:tensor",
        "//core:tensor_utils",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "f2_matrix",
    hdrs = ["f2_matrix.h"],
)

cc_test(
    name = "f2_matrix_test",
    srcs = ["f2_matrix_test.cc"],
    deps = [
        ":f2_matrix",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "f2_matrix_tables",
    hdrs = ["f2_matrix_tables.h"],
    deps = [
        ":f2_matrix",
    ],
)

cc_test(
    name = "f2_matrix_tables_test",
    srcs = ["f2_matrix_tables_test.cc"],
    deps = [
        ":f2_matrix",
        ":f2_matrix_tables",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "f2_symmetry",
    hdrs = ["f2_symmetry.h"],
    deps = [
        ":f2_matrix",
        ":f2_matrix_tables",
        "//core:bit_vec",
        "//core:gf_vec",
    ],
)

cc_library(
    name = "fp_symmetry",
    hdrs = ["fp_symmetry.h"],
    deps = [
        "//core:gf",
        "//core:gf_vec",
    ],
)

cc_test(
    name = "fp_symmetry_test",
    srcs = ["fp_symmetry_test.cc"],
    deps = [
        ":fp_symmetry",
        ":tensor",
        "//core:constraints",
        "//core:gf",
        "//core:gf_vec",
        "//core:rank_lower_bound_flatten",
        "//core:tensor",
        "//core:tensor_utils",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "problem",
    hdrs = ["problem.h"],
    deps = [
        ":f2_symmetry",
        ":fp_symmetry",
        ":tensor",
        "//core:bit_vec",
        "//core:constraints",
        "//core:gf_vec",
        "//core:tensor",
    ],
)

cc_test(
    name = "f2_symmetry_test",
    srcs = ["f2_symmetry_test.cc"],
    deps = [
        ":f2_symmetry",
        ":tensor",
        "//core:bit_vec",
        "//core:constraints",
        "//core:gf",
        "//core:gf_vec",
        "//core:rank_lower_bound_flatten",
        "//core:tensor",
        "//core:tensor_utils",
        "@googletest//:gtest_main",
    ],
)

# Compile-time problem selection shared by the prover, verifier, and table
# binaries in subspace_bounds/:
# the matrix problem's parameters are chosen via -DCP_* build defines — see
# chosen_problem.h.
cc_library(
    name = "chosen_problem",
    hdrs = ["chosen_problem.h"],
    deps = [
        ":problem",
    ],
)
