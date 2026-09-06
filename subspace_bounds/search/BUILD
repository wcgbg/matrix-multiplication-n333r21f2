# Search/prover code: orbit enumeration, the meet-in-the-middle OrbitMap, the
# top-down DP rank-lower-bound computer, and the backtracking-DFS / degenerate
# proof *generators*. This is the complex half of the project and is NOT part of
# the verifier trust base, see README.md, "What to trust". It depends inward on the
# trusted primitives in //core and the problem definitions in //matrix.
#
# The orbit_enumerator_main and rank_lower_bound_main binaries (the first and
# third pipeline stages) also live here, next to the libraries they drive.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "orbit_map",
    hdrs = ["orbit_map.h"],
    deps = [
        "//core:bit_vec",
        "//core:constraints",
        "//core:symmetry",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "orbit_map_test",
    srcs = ["orbit_map_test.cc"],
    deps = [
        ":orbit_map",
        "//core:bit_vec",
        "//core:constraints",
        "//core:gf_vec",
        "//matrix:problem",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "orbit_enumerator_slow",
    hdrs = ["orbit_enumerator_slow.h"],
    deps = [
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:symmetry",
        "//core:tensor",
        "//core:tensor_utils",
        "@boost.unordered",
        "@ng-log//:ng-log",
    ],
)

cc_test(
    name = "orbit_enumerator_slow_test",
    srcs = ["orbit_enumerator_slow_test.cc"],
    data = [":testdata_orbit_enumerator"],
    deps = [
        ":orbit_enumerator_slow",
        "//core:certificate_cc_proto",
        "//core:proto_io",
        "//matrix:problem",
        "@googletest//:gtest_main",
        "@protobuf",
    ],
)

cc_library(
    name = "orbit_enumerator",
    hdrs = ["orbit_enumerator.h"],
    deps = [
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:gf_vec",
        "//core:symmetry",
        "//core:tensor",
        "//core:tensor_utils",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "orbit_enumerator_test",
    srcs = ["orbit_enumerator_test.cc"],
    data = [":testdata_orbit_enumerator"],
    deps = [
        ":orbit_enumerator",
        "//core:certificate_cc_proto",
        "//core:proto_io",
        "//matrix:problem",
        "@googletest//:gtest_main",
        "@protobuf",
    ],
)

cc_library(
    name = "rank_lower_bound_degenerate",
    hdrs = ["rank_lower_bound_degenerate.h"],
    deps = [
        ":orbit_map",
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:gf_vec",
    ],
)

cc_library(
    name = "rank_lower_bound_backtracking",
    hdrs = ["rank_lower_bound_backtracking.h"],
    deps = [
        ":orbit_map",
        "//core:backtracking_proof",
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_library(
    name = "rank_lower_bound_computer",
    hdrs = ["rank_lower_bound_computer.h"],
    deps = [
        ":orbit_map",
        ":rank_lower_bound_backtracking",
        ":rank_lower_bound_degenerate",
        "//core:backtracking_proof",
        "//core:bit_vec",
        "//core:certificate_cc_proto",
        "//core:constraints",
        "//core:proto_io",
        "//core:rank_lower_bound_flatten",
        "//core:rank_lower_bound_forced_product",
        "//core:rank_lower_bound_rank_one_span",
        "//core:tensor",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "rank_lower_bound_computer_test",
    srcs = ["rank_lower_bound_computer_test.cc"],
    data = [
        ":testdata_orbit_enumerator",
        ":testdata_rank_lower_bound_all",
        ":testdata_rank_lower_bound_basic",
        ":testdata_rank_lower_bound_basic_degenerate",
    ],
    deps = [
        ":rank_lower_bound_computer",
        "//core:certificate_cc_proto",
        "//core:proto_io",
        "//matrix:problem",
        "@googletest//:gtest_main",
        "@protobuf",
    ],
)

cc_binary(
    name = "orbit_enumerator_main",
    srcs = ["orbit_enumerator_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":orbit_enumerator",
        ":orbit_enumerator_slow",
        "//core:certificate_cc_proto",
        "//core:proto_io",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
    ],
)

cc_binary(
    name = "rank_lower_bound_main",
    srcs = ["rank_lower_bound_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":orbit_map",
        ":rank_lower_bound_computer",
        "//core:certificate_cc_proto",
        "//core:proto_io",
        "//matrix:chosen_problem",
        "@gflags",
        "@ng-log//:ng-log",
    ],
)

# Regenerates the golden certificates below; run by hand when the enumerator or
# rank-lower-bound algorithms change. Writes into subspace_bounds/search/testdata/ by default.
cc_binary(
    name = "testdata_generator_main",
    srcs = ["testdata_generator_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":orbit_enumerator_slow",
        ":rank_lower_bound_computer",
        "//core:certificate_cc_proto",
        "//core:proto_io",
        "//matrix:problem",
        "@gflags",
        "@ng-log//:ng-log",
    ],
)

# Golden test inputs for the orbit-enumerator and rank-lower-bound tests above.
# Exposed as filegroups so they can be shared via `data`; the runtime path
# strings the tests read are "subspace_bounds/search/testdata/...".
filegroup(
    name = "testdata_orbit_enumerator",
    srcs = glob(["testdata/orbit_enumerator/*.pb.txt"]),
)

filegroup(
    name = "testdata_rank_lower_bound_basic",
    srcs = glob(["testdata/rank_lower_bound_basic/*.pb.txt"]),
)

filegroup(
    name = "testdata_rank_lower_bound_basic_degenerate",
    srcs = glob(["testdata/rank_lower_bound_basic_degenerate/*.pb.txt"]),
)

filegroup(
    name = "testdata_rank_lower_bound_all",
    srcs = glob(["testdata/rank_lower_bound_all/*.pb.txt"]),
)

# Plaintext + gzipped golden pair consumed by //core:proto_io_test's
# round-trip check.
filegroup(
    name = "testdata_proto_io_golden",
    srcs = [
        "testdata/orbit_enumerator/matrix_q02_n222.pb.txt",
        "testdata/orbit_enumerator/matrix_q02_n222.pb.txt.gz",
    ],
)
