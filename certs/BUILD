# Cached pre-computed certificates (see CLAUDE.md). The small matrix certs are
# exposed as test data for //core:rank_lower_bound_rank_one_span_test's golden
# soundness sweep; the runtime paths the tests read are "certs/matrix/...".
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "matrix_small_certs",
    srcs = glob([
        "matrix/cert_matrix_q02_n2*.pb.txt",
        "matrix/cert_matrix_q03_n233.pb.txt",
    ]),
)
