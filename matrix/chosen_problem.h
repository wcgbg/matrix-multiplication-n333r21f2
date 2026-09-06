#pragma once

// Compile-time problem selection, shared by the prover, verifier, and table
// binaries in subspace_bounds/. The profile enumerators select their problem
// independently.
//
// The problem is chosen at *build time* via preprocessor defines, NOT by
// editing this file: pass the characteristic and the three matrix dimensions
// as separate defines. They are kept comma-free on purpose: Bazel splits copt
// values on commas, so the commas below separate distinct -D options (not
// template arguments). Example:
//
//   # <3,3,3> over F_2
//   bazel build //subspace_bounds/verifier:verifier_main \
//     --per_file_copt='.*_main\.cc@-DCP_P=2,-DCP_N0=3,-DCP_N1=3,-DCP_N2=3'
//
// chosen_problem.h is included only by the *_main.cc binaries, so scoping the
// define to those translation units avoids invalidating the rest of the build
// cache (core/, matrix/, external deps) when the problem changes. run.py
// passes these flags automatically. A plain `bazel build //...` with no
// override uses the committed default below, so `//...` builds and tests keep
// working.
//
// `matrix::Problem<P, N0, N1, N2>` is the ⟨N0, N1, N2⟩ matrix-multiplication
// tensor over the prime field F_P. rank_upper_bound_main is hard-coded to
// P == 2 (the F_2 flip-graph search); for odd P run.py skips that stage.

#include "matrix/problem.h"

#if defined(CP_P) || defined(CP_N0) || defined(CP_N1) || defined(CP_N2)
#if !(defined(CP_P) && defined(CP_N0) && defined(CP_N1) && defined(CP_N2))
#error "pass all four of -DCP_P -DCP_N0 -DCP_N1 -DCP_N2"
#endif
#define CHOSEN_PROBLEM matrix::Problem<CP_P, CP_N0, CP_N1, CP_N2>
#else
// Committed default, used when no CP_* define is passed (e.g. plain
// `bazel build //...`): <3,3,3> over F_2.
#define CHOSEN_PROBLEM matrix::Problem<2, 3, 3, 3>
#endif

// Declared at global scope (no namespace): it is a single project-wide
// build-time selection, used by the *_main.cc binaries as `::ChosenProblem`.
using ChosenProblem = CHOSEN_PROBLEM;
