#pragma once

#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

namespace profiles {

enum class StopReason { kExhausted, kNodeBudget, kSolutionLimit };

template <class Point> struct SearchResult {
  StopReason stop_reason = StopReason::kExhausted;
  uint64_t nodes = 0;
  std::set<std::vector<Point>> solutions;
  bool exhausted() const { return stop_reason == StopReason::kExhausted; }
};

struct CaseResult {
  size_t id = 0;
  uint64_t jobs = 0, finished_jobs = 0, nodes = 0, profile_count = 0;
  uint64_t budget_stops = 0, solution_limit_stops = 0;
};

// Stable machine-readable boundary between the C++ enumerators and run.py.
// Human progress/verdict lines are deliberately not part of this interface.
struct EnumerationResult {
  int rank = 0;
  bool restricted = false;
  uint64_t jobs = 0, finished_jobs = 0, nodes = 0;
  uint64_t budget_stops = 0, solution_limit_stops = 0;
  std::vector<std::vector<uint32_t>> profiles;
  std::vector<CaseResult> cases;
  bool exhausted() const {
    return finished_jobs == jobs && budget_stops == 0 &&
           solution_limit_stops == 0;
  }
  void Print(std::ostream &out = std::cout) const {
    out << "result_json: "
           "{\"schema_version\":1,\"self_test_passed\":true,\"rank\":"
        << rank << ",\"scope\":\"" << (restricted ? "restricted" : "full")
        << "\""
        << ",\"exhausted\":" << (exhausted() ? "true" : "false")
        << ",\"jobs\":" << jobs << ",\"finished_jobs\":" << finished_jobs
        << ",\"nodes\":" << nodes << ",\"budget_stops\":" << budget_stops
        << ",\"solution_limit_stops\":" << solution_limit_stops
        << ",\"profiles\":[";
    for (size_t i = 0; i < profiles.size(); ++i) {
      if (i)
        out << ',';
      out << '[';
      for (size_t j = 0; j < profiles[i].size(); ++j) {
        if (j)
          out << ',';
        out << profiles[i][j];
      }
      out << ']';
    }
    out << "],\"cases\":[";
    for (size_t i = 0; i < cases.size(); ++i) {
      if (i)
        out << ',';
      const auto &c = cases[i];
      out << "{\"id\":" << c.id << ",\"jobs\":" << c.jobs
          << ",\"finished_jobs\":" << c.finished_jobs
          << ",\"nodes\":" << c.nodes
          << ",\"profile_count\":" << c.profile_count
          << ",\"budget_stops\":" << c.budget_stops
          << ",\"solution_limit_stops\":" << c.solution_limit_stops << '}';
    }
    out << "]}\n";
  }
};

} // namespace profiles
