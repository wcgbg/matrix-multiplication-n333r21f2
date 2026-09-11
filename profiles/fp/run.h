#pragma once

#include "profiles/fp/search.h"
#include "profiles/fp/self_test.h"
#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

namespace profiles::fp {

template <class Problem>
int Run(const pb::Certificate &cert, const profiles::Options &options) {
  constexpr int P = Problem::kP, N0 = Problem::kN0, N1 = Problem::kN1;
  ProblemData<P, N0, N1> data;
  data.BuildPoints();
  data.LoadBounds(ExpandSubspaceBounds<Problem>(cert));
  CHECK_GE(options.r, data.L0())
      << "r below the certified rank: trivially infeasible";
  data.BuildGroup();
  SelfTest(data);
  using E = Search<P, N0, N1>;
  auto engine = std::make_unique<E>(data, options);
  engine->SetupCandidates(options.ones_subset);

  if (!options.check_list.empty()) {
    const bool ok = engine->Check(options.check_list);
    std::printf("check_list: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
  }

  const auto prefixes = engine->Prefixes(options.split_depth);
  LOG(INFO) << prefixes.size() << " job(s)";
  std::mutex mu;
  std::atomic<size_t> next_job{0};
  uint64_t total_nodes = 0;
  int over_budget = 0;
  int solution_limit = 0;
  size_t finished_jobs = 0;
  std::set<std::vector<int>> all_solutions;
  const auto t_start = std::chrono::steady_clock::now();
  auto worker = [&](E *e) {
    while (true) {
      const size_t k = next_job.fetch_add(1);
      if (k >= prefixes.size())
        break;
      const auto result =
          e->RunFrom(prefixes[k].first, prefixes[k].second, options.budget);
      std::lock_guard<std::mutex> lock(mu);
      ++finished_jobs;
      total_nodes += result.nodes;
      if (result.stop_reason == StopReason::kNodeBudget)
        ++over_budget;
      if (result.stop_reason == StopReason::kSolutionLimit)
        ++solution_limit;
      for (const auto &s : result.solutions)
        all_solutions.insert(s);
      if (prefixes.size() > 1 && k % 100 == 0) {
        LOG(INFO) << "  progress: " << k << "/" << prefixes.size()
                  << " jobs, nodes=" << total_nodes << ", "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_start)
                         .count()
                  << "s";
      }
    }
  };
  {
    const int nthreads =
        std::max(1, std::min<int>(options.threads, prefixes.size()));
    std::vector<std::unique_ptr<E>> engines;
    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t) {
      // Copy mutable state; all workers borrow the same immutable prepared
      // data.
      engines.push_back(std::make_unique<E>(*engine));
      pool.emplace_back(worker, engines.back().get());
    }
    worker(engine.get());
    for (auto &th : pool)
      th.join();
  }
  std::string verdict = "INFEASIBLE";
  if (!all_solutions.empty())
    verdict = "FEASIBLE";
  else if (over_budget > 0)
    verdict = "OVER BUDGET";
  std::printf(
      "r=%d: %s jobs=%zu nodes=%llu profiles=%zu over_budget_jobs=%d "
      "time=%.1fs\n",
      options.r, verdict.c_str(), prefixes.size(),
      static_cast<unsigned long long>(total_nodes), all_solutions.size(),
      over_budget,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
          .count());
  for (const auto &sol : all_solutions) {
    std::string s;
    for (int p : sol)
      s += " " + std::to_string(data.point_code_[p]);
    std::printf("  profile:%s\n", s.c_str());
  }
  EnumerationResult result;
  result.rank = options.r;
  result.restricted = !options.ones_subset.empty();
  result.jobs = prefixes.size();
  result.finished_jobs = finished_jobs;
  result.nodes = total_nodes;
  result.budget_stops = over_budget;
  result.solution_limit_stops = solution_limit;
  for (const auto &sol : all_solutions) {
    std::vector<uint32_t> codes;
    for (int p : sol)
      codes.push_back(data.point_code_[p]);
    result.profiles.push_back(std::move(codes));
  }
  result.Print();
  return result.exhausted() ? 0 : 3;
}

} // namespace profiles::fp
