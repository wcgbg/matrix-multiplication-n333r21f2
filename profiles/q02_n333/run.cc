#include "profiles/q02_n333/run.h"
#include "matrix/problem.h"
#include "profiles/q02_n333/search.h"
#include "profiles/q02_n333/self_test.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

namespace profiles::q02_n333 {

namespace {
// " v1 v2 ..." (a space before every value): the format of every printed list.
std::string SpaceJoined(const std::vector<u16> &v) {
  std::string s;
  for (u16 x : v)
    s += " " + std::to_string(x);
  return s;
}

// --check_list: one explicit list against every capacity. Returns the exit
// code.
int RunCheckList(const CapTable &table, const std::vector<Perm> &group,
                 const std::vector<u16> &list, const Options &options) {
  Search search(table, group, options);
  const bool ok = search.Check(list);
  std::printf("check_list: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}

// " |W|=k:n" for every k in [--min_w, --max_w].
std::string OuterCaseHistogram(const std::vector<OuterCase> &cases,
                               const Options &options) {
  std::array<int, 9> hist{};
  for (const auto &c : cases)
    ++hist[c.w.size()];
  std::string h;
  for (int k = options.min_w; k <= options.max_w; ++k) {
    h += " |W|=" + std::to_string(k) + ":" + std::to_string(hist[k]);
  }
  return h;
}

// --list_only.
void PrintCaseList(const std::vector<OuterCase> &cases) {
  for (size_t i = 0; i < cases.size(); ++i) {
    std::printf("case %zu |W|=%zu n2=%d n3=%d:%s\n", i, cases[i].w.size(),
                cases[i].n2, cases[i].n3, SpaceJoined(cases[i].w).c_str());
  }
}

// --cases, or every case; each index must exist.
std::vector<size_t> SelectCases(size_t num_cases, const Options &options) {
  std::vector<size_t> selected;
  if (options.cases.empty()) {
    for (size_t i = 0; i < num_cases; ++i)
      selected.push_back(i);
  } else {
    for (unsigned long i : options.cases)
      selected.push_back(i);
  }
  for (size_t i : selected)
    CHECK_LT(i, num_cases);
  return selected;
}

// Jobs: (case, forced rank-one prefix, continuation index). A case whose W
// already violates a capacity gets no job and is INFEASIBLE outright.
struct Job {
  size_t sel = 0; // index into `selected`
  std::vector<u16> prefix;
  int start = 0;
};
struct Result {
  bool w_violates = false;
  int jobs = 0;
  int done = 0;
  int over_budget = 0;
  int solution_limit = 0;
  uint64_t nodes = 0;
  std::set<std::vector<u16>> profiles;
  double secs = 0;
};

// The jobs of the selected cases (the prefixes at --split_depth), recording the
// per-case job counts and W violations in *results.
std::vector<Job> BuildJobs(const CapTable &table,
                           const std::vector<Perm> &group,
                           const std::vector<OuterCase> &cases,
                           const std::vector<size_t> &selected,
                           std::vector<Result> *results,
                           const Options &options) {
  std::vector<Job> jobs;
  Search search(table, group, options);
  for (size_t j = 0; j < selected.size(); ++j) {
    const auto &c = cases[selected[j]];
    const std::string s = SpaceJoined(c.w);
    if (!search.Prepare(c.w)) {
      (*results)[j].w_violates = true;
      LOG(INFO) << "case " << selected[j] << " |W|=" << c.w.size() << " W:" << s
                << " -> INFEASIBLE (W violates " << search.last_violation()
                << ")";
      continue;
    }
    const auto prefixes = search.Prefixes(options.split_depth);
    for (const auto &p : prefixes)
      jobs.push_back({j, p.first, p.second});
    (*results)[j].jobs = prefixes.size();
    LOG(INFO) << "case " << selected[j] << " |W|=" << c.w.size()
              << " n2=" << c.n2 << " n3=" << c.n3 << " W:" << s << " -> "
              << prefixes.size() << " job(s)";
  }
  return jobs;
}

// The worker pool: every worker owns one Search, re-prepares it when a job's
// case changes, and records each job's result under the mutex.
struct JobPool {
  const Options &options;
  const CapTable &table;
  const std::vector<Perm> &group;
  const std::vector<OuterCase> &cases;
  const std::vector<size_t> &selected;
  const std::vector<Job> &jobs;
  std::vector<Result> *results;
  std::mutex mu;
  std::atomic<size_t> next_job{0};
  const std::chrono::steady_clock::time_point t_start =
      std::chrono::steady_clock::now();

  void Record(size_t k, const Job &job, const SearchResult<u16> &result,
              double secs) {
    std::lock_guard<std::mutex> lock(mu);
    Result &r = (*results)[job.sel];
    ++r.done;
    r.nodes += result.nodes;
    r.secs += secs;
    if (result.stop_reason == StopReason::kNodeBudget)
      ++r.over_budget;
    if (result.stop_reason == StopReason::kSolutionLimit)
      ++r.solution_limit;
    for (const auto &sol : result.solutions)
      r.profiles.insert(sol);
    if (r.done == r.jobs) {
      LOG(INFO) << "case " << selected[job.sel]
                << " complete: nodes=" << r.nodes
                << " profiles=" << r.profiles.size()
                << " over_budget_jobs=" << r.over_budget << " cpu=" << r.secs
                << "s";
    } else if (jobs.size() > 1 && (k % 64 == 0)) {
      LOG(INFO) << "  progress: " << k << "/" << jobs.size()
                << " jobs dispatched, "
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t_start)
                       .count()
                << "s elapsed";
    }
  }

  void Worker() {
    Search search(table, group, options);
    size_t prepared = SIZE_MAX;
    while (true) {
      const size_t k = next_job.fetch_add(1);
      if (k >= jobs.size())
        break;
      const Job &job = jobs[k];
      if (prepared != job.sel) {
        CHECK(search.Prepare(cases[selected[job.sel]].w));
        prepared = job.sel;
      }
      const auto t0 = std::chrono::steady_clock::now();
      const auto result = search.RunFrom(job.prefix, job.start, options.budget);
      const double secs =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      Record(k, job, result, secs);
    }
  }

  // nthreads workers including the calling thread.
  void Run(int nthreads) {
    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t)
      pool.emplace_back([this] { Worker(); });
    Worker();
    for (auto &th : pool)
      th.join();
  }
};

// The per-case verdict lines with their profiles on stdout, then the summary
// (stdout flushed first, then the LOG line, then the printed line).
int PrintSummary(const std::vector<OuterCase> &cases,
                 const std::vector<size_t> &selected,
                 const std::vector<Result> &results, int rank,
                 bool restricted) {
  EnumerationResult result;
  result.rank = rank;
  result.restricted = restricted;
  int feasible = 0;
  int infeasible = 0;
  int over = 0;
  for (size_t j = 0; j < selected.size(); ++j) {
    const Result &r = results[j];
    result.jobs += r.jobs;
    result.finished_jobs += r.done;
    result.nodes += r.nodes;
    result.budget_stops += r.over_budget;
    result.solution_limit_stops += r.solution_limit;
    result.cases.push_back({selected[j], static_cast<uint64_t>(r.jobs),
                            static_cast<uint64_t>(r.done), r.nodes,
                            r.profiles.size(),
                            static_cast<uint64_t>(r.over_budget),
                            static_cast<uint64_t>(r.solution_limit)});
    for (const auto &sol : r.profiles)
      result.profiles.emplace_back(sol.begin(), sol.end());
    std::string verdict = "INFEASIBLE";
    if (!r.profiles.empty())
      verdict = "FEASIBLE";
    else if (r.over_budget > 0)
      verdict = "OVER BUDGET";
    std::printf(
        "case %zu |W|=%zu: %s jobs=%d nodes=%llu profiles=%zu cpu=%.1fs%s\n",
        selected[j], cases[selected[j]].w.size(), verdict.c_str(), r.jobs,
        static_cast<unsigned long long>(r.nodes), r.profiles.size(), r.secs,
        r.w_violates ? " (W violates)" : "");
    for (const auto &sol : r.profiles) {
      std::printf("  profile:%s\n", SpaceJoined(sol).c_str());
    }
    if (verdict == "FEASIBLE")
      ++feasible;
    else if (verdict == "INFEASIBLE")
      ++infeasible;
    else
      ++over;
  }
  std::fflush(stdout);
  LOG(INFO) << "summary: feasible=" << feasible << " infeasible=" << infeasible
            << " over_budget=" << over << " of " << selected.size();
  std::printf("summary: feasible=%d infeasible=%d over_budget=%d of %zu\n",
              feasible, infeasible, over, selected.size());
  result.Print();
  return result.exhausted() ? 0 : 3;
}

} // namespace

int Run(const pb::Certificate &cert, const Options &options) {
  ValidateBinaryOptions(options);
  CapTable table(options.r);
  LoadBounds(ExpandSubspaceBounds<matrix::Problem<2, 3, 3, 3>>(cert), &table);
  const std::vector<Perm> group = BuildGroup();
  CHECK_EQ(group.size(), size_t{56'448});
  SelfTest(table, group);

  if (!options.check_list.empty()) {
    std::vector<u16> list;
    for (unsigned long v : options.check_list) {
      list.push_back(static_cast<u16>(v));
    }
    return RunCheckList(table, group, list, options);
  }

  const std::vector<Perm> stab = StabilizerOfB0(group);
  const std::vector<OuterCase> cases = EnumerateOuter(stab, options);
  LOG(INFO) << "outer cases (up to the stabilizer of B0): " << cases.size()
            << OuterCaseHistogram(cases, options);
  if (options.list_only) {
    PrintCaseList(cases);
    return 0;
  }

  const std::vector<size_t> selected = SelectCases(cases.size(), options);
  std::vector<Result> results(selected.size());
  const std::vector<Job> jobs =
      BuildJobs(table, group, cases, selected, &results, options);
  LOG(INFO) << jobs.size() << " jobs over " << selected.size() << " cases";

  JobPool pool{options, table, group, cases, selected, jobs, &results};
  pool.Run(std::max(1, std::min<int>(options.threads, jobs.size())));
  return PrintSummary(cases, selected, results, options.r,
                      options.experimental);
}

} // namespace profiles::q02_n333
