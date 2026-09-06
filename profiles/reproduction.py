"""Pure validation and supporting calculations for the paper workflows.

These helpers check recorded outcomes, not the mathematical completeness proofs.
The enumerators' current FEASIBLE verdict alone does not certify exhaustion.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import struct


class ValidationError(ValueError):
    """A reproduction did not establish its expected outcome."""


TERNARY_PROFILES = frozenset(
    tuple(sorted([1, 3, 9, 13, 27, 28, 55, 81, 84, 165, 243, 252, 351, extra]))
    for extra in (364, 495, 715)
)
DAMBROSIO_PROFILES = frozenset(tuple(sorted(row)) for row in (
    (1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 17, 20, 21, 32, 34, 40, 42, 48, 51),
    (1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 17, 20, 21, 32, 34, 40, 48, 51, 60),
    (1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 17, 20, 21, 32, 34, 40, 48, 51, 63),
    (1, 2, 3, 4, 5, 8, 10, 12, 16, 17, 20, 34, 40, 42, 48, 51, 58, 60, 63),
))
RESTRICTED_POINTS = (
    1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 73, 128, 146,
    192, 219, 256, 292, 320, 365, 384, 438, 448, 511,
)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


@dataclass(frozen=True)
class Report:
    profiles: frozenset
    jobs: int
    nodes: int


def validate_profiles(text, *, binary, rank, budget, max_solutions,
                      count=0, cases=tuple(range(35)), expected_profiles=None,
                      jobs=None, nodes=None):
    """Fail closed for named presets; use a conservative aggregate budget check.

    Total nodes below the per-job budget rules out even an unreported budget
    stop after finding a solution. The union of all profiles below the per-job
    solution limit rules out any job reaching that limit. Custom CLI runs do
    not call this validator or claim exhaustive reproduction.
    """
    require("self-test passed" in text, "missing enumerator self-test result")
    printed = [tuple(sorted(map(int, m.split())))
               for m in re.findall(r"^  profile:([ 0-9]+)$", text, re.M)]
    profiles = frozenset(printed)
    require(len(profiles) == len(printed), "duplicate profile output")
    require(len(profiles) == count, f"expected {count} profiles, got {len(profiles)}")
    require(all(len(p) == rank for p in profiles), "unexpected profile length")
    if expected_profiles is not None:
        require(profiles == expected_profiles, "profile set differs from the reference")
    if binary:
        summaries = re.findall(
            r"^summary: feasible=(\d+) infeasible=(\d+) over_budget=(\d+) of (\d+)$",
            text, re.M)
        require(len(summaries) == 1, "missing or duplicate binary summary")
        feasible, infeasible, over, total = map(int, summaries[0])
        rows = re.findall(
            r"^case (\d+) \|W\|=\d+: (INFEASIBLE|FEASIBLE|OVER BUDGET) "
            r"jobs=(\d+) nodes=(\d+) profiles=(\d+) cpu=", text, re.M)
        require(sorted(int(r[0]) for r in rows) == sorted(cases),
                "missing, duplicate, or unexpected outer cases")
        require(total == len(cases) and feasible + infeasible + over == total,
                "inconsistent binary summary")
        require(over == 0 and all(r[1] != "OVER BUDGET" for r in rows),
                "enumeration exceeded its budget")
        require(feasible == sum(r[1] == "FEASIBLE" for r in rows),
                "inconsistent feasible case count")
        require(infeasible == sum(r[1] == "INFEASIBLE" for r in rows),
                "inconsistent infeasible case count")
        require(all((r[1] == "FEASIBLE") == (int(r[4]) > 0) for r in rows),
                "verdict disagrees with profile count")
        require(sum(int(r[4]) for r in rows) == count, "inconsistent profile counts")
        actual_jobs = sum(int(r[2]) for r in rows)
        actual_nodes = sum(int(r[3]) for r in rows)
    else:
        summaries = re.findall(
            r"^r=(\d+): (INFEASIBLE|FEASIBLE|OVER BUDGET) jobs=(\d+) "
            r"nodes=(\d+) profiles=(\d+) over_budget_jobs=(\d+) time=", text, re.M)
        require(len(summaries) == 1, "missing or duplicate generic summary")
        r, verdict, j, n, c, over = summaries[0]
        require(int(r) == rank and int(c) == count, "unexpected rank or profile count")
        require(int(over) == 0, "enumeration exceeded its budget")
        require(verdict == ("FEASIBLE" if count else "INFEASIBLE"),
                "unexpected enumeration verdict")
        actual_jobs, actual_nodes = int(j), int(n)
    require(actual_nodes < budget,
            "cannot establish exhaustion: total nodes reach the per-job budget")
    require(count < max_solutions, "cannot establish exhaustion: solution limit reached")
    if jobs is not None:
        require(actual_jobs == jobs, f"expected {jobs} jobs, got {actual_jobs}")
    if nodes is not None:
        require(actual_nodes == nodes, f"expected {nodes} nodes, got {actual_nodes}")
    return Report(profiles, actual_jobs, actual_nodes)


def validate_table(text, records=None, pairs=None):
    matches = re.findall(
        r"^antitone check: subspaces=(\d+) pairs=(\d+) violations=(\d+)$", text, re.M)
    require(len(matches) == 1, "missing or duplicate monotonicity summary")
    n, p, violations = map(int, matches[0])
    require(n > 0 and violations == 0, "empty or non-monotone table")
    if records is not None:
        require(n == records, f"expected {records} subspaces, got {n}")
    if pairs is not None:
        require(p == pairs, f"expected {pairs} covering pairs, got {p}")


def table_histogram(path, p, dimension):
    """Read the legacy little-endian table; never accept a truncated record."""
    histogram = defaultdict(Counter)
    with open(path, "rb") as stream:
        while raw := stream.read(1):
            dim = raw[0]
            require(dim <= dimension, "invalid constraint dimension")
            payload = stream.read(2 * dim + 1)
            require(len(payload) == 2 * dim + 1, "truncated table record")
            rows = struct.unpack("<" + "H" * dim, payload[:-1])
            require(all(0 < row < p ** dimension for row in rows), "invalid row code")
            histogram[dim][payload[-1]] += 1
    require(bool(histogram), "empty table")
    return "\n".join(f"{dim} {sorted(histogram[dim].items())}" for dim in sorted(histogram))


def rank23_lists(path):
    """Extract the full-tensor decomposition, without hard-coding its orbit index."""
    with open(path) as stream:
        text = stream.read()
    blocks = re.findall(r"constrained_tensors\s*\{(.*?)^\}", text, re.M | re.S)
    # Protobuf text omits the empty constraints field by default.
    full = [b for b in blocks if not re.search(r'^\s*constraints:', b, re.M)
            or re.search(r'^\s*constraints:\s*""\s*$', b, re.M)]
    require(len(full) == 1, "expected one unconstrained certificate entry")
    proof = re.search(r'rank_upper_bound_proof:\s*"([^"\\]*)"', full[0])
    require(proof is not None, "certificate lacks a readable rank-23 decomposition")
    terms = re.findall(r"\(([^)]*)\)\*\(([^)]*)\)\*\(([^)]*)\)", proof[1])
    require(len(terms) == 23, "expected a 23-term decomposition")
    lists = []
    for axis, letter in enumerate("abc"):
        factors = []
        for term in terms:
            symbols = term[axis].split("+")
            require(all(re.fullmatch(letter + r"[0-8]", s) for s in symbols),
                    "unexpected decomposition factor syntax")
            require(len(set(symbols)) == len(symbols), "duplicate factor coordinate")
            factors.append(sum(1 << int(s[1:]) for s in symbols))
        lists.append(",".join(map(str, factors)))
    return lists


def ternary_restrictions():
    """Reproduce active sets, not the paper's mathematical hand exclusion."""
    e1, e2, f, g = (1, 0), (0, 1), (1, 1), (1, 2)
    c1, c2, c3, c4 = (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)
    common = [(x, c) for x in (e1, e2) for c in (c1, c2, c3, c4)]
    common += [(f, c1), (f, c2), (f, c3), (g, c1), (g, c2)]
    profiles = {"P1": common + [(f, c4)], "P2": common + [(g, c3)],
                "P3": common + [(g, c4)]}
    zs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 2),
          (1, 2, 0), (1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1),
          (1, 1, 2), (1, 2, 1), (2, 1, 1)]
    tight = {"P1": {c3}, "P2": {c1, c2, c3}, "P3": {c3}}
    seven = {c1, c2, (0, 1, 2), (1, 0, 2)}
    lines = []
    for name, profile in profiles.items():
        groups = defaultdict(set)
        for z in zs:
            active = [term for term in profile if sum(a*b for a, b in zip(term[1], z)) % 3]
            groups[len(active)].add(z)
            if len(active) <= 7:
                lines.append(f"{name} z0={z} active={len(active)} excess={len(active)-6} {active}")
        require(not any(n < 6 for n in groups), "unexpected restriction below six terms")
        require(groups[6] == tight[name], f"unexpected tight restrictions for {name}")
        if name != "P2":
            require(groups[7] == seven, f"unexpected seven-term restrictions for {name}")
    return "\n".join(lines)
