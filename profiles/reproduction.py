"""Pure validation and supporting calculations for the paper workflows.

These helpers check recorded outcomes, not the mathematical completeness proofs.
Enumeration completeness is reported explicitly by the C++ result schema.
"""

from collections import defaultdict
from dataclasses import dataclass
import re
import json


class ValidationError(ValueError):
    """A reproduction did not establish its expected outcome."""


TERNARY_PROFILES = frozenset(
    tuple(sorted([1, 3, 9, 13, 27, 28, 55, 81, 84, 165, 243, 252, 351, extra]))
    for extra in (364, 495, 715)
)
DAMBROSIO_PROFILES = frozenset(
    tuple(sorted(row))
    for row in (
        (1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 17, 20, 21, 32, 34, 40, 42, 48, 51),
        (1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 17, 20, 21, 32, 34, 40, 48, 51, 60),
        (1, 2, 3, 4, 5, 8, 10, 12, 15, 16, 17, 20, 21, 32, 34, 40, 48, 51, 63),
        (1, 2, 3, 4, 5, 8, 10, 12, 16, 17, 20, 34, 40, 42, 48, 51, 58, 60, 63),
    )
)
RESTRICTED_POINTS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    73,
    128,
    146,
    192,
    219,
    256,
    292,
    320,
    365,
    384,
    438,
    448,
    511,
)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


@dataclass(frozen=True)
class Report:
    profiles: frozenset
    jobs: int
    nodes: int


def validate_profiles(
    text,
    *,
    binary,
    rank,
    budget,
    max_solutions,
    count=0,
    cases=tuple(range(35)),
    expected_profiles=None,
    jobs=None,
    nodes=None,
    restricted=False,
):
    """Validate the versioned result record; progress text is not an API."""
    records = re.findall(r"^result_json: (.+)$", text, re.M)
    require(len(records) == 1, "missing or duplicate enumeration result")
    try:
        result = json.loads(records[0])
        require(result["schema_version"] == 1, "unsupported result schema")
        require(result["self_test_passed"] is True, "missing enumerator self-test result")
        require(result["rank"] == rank, "unexpected rank")
        require(result["scope"] == ("restricted" if restricted else "full"),
                "unexpected enumeration scope")
        for key in ("jobs", "finished_jobs", "nodes", "budget_stops", "solution_limit_stops"):
            require(type(result[key]) is int and result[key] >= 0, f"invalid {key}")
        require(result["budget_stops"] == 0, "enumeration exceeded its node budget")
        require(result["solution_limit_stops"] == 0, "enumeration reached a solution limit")
        require(result["exhausted"] is True, "enumeration was not exhausted")
        require(result["finished_jobs"] == result["jobs"], "unfinished enumeration jobs")
        printed = result["profiles"]
        require(isinstance(printed, list), "invalid profile array")
        require(all(isinstance(p, list) and len(p) == rank and
                    all(type(x) is int and x > 0 for x in p) for p in printed),
                "invalid profile length or point code")
        profiles = frozenset(tuple(sorted(p)) for p in printed)
        require(len(profiles) == len(printed), "duplicate profile output")
        require(len(profiles) == count, f"expected {count} profiles, got {len(profiles)}")
        if expected_profiles is not None:
            require(profiles == expected_profiles, "profile set differs from the reference")
        rows = result["cases"]
        if binary:
            require(sorted(r["id"] for r in rows) == sorted(cases),
                    "missing, duplicate, or unexpected outer cases")
            for row in rows:
                for key in ("jobs", "finished_jobs", "nodes", "profile_count", "budget_stops", "solution_limit_stops"):
                    require(type(row[key]) is int and row[key] >= 0, f"invalid case {key}")
                require(row["budget_stops"] == row["solution_limit_stops"] == 0 and
                        row["jobs"] == row["finished_jobs"], "incomplete outer case")
            for key in ("jobs", "finished_jobs", "nodes", "budget_stops", "solution_limit_stops"):
                require(sum(r[key] for r in rows) == result[key], f"inconsistent total {key}")
            require(sum(r["profile_count"] for r in rows) == count, "inconsistent profile counts")
        else:
            require(rows == [], "unexpected outer cases in generic result")
        actual_jobs, actual_nodes = result["jobs"], result["nodes"]
        if jobs is not None:
            require(actual_jobs == jobs, f"expected {jobs} jobs, got {actual_jobs}")
        if nodes is not None:
            require(actual_nodes == nodes, f"expected {nodes} nodes, got {actual_nodes}")
        return Report(profiles, actual_jobs, actual_nodes)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValidationError(f"malformed enumeration result: {error}") from error


def validate_table(text, records=None, pairs=None):
    matches = re.findall(
        r"^antitone check: subspaces=(\d+) pairs=(\d+) violations=(\d+)$", text, re.M
    )
    require(len(matches) == 1, "missing or duplicate monotonicity summary")
    n, p, violations = map(int, matches[0])
    require(n > 0 and violations == 0, "empty or non-monotone table")
    if records is not None:
        require(n == records, f"expected {records} subspaces, got {n}")
    if pairs is not None:
        require(p == pairs, f"expected {pairs} covering pairs, got {p}")


def rank23_lists(path):
    """Extract the full-tensor decomposition, without hard-coding its orbit index."""
    with open(path) as stream:
        text = stream.read()
    blocks = re.findall(r"constrained_tensors\s*\{(.*?)^\}", text, re.M | re.S)
    # Protobuf text omits the empty constraints field by default.
    full = [
        b
        for b in blocks
        if not re.search(r"^\s*constraints:", b, re.M)
        or re.search(r'^\s*constraints:\s*""\s*$', b, re.M)
    ]
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
            require(
                all(re.fullmatch(letter + r"[0-8]", s) for s in symbols),
                "unexpected decomposition factor syntax",
            )
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
    profiles = {
        "P1": common + [(f, c4)],
        "P2": common + [(g, c3)],
        "P3": common + [(g, c4)],
    }
    zs = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 2),
        (1, 2, 0),
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (2, 1, 1),
    ]
    tight = {"P1": {c3}, "P2": {c1, c2, c3}, "P3": {c3}}
    seven = {c1, c2, (0, 1, 2), (1, 0, 2)}
    lines = []
    for name, profile in profiles.items():
        groups = defaultdict(set)
        for z in zs:
            active = [
                term for term in profile if sum(a * b for a, b in zip(term[1], z)) % 3
            ]
            groups[len(active)].add(z)
            if len(active) <= 7:
                lines.append(
                    f"{name} z0={z} active={len(active)} excess={len(active)-6} {active}"
                )
        require(
            not any(n < 6 for n in groups), "unexpected restriction below six terms"
        )
        require(groups[6] == tight[name], f"unexpected tight restrictions for {name}")
        if name != "P2":
            require(
                groups[7] == seven, f"unexpected seven-term restrictions for {name}"
            )
    return "\n".join(lines)
