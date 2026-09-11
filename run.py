#!/usr/bin/env python3
"""Independent, named steps for reproducing the paper's computations.

Certificate search and binary certificate verification can take hours.
Neither is implicit in other commands. Use --list-recipes and COMMAND --help.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

# Even a first dry run must not create local import-cache files.
sys.dont_write_bytecode = True
from profiles import reproduction as rep

REPO_ROOT = Path(__file__).resolve().parent
SEARCH = "//subspace_bounds/search:"
VERIFIER = "//subspace_bounds/verifier:verifier_main"
CHECK_TABLE = "//subspace_bounds:check_table_main"
BINARY_PROFILE = "//profiles:profile_enum_q02_n333_main"
GENERIC_PROFILE = "//profiles:profile_enum_fp_main"


@dataclass(frozen=True)
class Recipe:
    p: int
    n0: int
    n1: int
    n2: int
    passes: tuple
    upper: object = None
    rank: int = 0
    records: int = 0
    pairs: int = 0

    @property
    def name(self):
        return f"q{self.p:02d}_n{self.n0}{self.n1}{self.n2}"

    @property
    def certificate_name(self):
        return f"cert_matrix_{self.name}.pb.txt"


# Former search_matrix_all recipes, explicitly selectable. Search can produce
# different proof records; verification remains separate.
RECIPES = {
    r.name: r
    for r in (
        Recipe(2, 2, 2, 2, ({},), {}),
        Recipe(2, 2, 2, 3, ({},), {}),
        Recipe(2, 2, 2, 4, ({},), {}),
        Recipe(2, 2, 3, 3, ({"backtracking_step_limit": 100_000},), {}),
        Recipe(
            2,
            3,
            2,
            4,
            (
                {
                    "backtracking_step_limit": 10_000_000,
                    "forced_product_max_iterations_log2": 32,
                },
            ),
            {},
            rank=19,
            records=2825,
            pairs=23562,
        ),
        Recipe(
            2,
            3,
            3,
            3,
            (
                {
                    "backtracking_step_limit": 100_000_000,
                    "rank1span_max_subspaces": 20_000_000_000_000,
                },
            ),
            {
                "last_only": True,
                "path_limit": 1_000_000,
                "max_steps_at_a_rank": 1000,
                "num_paths": 1000,
            },
            rank=20,
            records=8_283_458,
            pairs=213_188_689,
        ),
        Recipe(
            3,
            2,
            3,
            3,
            (
                {
                    "dim_min": 3,
                    "backtracking_step_limit": 100_000_000,
                    "backtracking_max_map_size": 1_000_000_000,
                    "rank1span_max_subspaces": 1_000_000_000,
                },
                {
                    "dim_max": 2,
                    "backtracking_step_limit": 100_000_000,
                    "backtracking_max_map_size": 1_000_000_000,
                },
            ),
            rank=14,
            records=56632,
            pairs=969696,
        ),
    )
}
CHECKS = {
    "q02_n333": ("positive", "symmetry", "no-symmetry", "split"),
    "q03_n233": ("no-symmetry", "restrictions"),
    "q02_n324": (),
}


def chosen_problem_copt(recipe):
    values = (recipe.p, recipe.n0, recipe.n1, recipe.n2)
    defines = ",".join(f"-DCP_{k}={v}" for k, v in zip(("P", "N0", "N1", "N2"), values))
    return r"--per_file_copt=.*_main\.cc@" + defines


def flags(options):
    return [
        f"--{k}={str(v).lower() if isinstance(v, bool) else v}"
        for k, v in options.items()
    ]


def executable(target):
    return "bazel-bin/" + target.removeprefix("//").replace(":", "/")


def parse_cert(path):
    match = re.fullmatch(r"cert_matrix_q(\d+)_n(\d)(\d)(\d)\.pb\.txt", Path(path).name)
    if not match:
        raise ValueError(f"unrecognised certificate name: {path}")
    p, n0, n1, n2 = map(int, match.groups())
    if p < 2 or any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
        raise ValueError("only prime fields are supported")
    if min(n0, n1, n2) < 1:
        raise ValueError("matrix dimensions must be positive")
    return Recipe(p, n0, n1, n2, ())


def archive_path(certificate):
    return certificate.with_name(certificate.name.removesuffix(".pb.txt") + ".btp")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Runner:
    """Stream output and capture one auditable log per invocation."""

    def __init__(self, root, dry_run=False):
        self.root = Path(root)
        self.dry_run = dry_run
        self.log = None
        self.started = time.monotonic()

    def begin(self, args, recipe, inputs):
        if self.dry_run:
            return
        directory = self.root / "log" / "reproduction"
        directory.mkdir(parents=True, exist_ok=True)
        self.log = tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"{args.command}-{recipe.name}-",
            suffix=".log",
            dir=directory,
            delete=False,
        )
        print(f"Log: {self.log.name}", flush=True)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.note(
            json.dumps(
                {
                    "recipe": recipe.name,
                    "arguments": vars(args),
                    "commit": commit.stdout.strip(),
                    "worktree_dirty": bool(status.stdout.strip()),
                    "inputs_sha256": {str(p): digest(p) for p in inputs},
                },
                default=str,
                sort_keys=True,
            )
        )

    def note(self, message):
        if self.log:
            self.log.write(message + "\n")
            self.log.flush()

    def message(self, message):
        print(message, flush=True)
        self.note(message)

    def run(self, command):
        command = list(map(str, command))
        self.message("+ " + shlex.join(command))
        if self.dry_run:
            return ""
        start, lines = time.monotonic(), []
        with subprocess.Popen(
            command,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as child:
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    self.note(line.rstrip("\n"))
                    lines.append(line)
                status = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        self.note(f"exit_status={status} elapsed_seconds={time.monotonic()-start:.3f}")
        if status:
            raise RuntimeError(
                f"command exited with status {status}: {shlex.join(command)}"
            )
        return "".join(lines)

    def build(self, recipe, *targets):
        command = ["bazel", "build", "--config=opt", chosen_problem_copt(recipe)]
        self.run(command + list(targets))

    def finish(self, success):
        self.note(
            f"runner_status={'PASS' if success else 'FAIL'} "
            f"elapsed_seconds={time.monotonic()-self.started:.3f}"
        )
        if self.log:
            self.log.close()


def require_input(path, hint, dry_run):
    if path.is_file():
        return
    if dry_run:
        print(f"Prerequisite: {path}; create with: {hint}")
    else:
        raise ValueError(f"missing input: {path}\nCreate it with: {hint}")


def protect_output(path, inputs, overwrite):
    if any(path.resolve() == p.resolve() for p in inputs):
        raise ValueError(f"output would overwrite an input: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"output is not a regular file: {path}")
    if path.exists() and not overwrite:
        raise ValueError(
            f"output exists: {path}; select another path or use --overwrite"
        )


def profile_options(recipe):
    if recipe.name == "q02_n333":
        return {
            "r": 20,
            "prop3": False,
            "min_w": 0,
            "split_depth": 4,
            "budget": 100_000_000,
            "max_solutions": 50,
        }
    return {
        "r": recipe.rank,
        "max_solutions": 1000,
        "split_depth": 2 if recipe.p == 3 else 0,
        "budget": 1_000_000_000,
    }


def standard_expectation(recipe):
    if recipe.name == "q02_n333":
        return {"count": 0, "jobs": 50830, "nodes": 15658474}
    if recipe.name == "q03_n233":
        return {
            "count": 3,
            "jobs": 3,
            "nodes": 7024,
            "expected_profiles": rep.TERNARY_PROFILES,
        }
    return {
        "count": 4,
        "jobs": 1,
        "nodes": 454831,
        "expected_profiles": rep.DAMBROSIO_PROFILES,
    }


def enumerate_profiles(runner, recipe, certificate, options, expectation, extra=()):
    binary = recipe.name == "q02_n333"
    target = BINARY_PROFILE if binary else GENERIC_PROFILE
    if extra:
        runner.message(
            "CUSTOM RUN: extra flags disable paper-result and exhaustion validation."
        )
    output = runner.run(
        [executable(target), f"--certificate={certificate}"]
        + flags(options)
        + list(extra)
    )
    if runner.dry_run:
        return None
    if extra:
        runner.message(
            "CUSTOM RUN: subprocess completed; no paper-result or exhaustion validation."
        )
        return None
    report = rep.validate_profiles(
        output,
        binary=binary,
        rank=options["r"],
        budget=options["budget"],
        max_solutions=options["max_solutions"],
        restricted=bool(options.get("experimental", False) or options.get("ones_subset")),
        **expectation,
    )
    runner.message("PASS: expected profiles/counts and explicit enumeration exhaustion.")
    return report


def cross_check(runner, recipe, certificate, args):
    if args.check == "restrictions":
        if runner.dry_run:
            runner.message(
                "Would compute and check the P1/P2/P3 column-restriction active sets."
            )
        else:
            runner.message(rep.ternary_restrictions())
            runner.message(
                "PASS: supporting restriction counts (not an automated hand proof)."
            )
        return
    options = profile_options(recipe)
    runner.build(recipe, BINARY_PROFILE if recipe.p == 2 else GENERIC_PROFILE)
    if args.check == "positive":
        if runner.dry_run:
            runner.message(
                f"Would extract the three rank-23 factor lists from {certificate}."
            )
            lists = [f"<rank23-axis-{axis}>" for axis in "abc"]
        else:
            lists = rep.rank23_lists(certificate)
        for factors in lists:
            output = runner.run(
                [
                    executable(BINARY_PROFILE),
                    f"--certificate={certificate}",
                    "--r=23",
                    f"--check_list={factors}",
                ]
            )
            if not runner.dry_run:
                rep.require(
                    re.findall(r"^check_list: (PASS|FAIL)$", output, re.M) == ["PASS"],
                    "positive control did not pass",
                )
        return
    if args.check == "no-symmetry":
        runner.message(
            "WARNING: full binary no-symmetry enumeration can take hours on a laptop."
            if recipe.p == 2
            else "Running the unrestricted ternary cross-check."
        )
        options.update(symmetry=False, split_depth=4)
        expectation = standard_expectation(recipe)
        if recipe.p == 2:
            options["budget"] = 1_000_000_000_000
            expectation.update(jobs=591061, nodes=2155604077)
        else:
            expectation.update(jobs=82153, nodes=52972264)
        enumerate_profiles(runner, recipe, certificate, options, expectation)
        return
    reports = []
    options["experimental"] = True
    if args.check == "symmetry":
        options.update(
            r=22,
            min_w=8,
            max_w=8,
            cases=0,
            prop3=False,
            split_depth=0,
            budget=1_000_000_000,
            max_solutions=1_000_000,
            ones_subset=",".join(map(str, rep.RESTRICTED_POINTS)),
        )
        for symmetry in (True, False):
            reports.append(
                enumerate_profiles(
                    runner,
                    recipe,
                    certificate,
                    dict(options, symmetry=symmetry),
                    {"count": 232, "cases": (0,)},
                )
            )
    else:
        for depth in (0, 3):
            reports.append(
                enumerate_profiles(
                    runner,
                    recipe,
                    certificate,
                    dict(options, cases=0, split_depth=depth),
                    {"count": 0, "cases": (0,)},
                )
            )
    if not runner.dry_run:
        rep.require(
            reports[0].profiles == reports[1].profiles,
            "cross-check profile sets differ",
        )
        runner.message(
            "PASS: normalized profiles and verdicts agree (timings/node counts ignored)."
        )


def execute(args, runner):
    legacy_path = args.command == "verify-cert" and args.recipe not in RECIPES
    if legacy_path:
        if args.certificate:
            raise ValueError(
                "use either the positional certificate path or --certificate"
            )
        recipe = parse_cert(args.recipe)
        recipe = RECIPES.get(recipe.name, recipe)
    else:
        if args.recipe not in RECIPES:
            raise ValueError(f"unknown recipe {args.recipe!r}; use --list-recipes")
        recipe = RECIPES[args.recipe]
    if args.command in ("profile", "cross-check") and not recipe.rank:
        raise ValueError("profile workflows support only q02_n333, q03_n233, q02_n324")
    if args.command == "cross-check" and args.check not in CHECKS[recipe.name]:
        raise ValueError(
            f"unsupported check; available for {recipe.name}: "
            + (", ".join(CHECKS[recipe.name]) or "use profile for the D'Ambrosio check")
        )
    if (
        args.command == "cross-check"
        and args.check == "restrictions"
        and args.certificate
    ):
        raise ValueError("the restrictions calculation does not consume a certificate")
    certificate = (
        Path(args.recipe).resolve()
        if legacy_path
        else (
            Path(args.certificate).resolve()
            if args.certificate
            else runner.root
            / ("tmp/search" if args.command == "search-cert" else "certs/matrix")
            / recipe.certificate_name
        )
    )
    if not certificate.name.endswith(".pb.txt"):
        raise ValueError(
            "certificate paths must end in .pb.txt (the runner uses text certificates)"
        )
    archive = archive_path(certificate)
    inputs = []
    if args.command in ("verify-cert", "verify-table", "profile") or (
        args.command == "cross-check" and args.check != "restrictions"
    ):
        require_input(
            certificate,
            f"python3 run.py search-cert {recipe.name} --certificate {shlex.quote(str(certificate))}",
            runner.dry_run,
        )
        inputs.append(certificate)
    if args.command == "verify-cert":
        require_input(
            archive,
            "retrieve the matching .btp archive with git lfs pull",
            runner.dry_run,
        )
        inputs.append(archive)
    outputs = [certificate, archive] if args.command == "search-cert" else []
    for path in outputs:
        protect_output(path, inputs, args.overwrite)
    runner.begin(args, recipe, inputs)
    for path in outputs:
        if not runner.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "search-cert":
        runner.message(
            "WARNING: certificate search can take hours on a many-core machine; "
            "verify the resulting certificate separately."
        )
        targets = [SEARCH + "orbit_enumerator_main", SEARCH + "rank_lower_bound_main"]
        if recipe.upper is not None:
            targets.append("//subspace_bounds/upper_bound:rank_upper_bound_main")
        runner.build(recipe, *targets)
        runner.run([executable(targets[0]), "--output_path", certificate])
        if recipe.upper is not None:
            runner.run([executable(targets[2]), certificate] + flags(recipe.upper))
        for i, options in enumerate(recipe.passes):
            runner.run(
                [executable(targets[1]), certificate]
                + flags(options)
                + (["--ignore_rank_lower_bound"] if i == 0 else [])
            )
        runner.message(
            f"Next: python3 run.py verify-cert {recipe.name} --certificate {shlex.quote(str(certificate))}"
        )
    elif args.command == "verify-cert":
        if recipe.name == "q02_n333":
            runner.message(
                "WARNING: binary certificate verification can take hours on a laptop."
            )
        runner.build(recipe, VERIFIER)
        output = runner.run([executable(VERIFIER), certificate])
        if not runner.dry_run:
            rep.require(
                "OK. Verified" in output, "missing certificate verification success"
            )
            if recipe.rank:
                bounds = re.findall(
                    r"UNCONSTRAINED TENSOR RANK LOWER BOUND: (\d+)", output
                )
                rep.require(
                    len(bounds) == 1 and int(bounds[0]) >= recipe.rank,
                    "certificate bound is below the recipe's expected bound",
                )
    elif args.command == "verify-table":
        runner.build(recipe, CHECK_TABLE)
        output = runner.run(
            [executable(CHECK_TABLE), certificate]
            + (["--histogram"] if args.histogram else [])
        )
        if not runner.dry_run:
            rep.validate_table(output, recipe.records or None, recipe.pairs or None)
        runner.message("Monotonicity only: this does not verify certificate proofs.")
    elif args.command == "profile":
        runner.message(
            "Expanding certificate bounds in memory; proof verification and monotonicity checking are separate."
        )
        runner.build(
            recipe, BINARY_PROFILE if recipe.name == "q02_n333" else GENERIC_PROFILE
        )
        enumerate_profiles(
            runner,
            recipe,
            certificate,
            profile_options(recipe),
            standard_expectation(recipe),
            args.extra,
        )
    else:
        cross_check(runner, recipe, certificate, args)
    if not runner.dry_run:
        for path in outputs:
            if path.is_file():
                runner.note(f"output_sha256 {path} {digest(path)}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-recipes", action="store_true", help="list recipes and supported checks"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print steps without building or writing"
    )
    sub = parser.add_subparsers(dest="command")
    descriptions = {
        "search-cert": "generate a certificate (expensive; default output: tmp/search/)",
        "verify-cert": "verify certificate proofs and archive (binary case can take hours)",
        "verify-table": "expand certificate bounds in memory and check monotonicity",
        "profile": "run and validate a documented enumeration using a certificate",
        "cross-check": "run a supporting check; no-symmetry can be expensive",
    }
    for command, description in descriptions.items():
        child = sub.add_parser(command, help=description, description=description)
        child.add_argument(
            "recipe", help="recipe name (verify-cert also accepts a certificate path)"
        )
        child.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
        child.add_argument(
            "--certificate", help="certificate path, relative to caller's directory"
        )
        if command == "search-cert":
            child.add_argument(
                "--overwrite",
                action="store_true",
                help="allow replacement of existing outputs",
            )
        if command == "verify-table":
            child.add_argument(
                "--histogram",
                action="store_true",
                help="also print L by constraint dimension",
            )
        if command == "cross-check":
            child.add_argument(
                "check",
                choices=(
                    "positive",
                    "symmetry",
                    "no-symmetry",
                    "split",
                    "restrictions",
                ),
            )
    extra = []
    if "--" in argv:
        separator = argv.index("--")
        argv, extra = argv[:separator], argv[separator + 1 :]
    if any(x in ("search-all", "search-and-verify-all") for x in argv):
        parser.error(
            "select a named recipe: run.py search-cert RECIPE, then run.py verify-cert PATH; use --list-recipes"
        )
    args = parser.parse_args(argv)
    if args.command and args.list_recipes:
        parser.error("use --list-recipes without a command")
    if not args.command and not args.list_recipes:
        parser.error("select a command or use --list-recipes")
    if extra and args.command != "profile":
        parser.error(
            "extra enumerator arguments after -- are supported only by profile"
        )
    if any(
        re.match(
            r"-{1,2}(?:bin|certificate|p|n0|n1|flagfile|fromenv|tryfromenv)(?:=|$)", x
        )
        for x in extra
    ):
        parser.error(
            "select the problem with a recipe and the input with --certificate, not extra flags"
        )
    for name, default in (
        ("certificate", None),
        ("overwrite", False),
        ("histogram", False),
        ("check", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    args.extra = extra
    return args


def main(argv=None):
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.list_recipes:
        for recipe in RECIPES.values():
            capabilities = "search-cert, verify-cert, verify-table"
            if recipe.rank:
                capabilities += ", profile"
            print(f"{recipe.name}: {capabilities}")
            if CHECKS.get(recipe.name):
                print("  cross-check: " + ", ".join(CHECKS[recipe.name]))
        return 0
    runner = Runner(REPO_ROOT, args.dry_run)
    success = False
    try:
        execute(args, runner)
        success = True
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        runner.message(f"ERROR: {error}")
        return 1
    except KeyboardInterrupt:
        runner.message("Interrupted; the computation has not been validated.")
        return 130
    finally:
        runner.finish(success)


if __name__ == "__main__":
    sys.exit(main())
