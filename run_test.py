"""Fast runner tests: python3 -m unittest -v run_test.

Subprocesses are mocked except for a tiny logging/failure-propagation test.
No certificate search or paper-scale enumeration is launched by this suite.
"""

import contextlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import run
from profiles import reproduction as rep


class RecordingRunner(run.Runner):
    def __init__(self, root):
        super().__init__(root, dry_run=True)
        self.commands = []

    def run(self, command):
        self.commands.append(list(map(str, command)))
        return ""


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def preview(self, *argv):
        runner = RecordingRunner(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            run.execute(run.parse_args(list(argv)), runner)
        return runner.commands

    def test_every_command_and_profile_recipe(self):
        for recipe in run.RECIPES:
            for command in ("search-cert", "verify-cert", "build-table", "verify-table"):
                with self.subTest(recipe=recipe, command=command):
                    self.assertTrue(self.preview(command, recipe))
        for recipe, checks in run.CHECKS.items():
            self.assertTrue(self.preview("profile", recipe))
            for check in checks:
                self.preview("cross-check", recipe, check)

    def test_search_preserves_named_passes_and_safe_default(self):
        binary = self.preview("search-cert", "q02_n333")
        self.assertIn("//subspace_bounds/upper_bound:rank_upper_bound_main", binary[0])
        self.assertIn(str(self.root / "tmp/search/cert_matrix_q02_n333.pb.txt"), binary[1])
        lower = [c for c in binary if "--ignore_rank_lower_bound" in c]
        self.assertEqual(len(lower), 1)
        self.assertIn("--rank1span_max_subspaces=20000000000000", lower[0])
        self.assertIn("--regenerate_backtracking_proofs", binary[-1])
        ternary = self.preview("search-cert", "q03_n233")
        self.assertFalse(any("upper_bound" in str(c) for c in ternary))
        self.assertEqual(len(ternary), 6)  # build, enumerate, three passes, archive
        self.assertIn("--dim_min=3", ternary[3])
        self.assertIn("--dim_max=2", ternary[4])

    def test_field_selection_and_independent_steps(self):
        for command, target in (("verify-cert", run.VERIFIER), ("build-table", run.TABLE),
                                ("verify-table", run.CHECK_TABLE)):
            cmds = self.preview(command, "q03_n233")
            self.assertEqual(len(cmds), 2)
            self.assertIn(target, cmds[0])
            self.assertIn("-DCP_P=3,-DCP_N0=2,-DCP_N1=3,-DCP_N2=3", cmds[0][3])
        cmds = self.preview("profile", "q03_n233", "--threads", "2")
        self.assertEqual(len(cmds), 2)
        self.assertEqual(cmds[0], ["bazel", "build", "--config=opt", run.GENERIC_PROFILE])
        self.assertIn("--threads=2", cmds[1])
        self.assertIn("--split_depth=2", cmds[1])
        self.assertIn("--max_solutions=1000", cmds[1])

    def test_cross_check_flags(self):
        commands = self.preview("cross-check", "q02_n333", "no-symmetry")
        self.assertIn("--budget=1000000000000", commands[-1])
        self.assertIn("--symmetry=false", commands[-1])
        commands = self.preview("cross-check", "q03_n233", "no-symmetry")
        self.assertIn("--split_depth=4", commands[-1])
        commands = self.preview("cross-check", "q02_n333", "symmetry")
        self.assertIn("--symmetry=true", commands[1])
        self.assertIn("--symmetry=false", commands[2])
        self.assertIn("--max_solutions=1000000", commands[1])
        commands = self.preview("cross-check", "q02_n333", "split")
        self.assertIn("--split_depth=0", commands[1])
        self.assertIn("--split_depth=3", commands[2])

    def test_verify_cert_path_and_overrides(self):
        path = self.root / "space here/cert_matrix_q03_n233.pb.txt"
        commands = self.preview("verify-cert", str(path))
        self.assertEqual(commands[-1][-1], str(path.resolve()))
        self.assertEqual(run.archive_path(path).name, "cert_matrix_q03_n233.btp")
        cmds = self.preview("build-table", "q03_n233", "--certificate", "relative.pb.txt",
                            "--table", "out table.bin")
        self.assertEqual(cmds[-1][-2:], [str(Path("relative.pb.txt").resolve()),
                                       str(Path("out table.bin").resolve())])

    def test_dry_run_no_subprocesses_or_files(self):
        with mock.patch.object(run.subprocess, "Popen") as popen, \
             mock.patch.object(run.subprocess, "run") as process, \
             contextlib.redirect_stdout(io.StringIO()):
            for argv in (("search-cert", "q03_n233", "--dry-run"),
                         ("--dry-run", "verify-cert", "q03_n233"),
                         ("build-table", "q03_n233", "--dry-run"),
                         ("verify-table", "q03_n233", "--histogram", "--dry-run"),
                         ("profile", "q03_n233", "--dry-run"),
                         ("cross-check", "q02_n333", "positive", "--dry-run")):
                args = run.parse_args(list(argv))
                self.assertTrue(args.dry_run)
                runner = run.Runner(self.root, args.dry_run)
                run.execute(args, runner)
                runner.finish(True)
            popen.assert_not_called()
            process.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_inputs_fail_before_build(self):
        for command, hint in (("verify-cert", "search-cert"), ("build-table", "search-cert"),
                              ("profile", "build-table"), ("verify-table", "build-table")):
            runner = run.Runner(self.root)
            with self.subTest(command=command), mock.patch.object(runner, "build") as build:
                with self.assertRaisesRegex(ValueError, hint):
                    run.execute(run.parse_args([command, "q03_n233"]), runner)
                build.assert_not_called()

    def test_overwrite_protection_including_archive(self):
        certificate = self.root / "cert_matrix_q03_n233.pb.txt"
        archive = run.archive_path(certificate)
        archive.write_bytes(b"existing archive")
        with self.assertRaisesRegex(ValueError, "output exists"):
            self.preview("search-cert", "q03_n233", "--certificate", str(certificate))
        self.preview("search-cert", "q03_n233", "--certificate", str(certificate), "--overwrite")
        certificate.write_bytes(b"certificate")
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            self.preview("build-table", "q03_n233", "--certificate", str(certificate),
                         "--table", str(certificate), "--overwrite")
        self.assertEqual(archive.read_bytes(), b"existing archive")

    def test_invalid_cli_and_custom_run(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (("search-all",), ("search", "q03_n233"), ("verify", "q03_n233"),
                         ("profile", "q03_n233", "--threads", "0"),
                         ("verify-cert", "q03_n233", "--", "--r=22"),
                         ("profile", "q03_n233", "--", "--bin=wrong.bin")):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    run.parse_args(list(argv))
        with self.assertRaisesRegex(ValueError, "unknown recipe"):
            self.preview("profile", "unknown")
        with self.assertRaisesRegex(ValueError, "support only"):
            self.preview("profile", "q02_n222")
        with self.assertRaisesRegex(ValueError, "unsupported check"):
            self.preview("cross-check", "q03_n233", "positive")
        cmds = self.preview("profile", "q02_n333", "--", "--list_only")
        self.assertEqual(cmds[-1][-1], "--list_only")
        runner = run.Runner(self.root)
        with mock.patch.object(runner, "run", return_value="custom output"), \
             mock.patch.object(rep, "validate_profiles") as validate, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            run.enumerate_profiles(runner, run.RECIPES["q03_n233"], "table.bin",
                run.profile_options(run.RECIPES["q03_n233"], 1), {}, ["--list_only"])
            validate.assert_not_called()
            self.assertIn("CUSTOM RUN", output.getvalue())

    def test_logging_and_subprocess_failure(self):
        runner = run.Runner(self.root)
        args = run.parse_args(["profile", "q03_n233"])
        data = self.root / "input.bin"
        data.write_bytes(b"input")
        with contextlib.redirect_stdout(io.StringIO()):
            runner.begin(args, run.RECIPES["q03_n233"], [data])
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                runner.run([sys.executable, "-c", "print('partial output'); raise SystemExit(7)"])
            runner.finish(False)
        log = Path(runner.log.name).read_text()
        for expected in (run.digest(data), "partial output", "exit_status=7", "runner_status=FAIL"):
            self.assertIn(expected, log)


def generic_output(nodes=7024, profiles=rep.TERNARY_PROFILES):
    profiles = list(profiles)
    return ("self-test passed\n"
            f"r=14: FEASIBLE jobs=3 nodes={nodes} profiles={len(profiles)} "
            "over_budget_jobs=0 time=1.0s\n" + "".join(
                "  profile: " + " ".join(map(str, p)) + "\n" for p in profiles))


class ValidationTest(unittest.TestCase):
    def generic(self, text, **overrides):
        kwargs = dict(binary=False, rank=14, budget=10**9, max_solutions=1000,
                      count=3, expected_profiles=rep.TERNARY_PROFILES, jobs=3, nodes=7024)
        kwargs.update(overrides)
        return rep.validate_profiles(text, **kwargs)

    def test_generic_reordered_output(self):
        text = generic_output(profiles=reversed(sorted(rep.TERNARY_PROFILES)))
        self.assertEqual(self.generic(text).profiles, rep.TERNARY_PROFILES)

    def test_missing_wrong_or_incomplete_results(self):
        good = generic_output()
        for bad in (good.replace("r=14:", "r=15:"), good.replace("jobs=3", "jobs=4"),
                    good.replace("over_budget_jobs=0", "over_budget_jobs=1"),
                    good.replace("self-test passed", ""), good.replace("  profile:", "other:", 1),
                    good.replace("351", "352"), "partial output"):
            with self.subTest(output=bad), self.assertRaises(rep.ValidationError):
                self.generic(bad)
        with self.assertRaisesRegex(rep.ValidationError, "budget"):
            self.generic(good, budget=7024)
        with self.assertRaisesRegex(rep.ValidationError, "solution limit"):
            self.generic(good, max_solutions=3)
        # A FEASIBLE verdict and zero reported budget stops do not suffice.
        with self.assertRaisesRegex(rep.ValidationError, "budget"):
            self.generic(generic_output(nodes=10**9 + 1), nodes=None)

    def test_binary_cases(self):
        text = ("self-test passed\n" + "".join(
            f"case {i} |W|=1: INFEASIBLE jobs=1 nodes=2 profiles=0 cpu=1.0s\n"
            for i in reversed(range(35))) +
            "summary: feasible=0 infeasible=35 over_budget=0 of 35\n")
        kwargs = dict(binary=True, rank=20, budget=100, max_solutions=50,
                      jobs=35, nodes=70)
        self.assertEqual(rep.validate_profiles(text, **kwargs).nodes, 70)
        for bad in (text.replace("case 34", "case 33"), text.replace("infeasible=35", "infeasible=34"),
                    text.replace("INFEASIBLE", "OVER BUDGET", 1)):
            with self.assertRaises(rep.ValidationError):
                rep.validate_profiles(bad, **kwargs)

    def test_monotonicity_summary(self):
        good = "antitone check: subspaces=56632 pairs=969696 violations=0\n"
        rep.validate_table(good, 56632, 969696)
        for bad in ("", good.replace("violations=0", "violations=1"),
                    good.replace("56632", "56631"), good + good):
            with self.assertRaises(rep.ValidationError):
                rep.validate_table(bad, 56632, 969696)

    def test_histogram_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.bin"
            path.write_bytes(bytes([0, 7, 1]) + struct.pack("<H", 1) + bytes([6]))
            self.assertEqual(rep.table_histogram(path, 2, 4), "0 [(7, 1)]\n1 [(6, 1)]")
            for data in (b"", b"\x05", b"\x01\x01", b"\x01\x00\x00\x06"):
                path.write_bytes(data)
                with self.assertRaises(rep.ValidationError):
                    rep.table_histogram(path, 2, 4)

    def test_restrictions_and_decomposition(self):
        text = rep.ternary_restrictions()
        self.assertEqual(text.count("active=6"), 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cert.pb.txt"
            proof = " + ".join(["(a0)*(b1)*(c2)"] * 23)
            for constraints in ("", '  constraints: ""\n'):
                path.write_text('constrained_tensors {\n' + constraints +
                                f'  rank_upper_bound_proof: "{proof}"\n}}\n')
                self.assertEqual(rep.rank23_lists(path), [",".join([x]*23) for x in ("1", "2", "4")])


if __name__ == "__main__":
    unittest.main()
