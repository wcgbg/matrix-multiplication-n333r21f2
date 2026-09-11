# A Lower Bound of 21 for 3 by 3 Matrix Multiplication over F2

This is the companion repository for the paper 
*A Lower Bound of 21 for 3×3 Matrix Multiplication over 𝔽₂* 
([arXiv:2609.06725](https://arxiv.org/abs/2609.06725)). The paper proves two results
about the tensor rank $\mathbf{R}$ of the matrix-multiplication tensor
$\langle n_0,n_1,n_2\rangle$ (an $n_0\times n_1$ matrix times an
$n_1\times n_2$ matrix) over a small field:

$$\mathbf{R}_{\mathbb{F}_2}(\langle 3,3,3\rangle)\ge 21,\qquad
  \mathbf{R}_{\mathbb{F}_3}(\langle 2,3,3\rangle)= 15 .$$

- **3 by 3 matrix multiplication over $\mathbb{F}_2$ needs at least 21
  multiplications.** The previous lower bound was 20, from Wang's automated
  framework ([arXiv:2603.07280](https://arxiv.org/abs/2603.07280)); before that,
  Bläser's lower bound was 19. Laderman's 23-product algorithm gives the best
  known upper bound.
- **2 by 3 times 3 by 3 matrix multiplication over $\mathbb{F}_3$ has rank
  exactly 15.** The previous lower bound was Bläser's 14. Hopcroft and
  Kerr's algorithm gives an upper bound of 15 and is valid over every field.

Both proofs are computer assisted. They use a machine-checked subspace lower-bound
table and an exhaustive enumeration of first-factor profiles against
the capacities implied by the table. Over $\mathbb{F}_3$, a short argument by hand
excludes the three surviving profiles. Everything needed to rebuild
and re-check them is in this repository.

## The method

Let $T$ be the multiplication tensor and $A$ the space of its first argument
(the $n_0\times n_1$ matrices). For a subspace $S\subseteq A$, $T_S$ is $T$
with its first argument restricted to $S$.

1. **A subspace lower-bound table.** The framework of the cited paper enumerates
   the subspaces $S\subseteq A$ up to the symmetry group of the tensor and
   assigns to every orbit a lower bound $L(S)\le\mathbf{R}(T_S)$ by a dynamic
   program over the orbits, starting with the smallest $S$. Its techniques are
   flattening, degenerate reduction (a lookup of an already-processed larger
   constraint set), forced products, substitution with backtracking, and the
   rank-one-span search added for this paper. The latter asks whether the slice
   space of $T_S$ lies in a subspace of dimension at most $r$ spanned by rank-one matrices. Every
   entry carries a proof record, and a separate verifier re-checks each record.
   The unconstrained entry $L(A)$ is the certified bound on $\mathbf{R}(T)$:
   20 for $\langle 3,3,3\rangle$ over $\mathbb{F}_2$, 14 for
   $\langle 2,3,3\rangle$ over $\mathbb{F}_3$. The table with its proofs is
   the **certificate** (`certs/matrix/`). The tools in `subspace_bounds/`
   produce and verify it, then expand its orbit bounds to every subspace.
2. **Capacities and profiles.** In a decomposition with $r$ terms, restricting
   the first argument to $S$ kills exactly the terms whose first factor
   vanishes on $S$, so at most $r-L(S)$ terms can do so: the *capacity* of
   $S$. The tools in `profiles/` expand certificate bounds in memory and exhaustively
   enumerate the *profiles*: lists of first factors compatible with
   all capacities at once (the strategy of D'Ambrosio's
   proof that the $\mathbb{F}_2$ rank of $\langle 2,3,4\rangle$ is 20). For
   $\langle 3,3,3\rangle$ over $\mathbb{F}_2$, no profile with 20 terms exists,
   so $\mathbf{R}\ge 21$. For $\langle 2,3,3\rangle$ over $\mathbb{F}_3$, three
   profiles with 14 terms survive, and the paper (Section 4) excludes each by
   comparing overlapping restrictions, so $\mathbf{R}\ge 15$.

**Convention.** The paper indexes everything by the restriction space $S$ and
sweeps $\dim S$ from 0 up to $\dim A$. The code, its flags (`--dim_min`,
`--dim_max`), the certificate field `dim`, the expanded tables, and the notes
count *constraints*: a subspace is stored as an echelon basis of its
annihilator $S^\perp\subseteq A^*$, and `dim` there is
$\dim S^\perp=\dim A-\dim S$. A "dim-6 orbit" of $\langle 3,3,3\rangle$ in
the code is a three-dimensional $S$ in the paper.

## Repository layout

The two method directories are `subspace_bounds/` and `profiles/`; both can
use the shared code in `core/` and `matrix/`. Within `subspace_bounds/`,
search and verification remain separate packages with distinct trust roles.
Bazel labels and executable paths follow this layout. The root `run.py`
provides independent commands for both steps and their cross-checks.

| directory | contents | trusted? |
|---|---|---|
| `core/` | field arithmetic (`gf.h`, `gf_vec.h`, `bit_vec.h`), tensors, constraints, the certificate proto and its IO, the backtracking-proof codec, and the recomputers of the self-contained proofs: flattening, forced products, and the rank-one-span engines (`rank_one_span_f2.h`, `rank_one_span_family_search.h`, `rank_lower_bound_rank_one_span_fp.h`) | yes |
| `matrix/` | the $\langle N_0,N_1,N_2\rangle$ tensor, its symmetry groups (`f2_symmetry.h`, table driven; `fp_symmetry.h`, arithmetic) and the build-time problem selection `chosen_problem.h` | yes |
| `subspace_bounds/` | `subspace_bounds.h` expands certificate bounds in memory; `check_table_main` checks monotonicity | table tools are outside the certificate, see [What to trust](#what-to-trust) |
| `subspace_bounds/verifier/` | the certificate checker: `verifier.h`, `backtracking_verifier.h`, `verifier_main.cc` | yes |
| `subspace_bounds/search/` | orbit enumeration, the meet-in-the-middle orbit map, the lower-bound dynamic program (`rank_lower_bound_computer.h`) and the proof generators | no: produces certificates |
| `subspace_bounds/upper_bound/` | the flip-graph search for decompositions ($\mathbb{F}_2$ only), which records rank upper bounds in the certificate | no |
| `profiles/` | the capacity-constrained enumerators `profile_enum_q02_n333_main` and `profile_enum_fp_main` | outside the certificate, see [What to trust](#what-to-trust) |
| `certs/matrix/` | the certificates `cert_matrix_q<p>_n<n0><n1><n2>.pb.txt` and their backtracking archives `.btp` (Git LFS) | data |
| `paper/` | `main.tex`, `refs.bib` | |
| `run.py` | named recipes for certificate search/verification, table expansion/checking, profiles, and supporting cross-checks | orchestration; not an additional proof verifier |

## Build

Prerequisites: [Bazel](https://bazel.build/) 8.3.1 (pinned in `.bazeliskrc`;
Bazelisk works), a C++20 toolchain including `std::format`, Python 3.9 or later,
and [Git LFS](https://git-lfs.com/) installed
before cloning (the certificates and their archives are LFS objects; the
$\langle 3,3,3\rangle$ archive is about 241 MB). The external libraries (protobuf,
gflags, ng-log, Boost, GoogleTest, oneTBB, mimalloc) are fetched by Bazel.

```bash
bazel build --config=opt //...     # release: -O3 -march=native -flto
bazel test --config=opt //...      # unit tests
bazel build --config=debug //...   # asan + ubsan + _GLIBCXX_DEBUG
python3 -m unittest -v run_test release_test # runner, results, artifact checks
```

### Paper workflows through run.py

`python3 run.py --list-recipes` lists the named recipes and cross-checks.
The paper instances are `q02_n333` and `q03_n233`; `q02_n324` is the
D'Ambrosio cross-check. The small certificate-search recipes are available
too, but do not have profile presets. Use `python3 run.py COMMAND --help`.

| command | operation |
|---|---|
| `search-cert RECIPE` | generate a certificate using the named search recipe; potentially hours on a many-core machine |
| `verify-cert RECIPE` | verify certificate proofs and the matching archive; the binary case can take hours on a laptop |
| `verify-table RECIPE --histogram` | expand certificate bounds in memory, check monotonicity, and optionally print bounds by constraint dimension |
| `profile RECIPE` | enumerate profiles using the documented settings and validate the expected result |
| `cross-check RECIPE CHECK` | run a named supporting calculation or comparison |

Each command builds its required binaries for the selected problem. `profile`
and `verify-table` expand the certificate's orbit bounds in memory at startup.
Certificate search, proof verification, and monotonicity checking remain
separate commands; profile enumeration does not implicitly run them.

For example, to reproduce the ternary computation from the committed certificate:

```bash
python3 run.py verify-cert q03_n233
python3 run.py verify-table q03_n233 --histogram
python3 run.py profile q03_n233
python3 run.py cross-check q03_n233 restrictions
```

Use the same first three commands with `q02_n333` for the binary result.
The restriction calculation prints and checks the active sets used in the
paper; it does not automate the exclusion argument given by hand.

Search writes to `tmp/search/cert_matrix_RECIPE.pb.txt` by default and never
modifies the committed certificate. Other commands use `certs/matrix/`.
`--certificate PATH` selects a different certificate; relative paths are
resolved against the caller's working directory. Replacing existing search outputs
requires `--overwrite`; reading a certificate does not.
To use newly searched data, pass its path explicitly to `verify-cert`,
`verify-table`, and `profile`. `verify-cert` also accepts a certificate path
instead of a recipe: `verify-cert certs/matrix/cert_matrix_...pb.txt`.

`--dry-run` prints the commands and prerequisite hints without launching
subprocesses or writing logs or certificates. Actual runs write
unique logs under `log/reproduction/`, including input hashes, the commit,
commands, output, exit status, and elapsed time. Named enumerations check
the documented profile sets and counts, applicable job and node totals, and
explicit exhaustion status; unexpected or incomplete results fail.
Extra enumerator arguments after `--` select a **custom run**, for example
`python3 run.py profile q02_n333 -- --list_only`. Such runs preserve the
tool output but do not claim validated paper reproduction or exhaustion.

### Direct build commands

Every pipeline binary and profile enumerator is compiled for one problem,
selected at build time by four defines scoped to the `*_main.cc` translation
units (kept comma-free individually because Bazel splits copt lists on commas):

```bash
CP2='--per_file_copt=.*_main\.cc@-DCP_P=2,-DCP_N0=3,-DCP_N1=3,-DCP_N2=3'   # <3,3,3> over F_2
CP3='--per_file_copt=.*_main\.cc@-DCP_P=3,-DCP_N0=2,-DCP_N1=3,-DCP_N2=3'   # <2,3,3> over F_3
bazel build --config=opt "$CP3" //subspace_bounds/verifier:verifier_main
```

The committed default in `matrix/chosen_problem.h` is $\langle 3,3,3\rangle$
over $\mathbb{F}_2$, so a plain `bazel build //...` works; do not edit that
file to switch problems. `run.py` derives the defines from the selected recipe
or certificate name. The generic profile enumerator supports F₂ first-factor
shapes 2×2, 2×3, and 3×2, and the F₃ shape 2×3; the specialized enumerator requires
F₂ ⟨3,3,3⟩. Unsupported selections fail at startup. Both read the certificate
specified by `--certificate=PATH`, whose recorded full problem name must match
the build. The manual test `//core:rank_lower_bound_rank_one_span_heavy_test` (the
$k=4$ and $k=5$ family-search exclusions of the $\langle 3,3,3\rangle$
certificate, each taking minutes) is excluded from `//...`.

## A subspace lower-bound table

This step produces and verifies certified lower bounds for restriction
subspaces, then expands them into a table for the capacity argument. To use
the committed certificates, skip the search and start with
[verification](#verify-the-certificates).

### Certificates

| certificate (`certs/matrix/`) | orbits | certified lower bound | upper bound |
|---|---|---|---|
| `cert_matrix_q02_n333` | 496 | 20 (21 with the paper) | 23 |
| `cert_matrix_q03_n233` | 31 | 14 (15 with the paper) | 15 |
| `cert_matrix_q02_n324` | 31 | 19 | 20 |
| `cert_matrix_q02_n233` | 31 | 15 | 15 |
| `cert_matrix_q02_n224` | 11 | 14 | 14 |
| `cert_matrix_q02_n223` | 11 | 11 | 11 |
| `cert_matrix_q02_n222` | 10 | 7 | 7 |

The first two are used in the paper; `cert_matrix_q02_n324` supports the D'Ambrosio
cross-check; the four small ones are test data for `//core` (the golden
soundness sweeps of the rank-one-span and forced-product recomputers).
Certificates are never edited by hand, and only a search run produces a new
one.

### Search: producing the certificates

```bash
python3 run.py search-cert q02_n333
```
<!--
AWS c8g.48xlarge
real    964m4s
user    31263m45s
-->
The run takes about 16 hours on a 192-core CPU (an AWS c8g.48xlarge Spot Instance
at ~1 USD/hour) and uses about 22 CPU-days.

```bash
python3 run.py search-cert q03_n233
```
<!--
AWS c8g.48xlarge
real    12m22.405s
user    174m38.345s
-->
The run takes less than an hour on a laptop and uses about 3 CPU-hours.

### Verify the certificates

Each command below builds `verifier_main` for the certificate's problem and runs it
on the certificate and its `.btp` archive.

```bash
python3 run.py verify-cert certs/matrix/cert_matrix_q02_n333.pb.txt
```
<!--
AWS c8g.8xlarge
real    454m20.270s
user    4740m24.405s
-->
Expected: `UNCONSTRAINED TENSOR RANK LOWER BOUND: 20` and `OK. Verified`.
The run takes about 8 hours on a 32-core CPU (an AWS c8g.8xlarge Spot Instance
at ~0.2 USD/hour) or 11 hours on an 8-core MacBook Air M4 laptop.
It uses about 3.3 CPU-days.

```bash
python3 run.py verify-cert certs/matrix/cert_matrix_q03_n233.pb.txt
```
<!--
real    0m2.516s
user    0m12.014s
-->
Expected: `UNCONSTRAINED TENSOR RANK LOWER BOUND: 14` and `OK. Verified`.
It takes seconds on a laptop.

### Expand and check the tables

```bash
python3 run.py verify-table q02_n333 --histogram
```
Expected: `Loaded 496 orbits`, `Enumerated 8283458 subspaces`, then
`antitone check: subspaces=8283458 pairs=213188689 violations=0`.
The histogram must match Table 1 of the paper, keyed by the number of
constraints, $9-\dim S$.

```bash
python3 run.py verify-table q03_n233 --histogram
```
Expected: `Enumerated 56632 subspaces`;
`antitone check: subspaces=56632 pairs=969696 violations=0`.

### Verifier trust boundary

To trust a certified bound, read the verifier and its trust base:

- `subspace_bounds/verifier/`: `verifier.h` (`VerifyRankLowerBound`, the sweep and the
  per-orbit dispatch), `backtracking_verifier.h` (the rank map and the trace
  replay), `verifier_main.cc`;
- `core/`: the field arithmetic, `tensor.h`, `constraints.h`, the certificate
  proto with `proto_io`, the backtracking-proof codec, the recomputers
  `rank_lower_bound_flatten.h`, `rank_lower_bound_forced_product.h` and
  `rank_lower_bound_rank_one_span.h` with its engines `rank_one_span_f2.h`,
  `rank_one_span_family_search.h` and `rank_lower_bound_rank_one_span_fp.h`,
  and `symmetry.h`, the concept the symmetry groups satisfy;
- `matrix/`: the tensor, the two symmetry groups, `chosen_problem.h`.

The search and upper-bound packages, the table tools in `subspace_bounds/`,
and the enumerators in `profiles/` are outside this boundary. The verifier
does not rely on their correctness to accept a certificate. One qualification: a
rank-one-span record is checked by re-running the search's own engine, so
that engine and the coverage proof of its family search (paper, Appendix B)
belong to the trust base.

## Capacities and profiles

The two `profile_enum_*_main.cc` files define CLI flags and select the problem.
The implementation lives in `profiles/q02_n333/` (binary algebra/capacities,
symmetry and outer cases, lattice updates, DFS, self-checks, and job orchestration)
and `profiles/fp/` (templated geometry, prepared data, search, self-checks, and
orchestration).

This step uses the [expanded tables](#expand-and-check-the-tables) to impose
capacity constraints on first-factor profiles. The binary enumeration finds
no profile with twenty terms; the ternary enumeration finds three profiles
with fourteen terms, which the paper excludes by hand.

### $\langle 3,3,3\rangle$ over $\mathbb{F}_2$

Runner: `python3 run.py profile q02_n333` (paper, Section 2.3 and Appendix C).

Expected: the self-test line, `outer cases (up to the stabilizer of B0): 35
|W|=0:1 |W|=1:3 |W|=2:5 |W|=3:6 |W|=4:8 |W|=5:5 |W|=6:4 |W|=7:2 |W|=8:1`, one
`case N ...: INFEASIBLE` line per case (16 of them `(W violates)`), and

```
summary: feasible=0 infeasible=35 over_budget=0 of 35
```

The run uses about 1,500 CPU-seconds for 19 searched cases, 50,830 jobs, and
15,658,474 nodes. It takes five to eight minutes on 8 laptop cores or about
a minute on 64 cores. 
The per-case job and node counts must match Table 3 of the paper.

Cross-checks, to be repeated after any change to the tool:

1. *Positive control.* The three factor lists of the certificate's 23-term
   decomposition satisfy every capacity at $r=23$ (each prints
   `check_list: PASS`):

   Runner: `python3 run.py cross-check q02_n333 positive`.

2. *Orderly generation.* A restricted feasible instance gives the same 232
   profiles with and without symmetry pruning (about 10 s and 200 s, respectively):

   Runner: `python3 run.py cross-check q02_n333 symmetry`. It compares the
   normalized profile sets, excluding timing, progress, and node-count lines.
   It runs single-threaded and takes a few minutes.

3. *Unpruned run on the full instance.* Adding `--symmetry=false
   --budget=1000000000000` to the enumeration command disables orderly
   generation while keeping the 35 symmetry-reduced outer cases. Every
   case is infeasible, with a total of 591,061 jobs and 2,155,604,077 nodes.
   A rerun reproduces the per-case node counts.

   Runner: `python3 run.py cross-check q02_n333 no-symmetry`.
   <!--
   AWS c8g.48xlarge
   real    74m33.221s
   user    5899m36.096s
   -->
   This run is explicitly opt-in and takes about 75 minutes on a 192-core CPU,
   using about 4 CPU-days.

4. *Split versus unsplit.* Runs using `--cases=0` with `--split_depth=0` and with
   `--split_depth=3` give the same verdict and similar node counts.

   Runner: `python3 run.py cross-check q02_n333 split`.
   It runs mostly single-threaded and takes a few minutes.

### $\langle 2,3,3\rangle$ over $\mathbb{F}_3$

Runner: `python3 run.py profile q03_n233`.

It takes less than a minute on a laptop.

Expected: `rank-1 points: L = 13`, `rank-2 points: L = 14` and

```
r=14: FEASIBLE jobs=3 nodes=7024 profiles=3 over_budget_jobs=0
```

The three profiles are formed by appending `364`, `495`, or `715`
to `1 3 9 13 27 28 55 81 84 165 243 252 351` (the point codes
$\sum_{ij}U_{ij}3^{3i+j}$ of the projective first factors, normalized to
leading coefficient 1). They are the profiles
$P_1,P_2,P_3$ in Section 4 of the paper, which excludes them by hand.

The unrestricted exhaustive cross-check (orderly generation off, all 52
rank-one candidates; the rank-two points have capacity 0) must print the same
three profiles, with 82,153 jobs and 52,972,264 nodes. It takes a few minutes
on a laptop.

Runner: `python3 run.py cross-check q03_n233 no-symmetry`.

The hand exclusion compares column restrictions with six or seven active
terms; the active sets of each profile are printed by
`python3 run.py cross-check q03_n233 restrictions`, which also checks the
expected tight and seven-term restrictions.

Output: $P_1$ is tight (six active terms) only at $z_0=(0,0,1)$ and has seven
active terms at $z_0\in\{(1,0,0),(0,1,0),(0,1,2),(1,0,2)\}$; $P_2$ is tight
at $z_0=e_1,e_2,e_3$; $P_3$ is tight only at $z_0=(0,0,1)$ and has seven
active terms at the same four $z_0$ as $P_1$.

### The D'Ambrosio cross-check

The generic enumerator reproduces the classification behind D'Ambrosio's
result on the framework's $\langle 3,2,4\rangle$ certificate (the transposed
form of his $\langle 2,3,4\rangle$ instance, certified bound 19): four
profiles up to symmetry at $r=19$.

Runner: `python3 run.py verify-table q02_n324`, then
`python3 run.py profile q02_n324`.

Expected: `r=19: FEASIBLE jobs=1 nodes=454831 profiles=4 over_budget_jobs=0`,
followed by the four profiles. The run takes less than 1 minute.

### What to trust

Beyond the [verifier trust boundary](#verifier-trust-boundary), the two
results rely on the lemmas of the paper (Sections 2 and 4 and
Appendices A and B), the table expansion in `subspace_bounds/subspace_bounds.h` (a mechanical
orbit lookup, cross-checked by verifying that the orbit sizes sum to the number of
subspaces), and the enumerator used (`profiles/q02_n333/` or `profiles/fp/`),
validated by the cross-checks above. No SAT solver or other external reasoning
tool is used.

## Release
### Release artifacts

`python3 release.py` checks certificate and archive hashes and the generated
`paper/proof_counts.tex` against `release/artifacts.json`; after regenerating a
certificate, rerun it with `--update`. Before tagging, run the commands listed
there and the cross-checks above on the final clean commit, and attach the
`log/reproduction/` logs in a manifest:

```bash
python3 release.py --manifest tmp/release-manifest.json --log log/reproduction/RUN.log
```

Profile executables end with a `result_json:` record; its `exhausted` field,
not `FEASIBLE`, establishes a complete classification (incomplete runs exit
with status 3). Runs that change the rank or restrict the search need
`--experimental=true` and report `scope="restricted"`.

### Automated checks and release validation

`.github/workflows/ci.yml` runs the tests, artifact checks, and ternary
workflows on Ubuntu and macOS; it does not build the paper. `--config=debug`
enables ASan, UBSan, and `_GLIBCXX_DEBUG`. Before a release, also run these
expensive checks on the final clean commit and keep the logs with
the manifest:

```bash
bazel test --config=opt //core:rank_lower_bound_rank_one_span_heavy_test
python3 run.py verify-cert q02_n333
python3 run.py verify-table q02_n333 --histogram
python3 run.py profile q02_n333
python3 run.py cross-check q02_n333 positive
python3 run.py cross-check q02_n333 symmetry
python3 run.py cross-check q02_n333 split
python3 run.py cross-check q02_n333 no-symmetry
python3 run.py cross-check q03_n233 no-symmetry
python3 run.py verify-table q02_n324
python3 run.py profile q02_n324
```

## Citation

```bibtex
@misc{wang2026lowerbound213times3,
      title={A Lower Bound of 21 for $3\times3$ Matrix Multiplication over $\mathbb{F}_2$},
      author={Chengu Wang},
      year={2026},
      eprint={2609.06725},
      archivePrefix={arXiv},
      primaryClass={cs.CC},
      url={https://arxiv.org/abs/2609.06725},
}
```
