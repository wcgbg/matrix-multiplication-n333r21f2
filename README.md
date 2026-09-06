# A Lower Bound of 21 for 3 by 3 Matrix Multiplication over F2

Companion repository of the paper *A Lower Bound of 21 for $3\times3$ Matrix
Multiplication over $\mathbb{F}_2$* (`paper/main.tex`). It proves two results
about the tensor rank $\mathbf{R}$ of the matrix-multiplication tensor
$\langle n_0,n_1,n_2\rangle$ (an $n_0\times n_1$ matrix times an
$n_1\times n_2$ matrix) over a small field:

$$\mathbf{R}_{\mathbb{F}_2}(\langle 3,3,3\rangle)\ge 21,\qquad
  \mathbf{R}_{\mathbb{F}_3}(\langle 2,3,3\rangle)= 15 .$$

- **3 by 3 matrix multiplication over $\mathbb{F}_2$ needs at least 21
  multiplications.** The previous lower bound was 20, from Wang's automated
  framework ([arXiv:2603.07280](https://arxiv.org/abs/2603.07280)), and 19
  by Bläser before that; Laderman's 23-product algorithm is the best upper
  bound.
- **2 by 3 times 3 by 3 matrix multiplication over $\mathbb{F}_3$ has rank
  exactly 15.** The previous lower bound was Bläser's 14; 15 is Hopcroft and
  Kerr's algorithm, valid over every field.

Both proofs are computer assisted: a machine-checked subspace lower-bound
table, an exhaustive enumeration of first-factor profiles against
the capacities the table implies, and, over $\mathbb{F}_3$, a short hand
argument excluding the three surviving profiles. Everything needed to rebuild
and re-check them is in this repository.

## The method

Let $T$ be the multiplication tensor and $A$ the space of its first argument
(the $n_0\times n_1$ matrices). For a subspace $S\subseteq A$, $T_S$ is $T$
with its first argument restricted to $S$.

1. **A subspace lower-bound table.** The framework of the cited paper enumerates
   the subspaces $S\subseteq A$ up to the symmetry group of the tensor and
   assigns to every orbit a lower bound $L(S)\le\mathbf{R}(T_S)$ by a dynamic
   program over the orbits, from the smallest $S$ up. Its techniques are
   flattening, degenerate reduction (a lookup of an already-processed larger
   constraint set), forced products, substitution with backtracking, and, added
   for this paper, the rank-one-span search: does the slice space of $T_S$ lie
   in a subspace of dimension at most $r$ spanned by rank-one matrices? Every
   entry carries a proof record and a separate verifier re-checks each record.
   The unconstrained entry $L(A)$ is the certified bound on $\mathbf{R}(T)$:
   20 for $\langle 3,3,3\rangle$ over $\mathbb{F}_2$, 14 for
   $\langle 2,3,3\rangle$ over $\mathbb{F}_3$. The table with its proofs is
   the **certificate** (`certs/matrix/`). The tools in `subspace_bounds/`
   produce and verify it, then expand its orbit bounds to every subspace.
2. **Capacities and profiles.** In a decomposition with $r$ terms, restricting
   the first argument to $S$ kills exactly the terms whose first factor
   vanishes on $S$, so at most $r-L(S)$ terms can do so: the *capacity* of
   $S$. The tools in `profiles/` consume the expanded table and enumerate
   exhaustively the *profiles*, the lists of first factors compatible with
   all capacities at once (the strategy of D'Ambrosio's
   proof that the $\mathbb{F}_2$ rank of $\langle 2,3,4\rangle$ is 20). For
   $\langle 3,3,3\rangle$ over $\mathbb{F}_2$ no profile with 20 terms exists,
   so $\mathbf{R}\ge 21$. For $\langle 2,3,3\rangle$ over $\mathbb{F}_3$ three
   profiles with 14 terms survive, and the paper (Section 4) excludes each by
   comparing overlapping restrictions, so $\mathbf{R}\ge 15$.

**Convention.** The paper indexes everything by the restriction space $S$ and
sweeps $\dim S$ from 0 up to $\dim A$. The code, its flags (`--dim_min`,
`--dim_max`), the certificate field `dim`, the expanded tables and the notes
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
| `subspace_bounds/` | `subspace_bounds_main` expands a certificate to all subspaces; `check_table_main` checks monotonicity; `tables/` holds the regenerated, git-ignored `.bin` tables | table tools are outside the certificate, see [What to trust](#what-to-trust) |
| `subspace_bounds/verifier/` | the certificate checker: `verifier.h`, `backtracking_verifier.h`, `verifier_main.cc` | yes |
| `subspace_bounds/search/` | orbit enumeration, the meet-in-the-middle orbit map, the lower-bound dynamic program (`rank_lower_bound_computer.h`) and the proof generators | no: produces certificates |
| `subspace_bounds/upper_bound/` | the flip-graph search for decompositions ($\mathbb{F}_2$ only), which records rank upper bounds in the certificate | no |
| `profiles/` | the capacity-constrained enumerators `profile_enum_q02_n333_main` and `profile_enum_fp_main` | outside the certificate, see [What to trust](#what-to-trust) |
| `certs/matrix/` | the certificates `cert_matrix_q<p>_n<n0><n1><n2>.pb.txt` and their backtracking archives `.btp` (Git LFS) | data |
| `paper/` | `main.tex`, `refs.bib` | |
| `run.py` | named recipes for certificate search/verification, table expansion/checking, profiles, and supporting cross-checks | orchestration; not an additional proof verifier |

## Build

Prerequisites: [Bazel](https://bazel.build/) 7 or later (Bazelisk works), a
C++20 toolchain, Python 3.9 or later, and [Git LFS](https://git-lfs.com/) installed
before cloning (the certificates and their archives are LFS objects; the
$\langle 3,3,3\rangle$ archive is 267 MB). The external libraries (protobuf,
gflags, ng-log, Boost, GoogleTest, oneTBB, mimalloc) are fetched by Bazel.

```bash
bazel build --config=opt //...     # release: -O3 -march=native -flto
bazel test --config=opt //...      # unit tests
bazel build --config=debug //...   # asan + ubsan + _GLIBCXX_DEBUG
python3 -m unittest -v run_test   # runner and result-validation tests
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
| `build-table RECIPE` | expand an existing certificate into a `.bin` table |
| `verify-table RECIPE --histogram` | check an existing table's monotonicity and optionally print bounds by constraint dimension |
| `profile RECIPE --threads 8` | enumerate profiles using the documented settings and validate the expected result |
| `cross-check RECIPE CHECK` | run a named supporting calculation or comparison |

Each command builds its required binaries. It does **not** implicitly run
another computational step: `profile` needs an existing table, and
`verify-table` never rebuilds one. Monotonicity checking does not establish
that a table came from a verified certificate. Certificate verification,
expansion, and table checking remain distinct obligations.

For example, to reproduce the ternary computation from the committed certificate:

```bash
python3 run.py verify-cert q03_n233
python3 run.py build-table q03_n233
python3 run.py verify-table q03_n233 --histogram
python3 run.py profile q03_n233 --threads 8
python3 run.py cross-check q03_n233 restrictions
```

Use the same first four commands with `q02_n333` for the binary result.
The restriction calculation prints and checks the active sets used in the
paper, not an automated proof of the hand exclusion.

Search defaults to `tmp/search/cert_matrix_RECIPE.pb.txt` and never modifies
the committed certificate by default. Other commands use `certs/matrix/`
and `subspace_bounds/tables/`. `--certificate PATH` selects a different
certificate for search, verification, expansion, or the positive control;
`--table PATH` selects the expanded input/output. Relative overrides are
relative to the caller's working directory. Existing outputs require
`--overwrite`; this is not needed to verify or enumerate from them.
To use newly searched data, pass its path explicitly to both `verify-cert` and
`build-table`. `verify-cert` also accepts a certificate path instead of a recipe:
`verify-cert certs/matrix/cert_matrix_...pb.txt`.
The commands formerly called `search` and `verify` are now `search-cert` and
`verify-cert`. The old `search-all` and `search-and-verify-all`
commands have been replaced by explicit recipe selection.

`--dry-run` prints the commands and prerequisite hints without launching
subprocesses or writing logs, tables, or certificates. Actual runs write
unique logs under `log/reproduction/`, including input hashes, the commit,
commands, output, exit status, and elapsed time. Named enumerations check
the documented profile sets/counts, applicable job/node totals, and
conservative exhaustion bounds; unexpected or incomplete results fail.
Extra enumerator arguments after `--` select a **custom run**, for example
`python3 run.py profile q02_n333 -- --list_only`. Such runs preserve the
tool output but do not claim validated paper reproduction or exhaustion.

### Direct build commands

Every pipeline binary is compiled for one problem, selected at build time by
four defines scoped to the `*_main.cc` translation units (kept comma-free
individually because Bazel splits copt lists on commas):

```bash
CP2='--per_file_copt=.*_main\.cc@-DCP_P=2,-DCP_N0=3,-DCP_N1=3,-DCP_N2=3'   # <3,3,3> over F_2
CP3='--per_file_copt=.*_main\.cc@-DCP_P=3,-DCP_N0=2,-DCP_N1=3,-DCP_N2=3'   # <2,3,3> over F_3
bazel build --config=opt "$CP3" //subspace_bounds/verifier:verifier_main
```

The committed default in `matrix/chosen_problem.h` is $\langle 3,3,3\rangle$
over $\mathbb{F}_2$, so a plain `bazel build //...` works; do not edit that
file to switch problems. `run.py` derives the defines from a certificate's
name. The manual test `//core:rank_lower_bound_rank_one_span_heavy_test` (the
$k=4$ and $k=5$ family-search exclusions of the $\langle 3,3,3\rangle$
certificate, minutes each) is excluded from `//...`.

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

The first two are the paper's; `cert_matrix_q02_n324` serves the D'Ambrosio
cross-check; the four small ones are test data for `//core` (the golden
soundness sweeps of the rank-one-span and forced-product recomputers).
Certificates are never edited by hand, and only a search run produces a new
one.

### Search: producing the certificates

Four stages, each a binary compiled for one problem and each reading and
writing one certificate:

1. `orbit_enumerator_main` writes one canonical representative per orbit of
   constraint subspaces.
2. `rank_upper_bound_main` ($\mathbb{F}_2$ only) runs the flip-graph search
   for decompositions and records the rank upper bounds.
3. `rank_lower_bound_main` runs the dynamic program: it sweeps the constraint
   counts from $\dim A$ down to 0, at each level computes every orbit's bound
   in parallel from the already-processed larger constraint sets, then writes
   the improved bounds and proofs back and checkpoints the certificate with
   its `.btp` archive. Techniques per orbit, in order: flattening, degenerate
   reduction, forced product, rank-one-span exclusion (climbing one rank at a
   time), backtracking. Flags: `--backtracking_step_limit` (0 disables),
   `--backtracking_max_map_size`, `--rank1span_max_subspaces` (the operation
   budget of the rank-one-span search; 0 disables),
   `--forced_product_max_iterations_log2`, `--dim_min`/`--dim_max` (the
   constraint counts to process), `--recompute_orbits=i,j` (clear and
   recompute only those orbits, with the rest seeding the map),
   `--regenerate_backtracking_proofs`, `--v=1` (the per-technique log).
4. `verifier_main`.

`python3 run.py search-cert q02_n333 --dry-run` prints the binary search recipe;
omit `--dry-run` to execute it. `search-cert q03_n233` selects the ternary recipe.
These named recipes preserve the former search settings and finish by
regenerating backtracking traces. Verify their outputs separately. Search
need not reproduce identical proof records or upper-bound decompositions.
The large runs need a many-core machine; the recorded ones used 192 cores.

### Verify the certificates

```bash
python3 run.py verify-cert certs/matrix/cert_matrix_q02_n333.pb.txt   # OK. Verified; bound 20
python3 run.py verify-cert certs/matrix/cert_matrix_q03_n233.pb.txt   # OK. Verified; bound 14
```

Each command builds `verifier_main` for the certificate's problem and runs it
on the certificate and its `.btp` archive. The last lines are
`UNCONSTRAINED TENSOR RANK LOWER BOUND: 20` (resp. `14`) and `OK. Verified`;
any failed check aborts with a `CHECK` failure. The $\langle 2,3,3\rangle$
verification takes seconds. The $\langle 3,3,3\rangle$ verification re-runs
the fifteen rank-one-span exclusions, seven of them family searches at $k=5$:
23 minutes on 192 cores (AWS c8g.48xlarge, less than 1 USD), hours on a laptop.

### Expand and check the tables

The expansion assigns each subspace the certified lower bound of its orbit.
The commands below also build the enumerators used in the second step.

A record is `uint8 dim, uint16 rows[dim], uint8 L`, the `dim` rows
spanning $S^\perp$. The check confirms that the table is monotone,
$L(S')\le L(S)$ for $S'\subseteq S$ (adding a constraint never raises $L$,
hence "antitone"), which the completeness proof of the enumeration uses
(paper, Section 6).

#### Binary table

Runner: `python3 run.py build-table q02_n333`, followed by
`python3 run.py verify-table q02_n333 --histogram`. Equivalent direct commands:

`subspace_bounds_main` and `check_table_main` are compiled for the problem;
the enumerator `profile_enum_q02_n333_main` is specific to this instance and
needs no defines.

```bash
bazel build --config=opt "$CP2" //subspace_bounds:subspace_bounds_main //subspace_bounds:check_table_main \
    //profiles:profile_enum_q02_n333_main
mkdir -p subspace_bounds/tables
bazel-bin/subspace_bounds/subspace_bounds_main $PWD/certs/matrix/cert_matrix_q02_n333.pb.txt \
    $PWD/subspace_bounds/tables/subspace_bounds_q02_n333.bin
bazel-bin/subspace_bounds/check_table_main $PWD/subspace_bounds/tables/subspace_bounds_q02_n333.bin
```

Expected: `Loaded 496 orbits`, `Enumerated 8283458 subspaces`,
`Wrote 8283458 records` (about 12 s), then
`antitone check: subspaces=8283458 pairs=213188689 violations=0` (about
15 s).

The distribution of $L$ per dimension must be Table 1 of the paper; this
prints it, keyed by the number of constraints, $9-\dim S$:

```bash
python3 - <<'EOF'
import struct
from collections import Counter, defaultdict
d=open('subspace_bounds/tables/subspace_bounds_q02_n333.bin','rb').read(); i=0; h=defaultdict(Counter)
while i<len(d):
    k=d[i]; i+=1+2*k; h[k][d[i]]+=1; i+=1
for k in sorted(h): print(k, sorted(h[k].items()))
EOF
```

#### Ternary table

Runner: `python3 run.py build-table q03_n233`, followed by
`python3 run.py verify-table q03_n233`. Equivalent direct commands:

```bash
bazel build --config=opt "$CP3" //subspace_bounds:subspace_bounds_main //subspace_bounds:check_table_main \
    //profiles:profile_enum_fp_main
C=$PWD/certs/matrix/cert_matrix_q03_n233.pb.txt
bazel-bin/subspace_bounds/subspace_bounds_main $C $PWD/subspace_bounds/tables/subspace_bounds_q03_n233.bin
bazel-bin/subspace_bounds/check_table_main $PWD/subspace_bounds/tables/subspace_bounds_q03_n233.bin
```

Expected: `Wrote 56632 records`;
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
that engine and the coverage proof of its family search (paper, Section 7)
belong to the trust base. Its operation counts are pinned by
`FamilySearchGoldenTest` in `core/rank_lower_bound_rank_one_span_test.cc`;
a change to them requires re-verifying the certificates.

## Capacities and profiles

This step uses the [expanded tables](#expand-and-check-the-tables) to impose
capacity constraints on first-factor profiles. The binary enumeration finds
no profile with twenty terms; the ternary enumeration finds three profiles
with fourteen terms, which the paper excludes by hand.

### 3 by 3 over F2

Runner: `python3 run.py profile q02_n333 --threads 8`.
The equivalent direct enumeration (paper, Sections 2.3 and 8):

```bash
bazel-bin/profiles/profile_enum_q02_n333_main \
    --bin=$PWD/subspace_bounds/tables/subspace_bounds_q02_n333.bin \
    --prop3=false --min_w=0 --threads=$(nproc 2>/dev/null || sysctl -n hw.ncpu) --split_depth=4 \
    --budget=100000000
```

Expected: the self-test line, `outer cases (up to the stabiliser of B0): 35
|W|=0:1 |W|=1:3 |W|=2:5 |W|=3:6 |W|=4:8 |W|=5:5 |W|=6:4 |W|=7:2 |W|=8:1`, one
`case N ...: INFEASIBLE` line per case (16 of them `(W violates)`), and

```
summary: feasible=0 infeasible=35 over_budget=0 of 35
```

About 1,800 CPU-seconds: 19 searched cases, 50,830 jobs, 15,658,474 nodes;
five to eight minutes on 8 laptop cores, about a minute on 64 cores (each case
line is printed on stdout and on stderr). The run is deterministic apart from
log interleaving; the per-case job and node counts must match Table 3 of the
paper.
Completeness of the run: `over_budget` must be 0, the per-job node budget
(`--budget`, 1e9 by default) is never approached (the largest job total is
3.2M nodes, case 7), and since no profile is found the `--max_solutions`
limit cannot have stopped a job. In general a `FEASIBLE` verdict with
`over_budget=0` is exhaustive only if its profile count stays below
`--max_solutions`.

Flags: `--list_only` prints the outer cases; `--cases=3,7` restricts to some;
`--symmetry=false` disables the orderly generation under the stabiliser of
$W$; `--r=N` changes the number of terms; `--check_list=...` checks one
explicit factor list; `--ones_subset=...` restricts the rank-one candidates
(testing only). The default flags add three coupled-capacity inequalities that
cut the outer cases to 13 (`--split_depth=3` suffices; expected
`summary: feasible=0 infeasible=13 over_budget=0 of 13`); the paper does not
use them.

Cross-checks, to be repeated after any change to the tool:

1. *Positive control.* The three factor lists of the certificate's 23-term
   decomposition satisfy every capacity at $r=23$ (each prints
   `check_list: PASS`):

   Runner: `python3 run.py cross-check q02_n333 positive`.

   ```bash
   python3 - <<'EOF' > /tmp/rank23_lists.txt
   import re
   t=open('certs/matrix/cert_matrix_q02_n333.pb.txt').read(); b=t[t.index('index: 495'):]
   s=re.search(r'rank_upper_bound_proof: "(.*?)"\n', b, re.S).group(1)
   terms=re.findall(r'\(([^)]*)\)\*\(([^)]*)\)\*\(([^)]*)\)', s); assert len(terms)==23
   m=lambda e,l: sum(1<<int(x[1:]) for x in e.split('+') if x.startswith(l))
   for k,l in enumerate('abc'): print(','.join(str(m(tm[k],l)) for tm in terms))
   EOF
   for L in $(cat /tmp/rank23_lists.txt); do
     bazel-bin/profiles/profile_enum_q02_n333_main --bin=$PWD/subspace_bounds/tables/subspace_bounds_q02_n333.bin \
         --r=23 --check_list=$L 2>&1 | grep check_list
   done
   ```

2. *Orderly generation.* A restricted feasible instance gives the same 232
   profiles with and without symmetry pruning (about 10 s and 200 s):

   Runner: `python3 run.py cross-check q02_n333 symmetry`. It compares the
   normalized profile sets, excluding timing, progress, and node-count lines.

   ```bash
   SUB=1,2,3,4,5,6,7,8,16,24,32,40,48,56,64,73,128,146,192,219,256,292,320,365,384,438,448,511
   for S in true false; do
     bazel-bin/profiles/profile_enum_q02_n333_main --bin=$PWD/subspace_bounds/tables/subspace_bounds_q02_n333.bin \
         --r=22 --min_w=8 --max_w=8 --cases=0 --ones_subset=$SUB --max_solutions=1000000 --symmetry=$S \
         | grep -E '^case|profile:' | sort > /tmp/profiles_$S.txt
   done
   diff /tmp/profiles_true.txt /tmp/profiles_false.txt && echo IDENTICAL
   ```

   (The `case` lines differ only in node counts and cpu time; compare the
   `profile:` lines if the diff shows just those.)

3. *Unpruned run on the full instance.* Adding `--symmetry=false
   --budget=1000000000000` to the enumeration command turns the orderly
   generation off while keeping the 35 symmetry-reduced outer cases: every
   case infeasible, 591,061 jobs, 2,155,604,077 nodes, 76 minutes on 192 cores
   on 2026-09-04; a rerun reproduces the per-case node counts.

   Runner: `python3 run.py cross-check q02_n333 no-symmetry --threads 192`.
   This is explicitly opt-in and can take hours on a laptop.

4. *Split versus unsplit.* `--cases=0` with `--split_depth=0` and with
   `--split_depth=3` give the same verdict and similar node counts.

   Runner: `python3 run.py cross-check q02_n333 split`.

### 2 by 3 times 3 by 3 over F3

Runner: `python3 run.py profile q03_n233 --threads 8`.
Equivalent direct command:

```bash
bazel-bin/profiles/profile_enum_fp_main --p=3 --bin=$PWD/subspace_bounds/tables/subspace_bounds_q03_n233.bin \
    --r=14 --max_solutions=1000 --threads=$(nproc 2>/dev/null || sysctl -n hw.ncpu) --split_depth=2
```

Expected: `rank-1 points: L = 13`, `rank-2 points: L = 14` and

```
r=14: FEASIBLE jobs=3 nodes=7024 profiles=3 over_budget_jobs=0
```

with the three profiles `1 3 9 13 27 28 55 81 84 165 243 252 351` plus `364`,
`495` or `715` (the point codes $\sum_{ij}U_{ij}3^{3i+j}$ of the projective
first factors, normalized to leading coefficient 1). They are the profiles
$P_1,P_2,P_3$ of the paper's Section 4, which excludes them by hand. Neither
stopping limit was reached: 7,024 nodes against a per-job budget of 1e9, and
three profiles against `--max_solutions=1000`.

The unrestricted exhaustive cross-check (orderly generation off, all 52
rank-one candidates; the rank-two points have capacity 0) must print the same
three profiles: 82,153 jobs, 52,972,264 nodes, 252 s on 8 laptop cores:

Runner: `python3 run.py cross-check q03_n233 no-symmetry --threads 8`.

```bash
bazel-bin/profiles/profile_enum_fp_main --p=3 --bin=$PWD/subspace_bounds/tables/subspace_bounds_q03_n233.bin \
    --r=14 --max_solutions=1000 --threads=$(nproc 2>/dev/null || sysctl -n hw.ncpu) --split_depth=4 --symmetry=false
```

The hand exclusion compares column restrictions with six or seven active
terms; the active sets of each profile are printed by

`python3 run.py cross-check q03_n233 restrictions`, which also checks the
expected tight and seven-term restrictions. The underlying calculation is:

```python
e1,e2,f,g=(1,0),(0,1),(1,1),(1,2); c1,c2,c3,c4=(1,0,0),(0,1,0),(0,0,1),(1,1,1)
common=[(x,c) for x in (e1,e2) for c in (c1,c2,c3,c4)]+[(f,c1),(f,c2),(f,c3),(g,c1),(g,c2)]
P={'P1':common+[(f,c4)],'P2':common+[(g,c3)],'P3':common+[(g,c4)]}
dot=lambda a,b: sum(p*q for p,q in zip(a,b))%3
zs=[(1,0,0),(0,1,0),(0,0,1),(0,1,2),(1,0,2),(1,2,0),(1,1,1),(1,1,0),(1,0,1),(0,1,1),(1,1,2),(1,2,1),(2,1,1)]
for n,prof in P.items():
    for z0 in zs:
        act=[t for t in prof if dot(t[1],z0)]; k=len(act)-6
        if k<=1: print(n,'z0=',z0,'active',len(act),'excess',k,act)
```

Output: $P_1$ is tight (six active terms) only at $z_0=(0,0,1)$ and has seven
active terms at $z_0\in\{(1,0,0),(0,1,0),(0,1,2),(1,0,2)\}$; $P_2$ is tight
at $z_0=e_1,e_2,e_3$; $P_3$ is tight only at $z_0=(0,0,1)$ and has seven
active terms at the same four $z_0$ as $P_1$.

### The D'Ambrosio cross-check

The generic enumerator reproduces the classification behind D'Ambrosio's
result on the framework's $\langle 3,2,4\rangle$ certificate (the transposed
form of his $\langle 2,3,4\rangle$ instance, certified bound 19): four
profiles up to symmetry at $r=19$.

Runner: `python3 run.py build-table q02_n324`,
`python3 run.py verify-table q02_n324`, then
`python3 run.py profile q02_n324 --threads 8`. Equivalent direct commands:

```bash
CP324='--per_file_copt=.*_main\.cc@-DCP_P=2,-DCP_N0=3,-DCP_N1=2,-DCP_N2=4'
bazel build --config=opt "$CP324" //subspace_bounds:subspace_bounds_main
bazel-bin/subspace_bounds/subspace_bounds_main $PWD/certs/matrix/cert_matrix_q02_n324.pb.txt \
    $PWD/subspace_bounds/tables/subspace_bounds_q02_n324.bin
bazel-bin/profiles/profile_enum_fp_main --p=2 --n0=3 --n1=2 \
    --bin=$PWD/subspace_bounds/tables/subspace_bounds_q02_n324.bin --r=19 --max_solutions=1000 --threads=8
```

Expected: `r=19: FEASIBLE jobs=1 nodes=454831 profiles=4 over_budget_jobs=0`
(about 10 s) followed by the four profiles.

### What to trust

Beyond the [verifier trust boundary](#verifier-trust-boundary), the two
results rely on the lemmas of the paper (Sections 2, 4,
6 and 7), the table expansion `subspace_bounds/subspace_bounds_main.cc` (a mechanical
orbit lookup, cross-checked by the orbit sizes summing to the number of
subspaces) and the enumerator used (`profiles/profile_enum_q02_n333_main.cc`,
about 1,000 lines, or `profiles/profile_enum_fp_main.cc`), validated by the
cross-checks above. No SAT solver or other external reasoning tool is used.

### Citation

```bibtex
@misc{wang2026lowerbound21,
  title  = {A Lower Bound of 21 for $3\times3$ Matrix Multiplication over $\mathbb{F}_2$},
  author = {Chengu Wang},
  year   = {2026},
  note   = {Companion repository: this repository},
}
```
