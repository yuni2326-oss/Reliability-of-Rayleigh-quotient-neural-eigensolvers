# A reproducible benchmark of Rayleigh-quotient neural eigensolvers: mode identification, failure diagnostics, and success-adjusted cost

Code and data for the benchmark study. This repository is the computational artifact: the
solver package, every committed result it produced, and the regression suite that checks the
two against each other.

## What is in here

```
eigen_benchmark/
  problems/    P1 cantilever beam, P2 annular plate, P3 rotational-spring interface,
               P4 L-shaped elasticity — each defines one weak form and nothing else
  bases/       classical trial spaces (B-spline, Legendre, Chebyshev, monomial, Hermite FEM)
  neural/      Rayleigh-quotient solvers, the sequential deflation arms, storage policies,
               span diagnostics
  drivers/     one entry point per experiment; each writes a CSV and a JSON manifest
  render/      turns the committed CSVs into tables and figures
  tests/       the regression suite

docs/_generated/data/paper2/    result tables (CSV), per-seed records (JSONL), stored mode
                                shapes and arrays (NPZ), decision files and manifests (JSON)
docs/_generated/fig/paper2/     the figures, regenerated from the same CSVs
```

Every CSV has a sibling `manifest_*.json` carrying the git SHA, timestamp, library versions,
platform and working precision of the run that wrote it. Where a run stores per-seed mode
shapes (`*_shapes.npz`), its cells can be re-scored without re-solving.

## Storage-policy and verification controls

`eigen_benchmark/drivers/run_paperA_mlst.py` holds the controls added after the primary grid.
Each one had its decision quantities and predictions fixed in a dated document before its own
data were generated.

| entry point | what it runs |
|---|---|
| `p1_rf`, `p1_basis`, `p1_basis_exact` | P1 mode 3 on a frozen mode-1/2 prefix: raw storage with diagonal subtraction, storage-time orthogonalization, exact projection, and basis changes of the same stored span |
| `p1_rf_fp32`, `p1_fp32_fixed_span` | storage-time orthogonalization in float32, and three float32 realizations of one projector applied to identical inputs |
| `p4_raw`, `p4_projected_rerun` | P4 modes 1–3 under raw and projected storage for the diagonal, Gram and weighted-MGS rules |
| `p4_adf`, `p4_adf_projected` | corner-graded quadrature (0, 4 and 8 layers) for the reference modes and for both storage policies |
| `p4_policy_fields` | field-level comparison of what the two storage policies hand to mode 3 |
| `stats`, `followup_stats`, `projected_stats` | the pre-specified decisions (seed-paired bootstrap), written to `paperA_mlst_*stats.json` |

Audits added after those data were seen (`drivers/paperA_mlst_audit.py`,
`drivers/paperA_mlst_v3_checks.py`, `p4_policy_posthoc` and `projected_posthoc`) re-read or
re-evaluate committed results. Their outputs are written to separate files and do not overwrite
a pre-specified decision.

`p4_params_raw.npz` and `p4_params_projected.npz` (about 44 MB each) hold the trained P4
networks of the two storage policies: float64 parameters for the three rules, modes 1–3 and 50
seeds, keyed `<rule>_m<mode>_<parameter>`. `p4_adf`, `p4_adf_projected`, `p4_policy_fields` and
`p4_policy_posthoc` re-evaluate them; `p4_policy_fields.csv` records the SHA-256 digest of every
mode-1 and mode-2 network. `p4_raw` and `p4_projected_rerun` regenerate them (GPU, about 1.5
hours each). P4 training is deterministic: in the recorded environment the projected-storage
rerun reproduced every committed eigenvalue bit for bit.

## Running it

```bash
pip install numpy scipy mpmath matplotlib pytest    # classical results, tables, figures
pip install torch                                   # neural arms only

PYTHONPATH=. python -m pytest eigen_benchmark/tests -q
```

The suite runs against the committed data and needs no GPU. A handful of checks skip rather
than fail, each stating why.

Regenerating results, rather than checking them:

```bash
PYTHONPATH=. python -m eigen_benchmark.drivers.run_p1     # classical, seconds
PYTHONPATH=. python -m eigen_benchmark.drivers.run_p3     # mpmath root-finding, ~20 min
PYTHONPATH=. python -m eigen_benchmark.drivers.run_p1_neural   # GPU, hours
PYTHONPATH=. python -c "from eigen_benchmark.drivers import run_paperA_mlst as ml; ml.p4_raw()"   # GPU, ~1.5 h
PYTHONPATH=. python -c "from eigen_benchmark.drivers import run_paperA_mlst as ml; ml.stats(); ml.followup_stats(); ml.projected_stats()"   # decisions, seconds
PYTHONPATH=. python -m eigen_benchmark.drivers.render_paper          # tables and figures
```

`render_paper` regenerates every table and figure from the committed CSVs, and reports what it
did.

## Precision

Classical solves and references run at float64, with mpmath at dps = 50 where a root-finding
step needs it. The reduced-precision study deliberately runs the projection arms at float32
and bfloat16; each row and each manifest records the working precision it used.

## Environment the committed artifacts came from

| | |
|---|---|
| CPU | 10× Cortex-X925 + 10× Cortex-A725, aarch64 |
| GPU | NVIDIA GB10 (neural arms only) |
| Python | 3.12.3 |
| NumPy / SciPy / mpmath | 2.1.0 or 2.5.0 / 1.17.1 or 1.18.0 / 1.3.0 |
| PyTorch | 2.12.0a0 (NGC 26.04) |

Two NumPy/SciPy combinations appear because the host environment was upgraded mid-study. Each
manifest records which combination produced its artifact, and the suite runs against the
committed data under either.

Regenerating a result under the other combination is not bitwise. Rows built on
well-conditioned bases (B-spline, Legendre, Chebyshev, Hermite elements) agree to about 1e-8
relative. Rows at the fp64 conditioning limit (the monomial bases, the orthonormalized bases
and the split-domain enrichment at N = 12) can change in their condition numbers, eigenvalue
digits, Cholesky outcome and whether e_λ ≤ 0.05. Cholesky of the raw-monomial mass matrix at
N = 12, for instance, fails for P1 and succeeds for P3 under NumPy 2.5.0 / SciPy 1.18.0, and
the reverse under NumPy 2.1.0 / SciPy 1.17.1.

## License

See `LICENSE`.
