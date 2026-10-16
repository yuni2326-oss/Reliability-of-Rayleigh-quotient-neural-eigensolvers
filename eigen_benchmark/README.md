# eigen_benchmark

Benchmark code measuring **reliability and error attribution** in Rayleigh-quotient neural
eigensolvers. Nine trial spaces and nine solver arms are run against four problems under one
weak form each, so that a failure can be traced to the component it came from rather than
attributed to the method as a whole.

Public mirror of the code and data:
<https://github.com/yuni2326-oss/Reliability-of-Rayleigh-quotient-neural-eigensolvers>
(the first release is at
<https://github.com/yuni2326-oss/Deflation-Accuracy-Governs-Mode-Order-Reliability-in-Rayleigh-Quotient-Neural-Eigensolvers_ver1.0>).

## Design contract

Every basis and solver exchanges only the **discrete (K, M) coming from one weak form per
problem**. That makes `e_model` zero by construction, so the remaining components are what is
actually measured:

```
e_measurement  |  e_model · e_approx · e_algebraic · e_optimization
```

The bar is deliberate. `e_measurement` — error in the instrument rather than in the solution —
is a **precondition** for the other four, not a peer of them. If the references or the scoring
rules cannot resolve the quantity being judged at the operating point in question, no
attribution among the rest is meaningful, so it is excluded first. `run_reference_certification`
performs that check for every quantity the scoring reads.

- `problems/*` know nothing about bases. `bases/*` know nothing about problems.
- `metrics.py`, `conditioning.py`, `cost.py` are pure functions (arrays in, scalars/dicts out).
- The deterministic core needs only numpy/scipy/mpmath. The neural arms in `neural/` need
  torch and a GPU; they run in the `gb10-pinn:26.04` container.

## Running

```bash
cd /home/super/project

# tests — over 400, no GPU needed (they run against the committed data)
PYTHONPATH=/home/super/project /home/super/equip-venv/bin/python -m pytest eigen_benchmark/tests -q

# deterministic results — one driver per problem regenerates that problem's results
for d in p1 p2 p3 p4; do
  PYTHONPATH=/home/super/project /home/super/equip-venv/bin/python -m eigen_benchmark.drivers.run_$d
done      # p2 and p3 take minutes to ~20 min each on mpmath root-finding

# certify the references against a higher-precision evaluation of themselves
PYTHONPATH=/home/super/project /home/super/equip-venv/bin/python \
  -m eigen_benchmark.drivers.run_reference_certification

# neural arms — GPU, hours per driver
docker run --rm --gpus all --ipc=host -v /home/super/project:/work -w /work \
  -e PYTHONPATH=/work gb10-pinn:26.04 \
  python3 -m eigen_benchmark.drivers.run_p1_neural
```

`main(outdir=..., quick=True)` is the reduced-grid smoke path every driver exposes; the tests
use it.

## Layout

| path | responsibility |
|---|---|
| `quadrature.py` | piecewise Gauss–Legendre (basis order `p` kept separate from quadrature order `n_q`) |
| `quadrature_study.py` | quadrature separation: hold the basis order and double the quadrature order, so quadrature error cannot pass for basis error |
| `bases/base.py` | basis protocol (`eval`/`d1`/`d2`/`d1_jump`/`breaks`) and the coordinate-map wrapper |
| `bases/monomial.py` | (i) raw monomials, (ii) orthonormalized |
| `bases/recurrence.py` | (iii) shifted Legendre, (iv) integrated Legendre, (v) Chebyshev |
| `bases/bspline.py` | (vi) continuity-controlled C⁰ B-spline |
| `bases/enriched.py` | (vii) interface-enriched / split-domain |
| `bases/fem.py` | (viii) Hermite C¹ beam element with rotation-DOF doubling at `x_c` |
| `bases/radial_fem.py` | radial Hermite C¹ elements for P2 (r-dependent coefficients force per-element quadrature) |
| `reference/transfer_matrix.py` | (ix) transfer-matrix exact reference (fp64 scan refined by mpmath) |
| `reference/bessel_annulus.py` | P2 exact Bessel reference (4×4 determinant, column scaling + mpmath recurrence) |
| `problems/p1_beam.py` | P1 cantilever beam — analytic modes, (K, M) assembly |
| `problems/p2_annulus.py` | P2 annular Kirchhoff plate — one 1-D radial problem per nodal diameter `m` |
| `problems/p3_spring.py` | P3 zero-width rotational spring, broken-H² single weak form (k̂ ∈ {1, 10, 100, 1000}) |
| `problems/p4_lshape.py` | P4 L-shaped plane elasticity — three-block graded Q2 mesh + Richardson |
| `degeneracy.py` | degenerate eigenspace construction in (r, θ) and the subspace metrics |
| `conditioning.py` | four condition-number normalizations, backward error, Cholesky, fp64 vs mpmath |
| `metrics.py` | `e_λ`, MAC, `r_h`, orthogonality, principal angles, **the pre-specified mode classification**, Wilson |
| `cost.py` | `t(ε)` line items and `E[T_success]` |
| `neural/core.py` | vmap seed-ensemble Rayleigh-quotient engine |
| `neural/deflation.py` | the deflation rules — **the attribution instrument of this study** |
| `neural/sequential.py` | sequential solver: modes 1..n in order, scored by the pre-specified rule |
| `neural/diagnostics.py` | diagnostics of the stored span, computed on what deflation is actually handed |
| `neural/ladder.py` | initialization ladder I0–I5 |
| `neural/subspace.py`, `net.py`, `curriculum.py`, `pielm.py`, `spacetime.py`, `p3_neural.py`, `p4_neural.py` | arms (c)–(g) and the 2-D solver |
| `render/` | tables and figures built from the committed CSVs |
| `drivers/` | one entry point per experiment, each writing a CSV and a manifest |

Results are written to `docs/_generated/data/paper2/`. Every CSV has a sibling
`manifest_*.json` carrying the git SHA, branch, library versions, platform and working
precision of the run that wrote it. Runs that store per-seed mode shapes (`*_shapes.npz`) can
be **re-scored without re-solving** — which was needed once, when a reference defect was found
after the solves had already been committed.

## Deflation arms

| arm | rule |
|---|---|
| (a) | penalty deflation |
| (a′) | diagonal-approximation projection (pilot) |
| (b) | exact M-orthogonal projection via the Gram matrix `G = ΦᵀWΦ` |
| (b′) | the same projector via W-orthonormalization, without forming `G` |
| (c) | coarse-Ritz curriculum |
| (d) | separate-net neural basis + Galerkin |
| (e) | shared-trunk subspace trace |
| (f) | Eig-PIELM (random features, no backpropagation) |
| (g) | space–time PINN (a negative result) |

(b) and (b′) are the **same projector** — proved, and checked numerically on each run's own
data. They differ only in what their coefficient accuracy is exposed to: `κ₂(G)` for one and
`κ₂(Φ) = √κ₂(G)` for the other. That difference is the whole point of the precision axis.

## Pre-specified values, frozen before any run

`metrics.MAC_MIN = 0.9` and `metrics.ELAM_MAX = 0.05` define the four-way mode classification
(`correct` / `lower_mode_basin` / `spurious` / `non_converged`), scored over 50 seeds at 4000
iterations with 95 % Wilson intervals. **They are not changed after seeing data.** The
constants are pinned by tests, so changing one fails the suite.

Arm (b′), the reduced-precision axis, the mode 11–16 extension and the two added P4
projections came after the pre-specified grid. They reuse the scoring unchanged, and nothing
added after the fact overrides a pre-specified outcome.

The storage-policy controls that followed (`drivers/run_paperA_mlst.py`: storage-time
orthogonalization on a frozen prefix, basis changes of a fixed span, an identical-input
precision test, raw- and projected-storage P4 runs and a corner-graded quadrature audit) each
had its decision quantities and predictions fixed in a dated document before its own data
existed. Audits added after those data were seen (`drivers/paperA_mlst_audit.py`,
`drivers/paperA_mlst_v3_checks.py` and the `*_posthoc` outputs) re-read or re-evaluate
committed results and do not overwrite any pre-specified decision.

**Two senses of the seed count.** A result reports how many seeds a cell was *scored* over and,
separately, how many its per-stage diagnostics *summarize*. The rule: diagnostics logged inside
a scored run summarize every seed of that cell; dedicated diagnostic runs state their own
count. This matters because `κ₂(G)` has a heavy seed-to-seed tail at depth, so summarizing a
subset shifts it — by up to a factor of three at k = 10 on P1 — while leaving every scored
outcome untouched, which makes the discrepancy invisible unless two tables are compared.

## Numerical facts established while implementing

Trusting the computation means knowing these are not accidents. All are pinned by regression
tests.

| fact | evidence |
|---|---|
| `1 + cos β cosh β` cancels catastrophically near β₁₀ ≈ 29.8 → **use `cos β + sech β`** | agrees with mpmath dps = 50 roots to 1e-14 relative |
| The analytic mode shape must be evaluated in **exponential form**, not as `cosh βx − σ sinh βx` | the textbook form loses every fp64 digit at β ≳ 40 and capped MAC at 0.887 from mode 14 up, classifying correct solutions as spurious |
| **References need the same certification as solvers** | the defect above was invisible to arm-to-arm comparison because both arms failed identically; it is an `e_measurement` failure, and `run_reference_certification` now covers every quantity the scoring reads |
| C¹ global bases make ⟦u′⟧ ≡ 0 on P3, cancelling the `k_θ` term → refinement converges to the **uniform beam**, not the spring beam | at ā = 0.3 the error against the spring reference is flat at 6.3e-2 from DOF 12 to 24 |
| The saturation gap of the C¹ bases scales exactly as **1/k̂** | 2.108 / 0.2106 / 0.02105 / 0.002104 at k̂ = 1/10/100/1000 (ratio 10.00) — the `e_approx` floor equals the first-order effect of interface compliance |
| Ramp enrichment has zero bending energy in the broken form → **on P1 it is a hinge mechanism** and K is singular | enriched bases are P3-only |
| Raw monomials fail fp64 Cholesky beyond N ≈ 16 (Hilbert-like mass matrix) | recorded as `cholesky_ok=False` + NaN rather than raised as an exception |
| Orthonormalization rescues **solvability but not accuracy** | the parent-basis evaluation has already lost rank in fp64 |
| Integrated Legendre bubbles must start at **L₂** | L₀ and L₁ vanish identically once the cubic Hermite correction is removed, seeding a null function and a spurious λ ≈ 0 |
| Misaligning quadrature with basis break points under-integrates stiffness → **the Ritz upper-bound property breaks** | `Basis.breaks` aligns automatically; misalignment is kept only as a control |
| Fewer Gauss points than basis DOF makes M rank-deficient by construction | quadrature-order floor = `n_dof` |
| High precision **cannot recover what assembly lost** | the fp64 Hilbert(14) is already not positive definite in exact arithmetic, so mpmath Cholesky refuses it and it is reported through the general pencil path |
| An fp64 function cannot be refined by mpmath | the residual stalls at 1e-50 — Bessel needs its own high-precision recurrence path |
| Q2 mid-side nodes placed at the **grading-parameter midpoint** degenerate the map | for β ≥ 2 they leave the middle half of the edge and the Jacobian goes non-positive; grade the corners only |
| **Over-grading hurts** | on P4, β = 5 is worse than β = 3 (6/6 → 1/6 targets met). Accuracy is governed by refinement level, not grading |
| Mirror symmetry (Z₂) does not force degeneracy | it has only 1-D irreps; double roots come from non-abelian groups like C₄ᵥ. The correct consequence is "each mode is symmetric or antisymmetric" |
| The Bessel 4×4 **requires column scaling** | I_m and K_m differ by 10¹¹ at kb ≈ 26, so the determinant is meaningless to cancellation without it |
| `κ₂(G)·u` is a usable stopping condition for the Gram route, and **only** for it | at float32 the product first exceeds one at the span handed to the mode-6 solve, which is exactly where arm (b) breaks; arm (b′), which never forms G, holds two modes deeper |
| Below float32 the **scoring rules leave their validity**, not the arms | at bfloat16 the convergence gate asks for 1e-3 against a unit roundoff of 7.8e-3, so mode 1 loses seeds with no deflation applied at all while every seed found the right mode |
| PyTorch has no `lu`/`cholesky`/`eigh` kernel at float16 or bfloat16 on either device | that is a statement about a library, not the method: an unblocked Cholesky in elementary tensor operations runs the arm and reproduces `torch.linalg.solve` to rtol 1e-10 in fp64 |

## Boundaries

This package **never imports `impeller_pinn`**; the rule is enforced as a regression in
`tests/test_paper_boundary.py`. P3 uses the rotational spring as a **generic interface
benchmark** and does not parameterize it by a damage mechanism such as crack depth.

**Non-dimensional rule.** No absolute dimensions, material properties or frequencies in Hz.
Every conclusion is a non-dimensional quantity (Λ, `e_λ`, MAC, principal angle, condition
number), so nothing is lost, and absolute dimensions would tie the test problems to one
specific machine. `sandwich_props`, `lambda_to_hz`, `omega_scale` and `annulus_frequencies`
were removed for that reason. P2 is fixed by a/b and ν alone (b = 1), and P4 sets L = E = ρ = 1
so Ω² = ω²ρL²/E comes out directly.
