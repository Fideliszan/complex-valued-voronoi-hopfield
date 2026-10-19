# Theory-to-code audit

[Overview](OVERVIEW.md) · [Experiments](EXPERIMENTS.md) · [Validation](VALIDATION.md)

The default launchers use an independent implementation in `src/cvhnn/independent.py`,
`independent_search.py`, and `independent_benchmark.py`. Reference
kernels remain in `saturation.py`, `recall.py`, and `activation_cost.py` and can
be selected with `--engine reference`. Shared experimental inputs, trial enumeration,
and CSV schemas are deliberate: independence concerns the numerical algorithms,
not a different set of random experiments. This is a same-team implementation
audit, not an external replication.

## Traceability

| Mathematical or methodological requirement | Executable implementation | Check |
|---|---|---|
| Finite nonempty alphabet and fixed total order | `configs/ordered_alphabets.json`; array position is order | Frozen coordinates; archive hash |
| Classical real inner-product maximum | `independent.decision`, `rule='classic'` | Constant-modulus equivalence; full recomputation |
| Euclidean projection and current-state retention | `independent.decision`, `rule='voronoi'` | Boundary and origin cases |
| Static quantizer, first minimizer | `independent.quantize` | Origin and ordered boundary tests |
| Hermitian random coupling | `independent.random_weights` | Gaussian scale; real nonnegative diagonal |
| Projection storage, diagonal retained | `independent.projection_weights` | `WX=X`, Hermitian symmetry, duplicate memories |
| One coordinate at a time, full local field | `independent.dynamics` | Fresh row-dot comparison at every change |
| Correct Lyapunov function | `independent.energy`, `dynamics` | Coordinate identity and fresh quadratic form |
| Fair scheduling | One random permutation per sweep, shared by each pair | No-change complete sweep required to stop |
| HSV encoding and corruption | `independent.encode`, `corrupt_hsv`, `corrupt_symbols` | Channel semantics and paired seeds |
| SER, radial error, similarity, hue error, PSNR | `independent.retrieval_metrics` | Direct definitions and zero-error sentinel |
| Distance, score, KD-tree, structured search | `independent_search.Search` | All cardinalities, boundaries, current retention |

## Numerical decisions

All states and weights use complex128. In saturation, values within absolute
tolerance `1e-12` of the optimum form the numerical tie set. In recall dynamics,
the tolerance is `1e-12 * max(1, abs(optimum))`, matching the experimental protocol.
The current state is retained if eligible; otherwise the first eligible array
index is used. Static quantization and benchmark searches use an absolute
`1e-12` tolerance on squared distance. This prevents nominally equal-radius
symbols at zero from being selected by the last bit of a norm calculation.

These conventions approximate exact arithmetic. The theorem is stated for
exact minimizers; numerical fixed points are checked under the specified
tolerance. They are not asserted to be exact symbolic fixed points. A changing
coordinate must have strictly smaller objective than the previous coordinate
under this conservative selection. The audit records every computed energy
change and checks a fresh terminal field, with `1e-10` as the reporting tolerance
for an increase. Tests also compare small full trajectories against a reference
that recomputes the field and complete energy after every update.

The field is maintained incrementally. Recall refreshes it after each sweep;
saturation retains the original incremental convention and measures its drift
against a fresh matrix-vector product. A complete no-change sweep certifies
stopping. The limit of 80 sweeps is an experimental cap, not a theorem bound.

## Inputs and randomization

For saturation, the real and imaginary entries of the unmasked matrix are
independent standard normals divided by `sqrt(2N)`; the diagonal is then zeroed.
The Hermitian part is multiplied by beta=1, with alpha=0 on the diagonal. There
are 100 independent matrices per alphabet and initialization; the two rules
share each matrix, starting indices, and all permutations. Uniform and inner
initializations are separate experiments. The master seed is 20260831.

Recall uses six deterministic 24-by-24 synthetic HSV patterns generated with
seed 20260901; the noise study stores the first four. The complete procedural
specification is `recall.generate_synthetic_hsv_memories`: an angular hue field
with seeded rotations and sinusoidal textures, plus normalized radial, wave,
blob, and diagonal saturation components. Its role is to define common input
data. It is not an activation or optimization routine under test. The stored
patterns are the quantized images, not the prequantized continuous fields.

The pseudoinverse cutoff is `rcond=1e-10`. Symbol replacements are Bernoulli per
neuron and include the original symbol; nominal probability is therefore not
the realized SER. HSV hue and saturation receive independent zero-mean Gaussian
noise of the stated standard deviation, followed by wrapping/clipping and
quantization. Thirty trials alternate targets with `trial % 4`; they are noise
replicates for a fixed bank, not independent draws of memory sets. The load
experiment has two trials per condition and is exploratory.

## Cardinality experiment

The four values 16, 64, 256, and 1024 denote **alphabet cardinalities**, not
neuron counts or real-space dimensions. All alphabets lie in the complex plane;
the field-generating network always has N=576 neurons and P=4 memories. Its
memory bank is the same first four patterns from the six-pattern recall
bank.

For each of four families and each cardinality, use 20,000 uniform fields in
[-1.25 Rmax, 1.25 Rmax]^2 and 20,000 network fields collected initially and after
up to three Voronoi sweeps. Network queries retain their actual current symbol;
uniform queries receive a reproducibly sampled current index. Each pair of
field and current index is shared across methods and saved in `data/queries`.

All methods retain the current symbol on numerical ties. KD-tree and structured
paths detect competing nearest candidates and use exhaustive distance resolution
when ambiguous. This fallback can cost O(|S|) at boundaries; expected fast-path
cost is not a worst-case guarantee. Correctness is checked before timing and
on nearest-neighbor midpoints and zero, with and without a current state.

Five timed repetitions follow a warm-up. Method order rotates across repetitions;
one BLAS thread and one KD-tree worker are used. Preprocessing is recorded
separately. The figure reports median **batch-average** microseconds per query,
with the full observed repetition range. It does not measure single-call latency
or end-to-end recall acceleration. The analytical flop counts describe the
search core and exclude tolerance checks and boundary fallback.
