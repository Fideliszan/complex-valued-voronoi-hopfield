# Methods, protocols, and metrics

[Home](../README.md) · [Usage](USAGE.md) · [Validation](VALIDATION.md)

1. [Model](#model) · 2. [Scope](#scope) · 3. [Experiments](#experiments) · 4. [Numerical conventions](#numerical-conventions) · 5. [Inputs and randomization](#inputs-and-randomization) · 6. [Cardinality experiment](#cardinality-experiment) · 7. [Metrics](#metrics) · 8. [Theory-to-code traceability](#theory-to-code-traceability)

## Model

For a finite complex alphabet S, the classical local decision maximizes `Re(conj(a) v)` over symbols a. The Voronoi decision minimizes `|a-v|²`, equivalently maximizing `Re(conj(a) v)-|a|²/2`. On a constant-modulus alphabet the penalty is constant; on multiple radii it changes the choice.

The network updates one coordinate at a time. It keeps the current symbol whenever it belongs to the tied nearest set and otherwise uses the first symbol in the published order. The theorem assumes Hermitian weights, nonnegative diagonal entries, a fair asynchronous schedule, and exact arithmetic. Finite state space plus strict descent at every actual change yields termination at a fixed point.

The classical and inertial energies can both be evaluated on any trajectory. Only the energy associated with the update rule has the proved descent guarantee. Recovery of a specified target is a separate property.

Closed Voronoi cells can share boundaries. Figures clip exact half-plane intersections to a plotting window. The generic figure uses 12 explicitly specified, separated symbols; it is not an additional retrieval experiment. The four regular panels use the actual Experiment 1 alphabets.

Published alphabet coordinates are loaded from `configs/ordered_alphabets.json` so equal-radius ordering does not depend on platform-specific evaluations of complex exponentials and absolute values. Because ties fall back to the first symbol in array order, this order is part of the model definition. The original constructors remain available for inspection and custom, non-publication experiments.

## Scope

Experiment 1 diagnoses amplitude drift, not memory capacity. Experiment 2 uses a fixed synthetic HSV memory bank and paired corruptions. Experiment 3 measures batch query implementations; tree operation counts are proxies, not measured instruction counts. All benchmark methods include conservative current-state retention and exhaustive resolution of ambiguous boundary cases. Batch throughput does not establish the same speedup for an entire recurrent network.

The related paper by Garimella, Swamy, and Valle on asymmetric discrete-time networks concerns left/right update orientation. This repository retains Hermitian weights and studies symbol geometry; it does not extend the convergence theorem to arbitrary asymmetric weights.

## Experiments

| Stage | Python entrypoint | Notebook | Full protocol |
|---|---|---|---|
| Geometry and figures | `scripts/render_figures.py` | [00_geometry](../colab/00_geometry.ipynb) | Five paper figures and a recall overview |
| Saturation | `experiments/01_saturation.py` | [01_saturation](../colab/01_saturation.ipynb) | N=100; 4 alphabets; 2 initializations; 100 paired runs; 80-sweep cap |
| HSV recall | `experiments/02_hsv_recall.py` | [02_hsv_recall](../colab/02_hsv_recall.ipynb) | 24×24 images; P=4; 4 alphabets; 2 channels; 5 noise levels; 30 paired trials |
| Activation cost | `experiments/03_activation_cost.py` | [03_activation_cost](../colab/03_activation_cost.ipynb) | 4 families; cardinalities 16, 64, 256, 1024; 2 field sets; 20,000 queries; 5 repeats |

The noise study contains 2,400 trajectories. The additional load sweep contains 64 trajectories with only two trials per condition and is exploratory. The selected successful HSV illustration uses seed 120044901; it does not estimate average retrieval performance.

The saturation master seed is 20260831. The synthetic image bank uses seed 20260901. Per-trial seeds are present in the CSV measurements.

The default `independent` engine (`src/cvhnn/independent.py`, `independent_search.py`, `independent_benchmark.py`) supplies the paper's numerical kernels. Reference kernels remain in `saturation.py`, `recall.py`, and `activation_cost.py` and can be selected with `--engine reference`; that engine is for algorithm inspection and is not the source of the publication benchmark. Shared experimental inputs, trial enumeration, and CSV schemas are deliberate: independence concerns the numerical algorithms, not a different set of random experiments. This is a same-team implementation audit, not an external replication.

## Numerical conventions

All states and weights use complex128. In saturation, values within absolute tolerance `1e-12` of the optimum form the numerical tie set. In recall dynamics, the tolerance is `1e-12 * max(1, abs(optimum))`, matching the experimental protocol. The current state is retained if eligible; otherwise the first eligible array index is used. Static quantization and benchmark searches use an absolute `1e-12` tolerance on squared distance. This prevents nominally equal-radius symbols at zero from being selected by the last bit of a norm calculation.

These conventions approximate exact arithmetic. The theorem is stated for exact minimizers; numerical fixed points are checked under the specified tolerance. They are not asserted to be exact symbolic fixed points. A changing coordinate must have strictly smaller objective than the previous coordinate under this conservative selection. The audit records every computed energy change and checks a fresh terminal field, with `1e-10` as the reporting tolerance for an increase. Tests also compare small full trajectories against a reference that recomputes the field and complete energy after every update.

The field is maintained incrementally. Recall refreshes it after each sweep; saturation retains the original incremental convention and measures its drift against a fresh matrix-vector product. A complete no-change sweep certifies stopping. The limit of 80 sweeps is an experimental cap, not a theorem bound.

One BLAS thread is used by the launchers. Timing values depend on CPU, operating system, numerical libraries, and background load.

## Inputs and randomization

For saturation, the real and imaginary entries of the unmasked matrix are independent standard normals divided by `sqrt(2N)`; the diagonal is then zeroed. The Hermitian part is multiplied by beta=1, with alpha=0 on the diagonal. There are 100 independent matrices per alphabet and initialization; the two rules share each matrix, starting indices, and all permutations. Uniform and inner initializations are separate experiments. The master seed is 20260831.

Recall uses six deterministic 24-by-24 synthetic HSV patterns generated with seed 20260901; the noise study stores the first four. The complete procedural specification is `recall.generate_synthetic_hsv_memories`: an angular hue field with seeded rotations and sinusoidal textures, plus normalized radial, wave, blob, and diagonal saturation components. Its role is to define common input data. It is not an activation or optimization routine under test. The stored patterns are the quantized images, not the prequantized continuous fields.

The pseudoinverse cutoff is `rcond=1e-10`. Symbol replacements are Bernoulli per neuron and include the original symbol; nominal probability is therefore not the realized SER. HSV hue and saturation receive independent zero-mean Gaussian noise of the stated standard deviation, followed by wrapping/clipping and quantization. Thirty trials alternate targets with `trial % 4`; they are noise replicates for a fixed bank, not independent draws of memory sets. The load experiment has two trials per condition and is exploratory.

## Cardinality experiment

The four values 16, 64, 256, and 1024 denote **alphabet cardinalities**, not neuron counts or real-space dimensions. All alphabets lie in the complex plane; the field-generating network always has N=576 neurons and P=4 memories. Its memory bank is the same first four patterns from the six-pattern recall bank.

For each of four families and each cardinality, use 20,000 uniform fields in [-1.25 Rmax, 1.25 Rmax]^2 and 20,000 network fields collected initially and after up to three Voronoi sweeps. Network queries retain their actual current symbol; uniform queries receive a reproducibly sampled current index. Each pair of field and current index is shared across methods. Every run regenerates the 32 query files from fixed seeds and writes them, with their hashes, to `<output>/queries/`.

All methods retain the current symbol on numerical ties. KD-tree and structured paths detect competing nearest candidates and use exhaustive distance resolution when ambiguous. This fallback can cost O(|S|) at boundaries; expected fast-path cost is not a worst-case guarantee. Correctness is checked before timing and on nearest-neighbor midpoints and zero, with and without a current state.

Five timed repetitions follow a warm-up. Method order rotates across repetitions; one BLAS thread and one KD-tree worker are used. Preprocessing is recorded separately. The figure reports median **batch-average** microseconds per query, with the full observed repetition range. It does not measure single-call latency or end-to-end recall acceleration. The analytical flop counts describe the search core and exclude tolerance checks and boundary fallback.

## Metrics

- **E_c(s)** = `-Re(vdot(s, W@s))/2`.
- **E_r(s)** = `E_c(s) + vdot(s,s).real/2`. The added term compensates the candidate-magnitude contribution in nearest-symbol selection.
- **rho** = `sum(abs(s)**2)/(N*Rmax**2)`. This is normalized state magnitude, not recovery accuracy.
- **b** = fraction of neurons at maximum modulus, checked with numerical tolerance.
- **SER** = mean of unequal target and output symbol indices under the same alphabet order.
- **Exact recovery** = 1 iff SER is zero; its mean is the fraction of fully recovered targets.
- **Radial error** = mean absolute modulus difference divided by Rmax.
- **Angular error** is measured in cycles by the recall implementations; it averages pairs for which both normalized magnitudes exceed 1e-10 and returns zero when no such pair exists. It is not in radians.
- **RGB PSNR** uses decoded RGB images with values in [0,1]. The implementation reports 100 dB as a finite sentinel for zero MSE; the mathematical PSNR would be infinite.
- **Sweeps** include the final sweep with no changes, which verifies a fixed point.
- **Query time** is elapsed batch time divided by query count, in microseconds. Preprocessing is recorded separately. Speedup is relative to exhaustive Euclidean search on the same machine and input set.

The saturation figure shows mean ± one sample standard deviation across 100 runs, not a confidence interval. Paired rules share weights, initial states, and schedules. Recall means use 30 corruption/schedule trials per condition on one fixed synthetic memory bank; they are not estimates across independently sampled image collections.

Sweep-level violation counts are evaluated at sweep endpoints. The independent engine additionally checks each changed coordinate with a freshly computed local field, compares accumulated changes with complete quadratic forms, and verifies every coordinate of the terminal state. Reports are in `validation/*_coordinate_audit_summary.json`. The exact-arithmetic theorem is distinct from floating-point checks and numerical-tolerance fixed points.

## Theory-to-code traceability

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
