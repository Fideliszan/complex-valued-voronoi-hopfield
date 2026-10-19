# Metrics and sampling units

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

Sweep-level violation counts are evaluated at sweep endpoints. The independent
engine additionally checks each changed coordinate with a freshly computed
local field, compares accumulated changes with complete quadratic forms, and
verifies every coordinate of the terminal state. Reports are in
`validation/*_coordinate_audit_summary.json`. The exact-arithmetic theorem is
distinct from floating-point checks and numerical-tolerance fixed points.
