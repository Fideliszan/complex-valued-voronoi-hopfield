# Research overview

For a finite complex alphabet S, the classical local decision maximizes `Re(conj(a) v)` over symbols a. The Voronoi decision minimizes `|a-v|²`, equivalently maximizing `Re(conj(a) v)-|a|²/2`. On a constant-modulus alphabet the penalty is constant; on multiple radii it changes the choice.

The network updates one coordinate at a time. It keeps the current symbol whenever it belongs to the tied nearest set and otherwise uses the first symbol in the published order. The theorem assumes Hermitian weights, nonnegative diagonal entries, a fair asynchronous schedule, and exact arithmetic. Finite state space plus strict descent at every actual change yields termination at a fixed point.

The classical and inertial energies can both be evaluated on any trajectory. Only the energy associated with the update rule has the proved descent guarantee. Recovery of a specified target is a separate property.

## Geometry

Closed Voronoi cells can share boundaries. Figures clip exact half-plane intersections to a plotting window. The generic figure uses 12 explicitly specified, separated symbols; it is not an additional retrieval experiment. The four regular panels use the actual Experiment 1 alphabets. Filled points mark symbols, hollow rings identify maximum-modulus symbols, and thin lines delimit the Voronoi cells. Muted fills distinguish adjacent cells without encoding numerical values.

## Scope

Experiment 1 diagnoses amplitude drift, not memory capacity. Experiment 2 uses a fixed synthetic HSV memory bank and paired corruptions. Experiment 3 measures batch query implementations; tree operation counts are proxies, not measured instruction counts. All benchmark methods include conservative current-state retention and exhaustive resolution of ambiguous boundary cases. Batch throughput does not establish the same speedup for an entire recurrent network.

The related paper by Garimella, Swamy, and Valle on asymmetric discrete-time networks concerns left/right update orientation. This repository retains Hermitian weights and studies symbol geometry; it does not extend the convergence theorem to arbitrary asymmetric weights.
