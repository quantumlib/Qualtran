# Cultivation Simulation Data: Methodology

This document describes how `cultivation_simulation_summary.csv` was produced.
This CSV is consumed by `cultivation_analysis.py` in the FLASQ cost model to
estimate the spacetime volume of magic state cultivation.

**Provenance:** Both the simulation data and the pipeline that produced it
are based on the code and methodology released alongside
*Magic state cultivation: growing T states as cheap as CNOT*
(Gidney, Jones, and Shutty, 2024). Our pipeline wraps their `cultiv`
package — using it for circuit construction, noise simulation, gap
splitting, and volume estimation — and extends it to a denser grid of
physical error rates.

## 1. Overview

Magic state cultivation produces high-fidelity $T$ states through a
post-selected circuit that injects, cultivates, and escapes a color code
into a surface code. The cost depends on three quantities:

- **Logical error rate** (`t_gate_cultivation_error_rate`): probability of a
  faulty $T$ state after cultivation.
- **Discard rate** (`keep_rate`): fraction of attempts surviving
  postselection.
- **Spacetime volume** (`expected_volume`): physical qubit$\cdot$rounds
  consumed per successful attempt, including resources wasted on
  intermediate postselection failures.

These quantities vary with the physical error rate $p$ and the
gap postselection threshold. This CSV tabulates them for all
$(p, d_1, \text{gap})$ combinations explored in the cultivation paper
(Gidney, Jones, and Shutty, 2024).

## 2. Raw Data Collection

**Source:** Sinter Monte Carlo simulation of the end-to-end cultivation
circuit with desaturation decoding.

**Compute:** ~3 months on a 128-core machine.

**Parameters:**
- Cultivation distances: $d_1 \in \{3, 5\}$
- Surface code distance: $d_2 = 15$
- 29 physical error rates from $p = 1.25 \times 10^{-4}$ to
  $p = 4.0 \times 10^{-3}$ (approximately $\sqrt{2}$ spacing)
- Shot counts: 5 billion ($d_1=3$) and 60 billion ($d_1=5$)
- Noise model: uniform depolarizing

**Output:** `stats_new.csv` — 4.4 million rows of standard sinter output
with custom detector counts encoding the decoding gap.

## 3. Gap Splitting (Deterministic)

The raw sinter data encodes the decoding confidence gap in custom
detector counts. `split_by_gap_threshold` (from the cultivation paper's
`cultiv` package) partitions each $(p, d_1)$ group into rows indexed
by gap threshold, where each row aggregates shots with gap $\geq$
threshold.

This step produces `end2end_gap_split.csv` with 5,595 rows. Each row
has columns: `error_rate`, `cultivation_distance`, `shots`, `errors`,
`discards`, `gap`, `attempts_per_kept_shot`.

**Terminology note:** The paper (Appendix D) refers to cultivation
postselection as filtering based on a "postselection threshold" that
controls the tradeoff between the discard rate and the logical error
rate of cultivated states. The `gap` column in this CSV is the
numerical realization of that threshold: it is the minimum decoding
confidence gap (from the desaturation decoder) required to accept a
cultivation attempt. Higher `gap` values correspond to stricter
postselection — lower logical error rates at the cost of higher
discard rates and therefore higher expected spacetime volume per
successful T state. The paper does not use the word "gap" directly;
it is inherited from the `cultiv` simulation package.

## 4. Binomial Fitting (Deterministic)

For each row, `sinter.fit_binomial` computes the maximum-likelihood
estimate of the logical error rate and confidence intervals at
factors 10, 100, and 1000:

$$
p_{\text{logical}} = \frac{\text{errors}}{\text{shots} - \text{discards}}
$$

Columns added: `best_estimate`, `low_10`, `high_10`, `low_100`,
`high_100`, `low_1000`, `high_1000`.

## 5. Derived Columns

### 5.1. `keep_rate`

$$
\text{keep\_rate} = 1 - \frac{\text{discards}}{\text{shots}}
$$

Fraction of cultivation attempts surviving all postselection stages
(both intermediate detector checks and the final gap threshold). Computed
directly from sinter data — deterministic.

### 5.2. `expected_volume` (Stochastic)

The expected spacetime volume per successful $T$ state, in physical
qubit$\cdot$rounds.

**Procedure:** For each $(p, d_1)$ group, we construct the noiseless
end-to-end cultivation circuit with parameters:

| Parameter | Value |
|-----------|-------|
| `dcolor` | $d_1$ |
| `dsurface` | 15 |
| `basis` | Y |
| `r_growing` | $d_1$ |
| `r_end` | 10 |
| `inject_style` | unitary |

We then apply uniform depolarizing noise at strength $p$ (matching the
row's physical error rate) and simulate $\sim 10^7$ shots using
`stim.FlipSimulator`. The simulation tracks:

1. **Active qubits** $q_k$: how many qubits are active at timestep $k$
   (monotonically non-decreasing as the color code grows into the surface
   code).
2. **Surviving shots** $s_k$: how many shots survive intermediate
   postselection at each timestep (checked via postselected detectors
   in the circuit).

The final timestep survival is overridden to match the sinter-derived
`keep_rate`, since the simulation cannot replicate the full decoder's
gap-based postselection:

$$
s_{K} = s_{K-1} = N_{\text{shots}} \times \text{keep\_rate}
$$

The expected volume is then:

$$
V = \sum_{k=0}^{K} \frac{s_k \cdot q_k}{N_{\text{shots}} \cdot \text{keep\_rate}}
$$

This integral weights each timestep by the fraction of shots alive and the
number of qubits active, amortized over the keep rate to account for
discarded attempts.

**Stochastic note:** This value depends on the `stim` random seed and is
not portable across `stim` versions or SIMD architectures. The data in
this CSV was generated with `stim==1.15.0`, seed 42, on x86-64 Linux
with ~$10^7$ shots per group.

### 5.3. `t_gate_cultivation_error_rate`

$$
p_T = 2 \times p_{\text{best\_estimate}}
$$

This factor of 2 comes from **Assumption 1** of the cultivation paper
(Gidney, Jones, and Shutty, 2024):

> For the cases considered by this paper, the logical error rate of
> $T|{+}\rangle$ cultivation can be estimated by doubling the logical
> error rate of $S|{+}\rangle$ cultivation (using the same circuit but
> with $T$ replaced by $S$).

The simulations measure $S$-state cultivation error rates (which are
computationally tractable). The ratio $p_T / p_S$ was observed to be
slowly growing as noise decreases and was verified only for $d_1 = 3$
(not $d_1 = 5$). The factor of 2 is a conservative upper bound within
the paper's regime of interest.

### 5.4. Ad-Hoc Parameters in `cultivation_analysis.py`

The following parameters are used when consuming this CSV in the FLASQ
cost model. Both are empirical choices without theoretical justification.

#### `slack_factor` (default: 0.995)

`round_error_rate_up` finds the smallest tabulated error rate ≥
`physical_error_rate * slack_factor`. The 0.5% slack prevents
floating-point rounding from pushing a queried rate just above a
nominally equal tabulated value (e.g., querying $p = 0.001$ when the
table contains $0.0009999...$). The value 0.995 was chosen ad-hoc as
"small enough to never jump to the wrong grid point, large enough to
absorb rounding."

#### `uncertainty_cutoff` (default: 100)

When selecting cultivation data for a given $(p, d_1)$, rows are
filtered by the ratio `high_10 / low_10` — the width of the 10×
likelihood-ratio confidence interval on the logical error rate. Rows
where this ratio exceeds 100 are discarded. This removes data points
deep in the low-error-rate tail where few or zero logical errors were
observed, making the binomial estimate unreliable. The value 100 was
chosen ad-hoc: loose enough to retain useful data across most of the
parameter space, tight enough to exclude points with essentially no
statistical power.

## 6. Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `error_rate` | float | Physical error rate $p$ |
| `cultivation_distance` | int | Cultivation fault distance $d_1$ |
| `shots` | int | Total sinter shots |
| `errors` | int | Logical errors observed |
| `discards` | int | Shots discarded by postselection |
| `gap` | int | Decoder confidence gap threshold |
| `attempts_per_kept_shot` | float | $\approx (\text{shots}+1)/(\text{shots}-\text{discards}+2)$ |
| `best_estimate` | float | MLE logical error rate ($S$-state) |
| `low_N` / `high_N` | float | Confidence interval at factor $N$ |
| `keep_rate` | float | $1 - \text{discards}/\text{shots}$ |
| `expected_volume` | float | Qubit$\cdot$rounds per kept $T$ state |
| `t_gate_cultivation_error_rate` | float | $2 \times \text{best\_estimate}$ |

## 7. Reproducing This Data

```bash
cd flasq_cultivation/

# Step A: Gap splitting (deterministic, ~14 min on 7GB CSV)
python scripts/analyze_sinter_data.py

# Step B: Volume computation (stochastic, ~80 min)
python scripts/compute_volumes.py --noise-model matched --seed 42
```

The pipeline code lives in the `flasq_cultivation` repository alongside
the raw data. It depends on the `cultiv` package released with
Gidney, Jones, and Shutty (2024) for circuit construction, gap
splitting, and volume estimation.

## 8. Known Limitations

1. **Assumption 1 (2× factor):** The $T/S$ error rate ratio was verified
   only for $d_1 = 3$ and is slowly growing at low noise. For $d_1 = 5$,
   the factor of 2 may underestimate the true $T$-state error rate.

2. **Stochastic volume:** `expected_volume` has ~0.1% relative noise at
   $10^7$ shots. Different seeds produce slightly different values.

3. **Seed portability:** Reproducing exact values requires the same
   `stim` version (1.15.0) and CPU architecture (SIMD width matters).

4. **Noise model:** The simulation uses uniform depolarizing noise,
   matching the cultivation paper's convention. Real hardware noise
   may differ.
