# CodeSim roadmap: CIGRE TB 575 benchmarks, unbalanced systems, impedance scanner, voltage stability, GPU EMT

## Context

You asked five design questions about extending CodeSim (`codesimpy`, installed from
`github.com/NoClassDefError/codesimpy` into the `hunting` conda env; workspace models in `test261113/`):

1. Which components are needed to build the CIGRE TB 575 benchmark examples?
2. What has to change to support unbalanced systems?
3. How to build an impedance scanner, both numerically and analytically?
4. How to add a static voltage stability tool that draws PV and QV curves?
5. How to add a parallel computing engine for CUDA-GPU-based EMT simulation?

The answers below come from reading TB 575 (chapters 4–9) and the CodeSim core
(`core/device.py`, `core/rl_grid.py`, `core/system.py`, `core/small_signal_analyzer.py`),
plus the workspace models (`cigre14.py`, `pq_load.py`, `induction_motor.py`, …).

### What CodeSim is today (constraints that shape every answer)

- **Each node has exactly one Voltage device and one Current device** (`Vdev[i]`, `Idev[i]`).
  The node voltage comes from the Voltage device (Capacitor, GFM, Infbus). Nodes with neither
  are removed by `RLGrid.kronReduction`.
- **The network is RL branches only**: `Abar = kron(A, I2)` and `L di/dt = Abar^T v - R i`, all
  in one per-unit base. There are no transformers, tap or phase shift, no switches, and no
  mutual coupling.
- **Ports are 2-D αβ.** There is no zero sequence and no per-phase representation.
- **Loads are written as RL branches to a `ground` node held by an `Infbus` at 0 V.** This
  workaround is in `cigre14.py:assembleNetwork`. Line charging is a `Capacitor` at the bus.
- **The power flow is positive-sequence only** (`RLGrid.solvePowerFlow`). Initialization
  looks for a dq equilibrium (`getOperatingPoints` → `least_squares` on `f_init`).
- **Linearization is done in the global DQ frame**: `A_cse` = the Jacobian of `f_init_sym`,
  where branch rows carry the `-J i` term. Analytic port models already exist:
  `SmallSignalAnalyzer.device_admittance` (one device, 2×2 dq) and `grid_impedance` (the
  network seen from a bus).
- **`kronReduction` with degree > 2 combines R + jL at the fundamental frequency.** That is
  only exact at 50/60 Hz, so it is wrong for EMT. Keep explicit (capacitive) nodes instead
  of reducing them.

---

## 1. Components for the TB 575 benchmarks

### 1a. Network building blocks (core work: not Devices today)

| Need (TB 575 section) | Status | What to build |
|---|---|---|
| Ω/km line data → pu, π-sections (all chapters; HV lines are 100–600 km, Tables 5.4/5.13) | Done by hand in `cigre14.py` | `NetworkBuilder` helper: `addBus(name, kV)`, `addLine(from, to, R', X', B', km, nPi)`, voltage zones → pu, `B'` split into bus `Capacitor`s. Use chained π-sections for long lines, roughly 1 per 50–100 km for EMT accuracy. |
| Grid equivalent from `V_G, S_SC, R/X` (§4, Tables 4.1/4.2, 6.6, 6.14, 7.16, 7.29) | Infbus + branch by hand | `gridEquivalent(Vg, Ssc, RX)` builder. Make `Infbus.w0` a reference instead of a parameter, so the frequency-perturbation tests in §4.4.2 work. |
| Two-winding transformer: R + jX, off-nominal tap, phase shift (YNd11 = 30°, Dyn1), Tables 5.5, 5.14, 5.19, 6.5, 6.13, 7.15, 7.28 | Missing | Generalize the incidence from `kron(A, I2)` to **2×2 block incidence**: a branch end carries `t·R(φ)`. This is a core change in `RLGrid.getStateAndOutputEquationsSym`, `_buildSymbolicEntry`, `getRMSEquationsSym`, `solvePowerFlow` and `getOperatingPoints`. Tap changers (±10 % in 0.625 % steps) become a tap parameter or reference. |
| Switches, breakers, fuses: S1–S3 meshing, grid switch, reclosing, islanding (§6.1.1, §7.2.1, §8.1, §8.2) | Missing | A switch is a branch whose R comes from a reference or schedule. Parameter values are passed at call time and are not part of `_structureKey`, so add a **parameter schedule** to `System.integrate` that updates `rlGrid.R` at breakpoints. No recompilation is needed. |
| Shunt capacitor banks: fixed Q at rated V (Tables 5.9, 5.18) | `Capacitor` exists | Builder only: `C = Q / V²`. |
| Shunt reactors: cable compensation (§5.4, Fig. 5.6) | Missing | `ShuntRL` Current device: states `i_d, i_q`, `L di/dt = v - R i - wLJ i`. |
| Constant-impedance loads (the PSCAD study used R, L loads) | Workaround with the ground node | `ImpedanceLoad` Current device, same form as `ShuntRL` from P, Q, pf at rated V. This removes the `ground` Infbus and makes loads visible to the impedance scanner. |
| Constant-PQ and ZIP loads, daily profiles (Figs 6.4, 7.6), coincidence factor (§9.3.4) | `PQLoad`, `ZIP` exist | Profiles via time-varying `referenceMap` entries (already supported). Add a small `coincidence(S_k, pf_k)` data helper. |
| Load-shedding relay (§8.1) | Missing | A switch plus a threshold event. This reuses `eventHandler`, extended to more than one event. |

### 1b. Generation and DER devices

| Need | Status | What to build |
|---|---|---|
| HV synchronous generators: H, Xd, X'd, X''d, T'd0, T''d0, Xq, X''q, T''q0, one q damper, first-order exciter (Ka = 20, Ta = 0.05), constant torque (Tables 5.7/5.16, §5.4.2) | `SG` is 8th-order with fundamental parameters (Xad, …) and a governor | `SG6` written in **standard parameters** (or a standard → fundamental converter feeding `SG`): M = 2H, exciter as Table 5.4.2, Pm and Vref as references. The slack generator at bus 9 stays an `Infbus` behind its transformer, as in the brochure. |
| DFIG wind farm, 350 MVA aggregated, with LVRT (§5.4) | Missing | Averaged DFIG: reuse the `InductionMotor` flux model with rotor voltage as an input, add a rotor-side converter (current control), a grid-side converter, a DC-link state, and crowbar/LVRT logic. This is the largest single device. |
| Fixed-speed induction generator, 5.5 kW (§7.4) | `InductionMotor` exists | Run it with negative `Tm`. Parameter set only. |
| PV, fuel cell, battery, CHP fuel cell as power profiles (§6.4, Table 6.18); wind as an in-phase current source (§8.2) | `GFL_C_F_L` / `GFL_P_F_L` exist | Time-varying `Pref(t)` / `Qref(t)`. No new device. |
| CHP diesel (310 kW) | — | A small `SG6`. |
| Storage converter with PLL-frequency governor (Ki integral + droop KSD) and an island switch to a fixed-frequency voltage source (Figs 8.14/8.15) | Partial (GFL + PLL) | `GFL_P_F_L` variant with an outer f→P loop plus a mode reference that selects the Fig. 8.15 voltage-source branch. Mode switching uses the switch/schedule mechanism from 1a. |
| Resource-side VSI (§8.1, Table 8.1): P-f / Q-V droop, Pmax loop, rms current limit → current control, current-peak hysteresis, resynchronization | Close to the existing GFM droop models | Droop GFM with an rms current limit (the existing CLC/AVI styles fit). Peak hysteresis is a switching effect: approximate it with a smooth saturation of `|i|` at `I_th`, and say so in the docstring. Resynchronization becomes a current-reference-to-zero mode. |
| DC bus with SMES and H₂ plant (§4.5) | — | Out of scope for EMT; it is a 24-hour energy-management study. At most, a converter with a `v_dc` state and `P_dc` as an input. |

### 1c. Benchmark systems to assemble (in `test261113/`, `@system`-decorated)

1. **EU MV, balanced** (§6.2). This is essentially `cigre14.py`. Move it onto `NetworkBuilder`, add the two
   110/20 kV Dyn1 transformers (Table 6.13) and switches S1–S3, and validate against the power-flow tables
   in Appendix 9.2.4.
2. **Resource-side benchmark** (§4, §8.1): grid equivalent + two droop VSIs + loads + a shedding relay, with
   the three-phase fault and reclose sequence of Fig. 8.6.
3. **HV transmission** (§5, NA and EU): 13 buses, `SG6` × 3, YNd11/YNyn0 transformers, π-chained
   300–600 km lines, shunt capacitors. Validate against Appendix 9.2.1/9.2.2. Then build the §5.4 DFIG + cable variant.
4. **MV islanding** (§8.2): the EU MV network + storage converter + wind current source at bus 7.
5. **NA MV and both LV benchmarks** (§6.1, §7): these **need the unbalanced support in part 2**.

---

## 2. Supporting unbalanced systems

### What already works and what does not
- EMT in stationary αβ **already carries negative sequence**. Unbalanced voltages and currents
  are represented, provided the network is symmetric and there is no zero sequence.
- Missing:
  - the zero sequence (grounding, YN transformers, single-line-to-ground faults);
  - asymmetric 3×3 branch matrices (non-transposed MV/LV lines, Tables 7.6–7.13 and 7.23–7.26);
  - single-phase and two-phase branches (NA laterals);
  - neutral conductors and grounding impedances (Tables 7.14, 7.27);
  - single-phase and Δ-connected loads and transformers, including the split-phase 7.2 kV/120-240 V transformer (Fig. 7.3);
  - an unbalanced power flow and initialization;
  - small-signal analysis around an operating point that is no longer an equilibrium.

### Recommended approach: a conductor-level (phase-domain) network, in three stages

**Stage 1: αβ0 with 3×3 branch matrices.** This is the smallest change that covers HV/MV unbalanced faults and the EU MV/LV networks without a neutral wire.
- Ports become 3-D (α, β, 0). Branch R and L become 3×3 per branch: convert `Z_abc` from the tables with the
  power-invariant Clarke matrix. `Abar` becomes block incidence with 3×3 blocks, reusing the block form from 1a.
- Existing 3-wire devices keep working through an adapter. They see αβ, and their 0-channel is an
  open circuit (zero current), or a grounding branch for YN windings.
- New elements:
  - `FaultBranch`: SLG, LL and LLG via switched R between conductors and ground;
  - per-phase `ImpedanceLoad`;
  - transformer vector groups as 3×3 incidence blocks (Δ blocks the 0-channel; YN passes it through `Z_gnd`).

**Stage 2: an arbitrary conductor count per node.** This covers the NA MV laterals and both LV benchmarks with a neutral.
- Each node has a conductor list (`a`, `b`, `c`, `n`, or a subset). Each branch maps conductors and carries
  its own R/L matrix: the primitive 4×4 matrix from Tables 7.10–7.13/7.25–7.26, or the 3×3 / 2×2 / 1×1
  after Kron reduction. Ground is the reference conductor (0 V). The `Infbus`-as-ground workaround goes away.
- Keep the CodeSim invariant that **every conductor-node has a voltage-defining device**: line
  capacitance, or a small parasitic C at neutral nodes. Then the ODE structure, the compiled
  `f/g` and the model cache are unchanged in spirit; only the dimensions grow.
- Devices declare which conductors they attach to (phase–neutral, phase–phase, or 3/4-wire).
  A 4-wire inverter with per-phase P/Q for the §7.4 microgrid is optional.

**Stage 3: steady state and analysis under unbalance.**
- **Power flow:** a three-phase current-injection Newton on `Y_abc(ω0)`, built with the same builder.
  Device power-flow equations become per phase. This replaces `solvePowerFlow` when there is more than one conductor.
- **Initialization:** an unbalanced steady state is **periodic**, not an equilibrium, in every frame. So
  `getOperatingPoints` (least squares on `f_init = 0`) no longer applies. Use the three-phase power flow as
  the guess, then **shooting** on one fundamental period (Newton on `x(T) - x(0)`, using `A_core` for
  the monodromy sensitivity).
- **RMS model:** a sequence-phasor network (positive, negative and zero networks coupled only at
  asymmetric elements and faults). This extends `getRMSEquationsSym`.
- **Small-signal:** linear time-periodic (LTP) analysis. Floquet multipliers from the monodromy matrix,
  or harmonic state space truncated at ±2 harmonics. `analyzeSmallSignal` stays valid only for balanced
  operating points.
- **Converter controls:** a dual-SRF current control and a sequence-separating PLL become optional
  variants of the existing GFL/GFM devices. They are not needed to run unbalanced cases, only to control them well.

---

## 3. Impedance scanner (analytical and numerical)

### 3a. Analytical: build on the existing `SmallSignalAnalyzer`
These already exist: the device-level `device_admittance` (2×2 dq, `delta i_out = -Y delta v`),
the network-level `grid_impedance`, and the conversion `dqToSequence`.

What to add (in `core/small_signal_analyzer.py`):
1. **Subsystem impedance at a cut.**
   - Reuse the component split from `System.scanPQ` (`system.py:110-138`): remove a branch, flood-fill one side.
   - Build that side as its own RLGrid. At the cut node, put a new `ScanSource` Voltage device whose voltage
     is the operating point plus the input `Δv`. Take the port current as the output.
   - Linearize with the existing CSE Jacobian machinery to get `(A, B, C, D)` with respect to `Δv`. Then
     `Y_side(s) = C (sI - A)^-1 B + D`.
   - This captures every converter, SG, load and line on that side, not just one device. Do it for both
     sides to get `Z_source` and `Y_load`.
2. **Closed-loop bus impedance as a cross-check.**
   - Add a perturbation current input at node k of the full model; this is the same input-augmentation idea.
   - Then `Z_bus = (Y_left + Y_right)^-1`, and its poles must equal the eigenvalues of `rlGrid.analyzeSmallSignal`.
3. **Stability tools on the 2×2 MIMO pair:**
   - the generalized Nyquist criterion: eigenloci of `Z_source · Y_load`, with an encirclement count;
   - `det(I + Z_s Y_l)` zeros;
   - a passivity check (eigenvalues of the Hermitian part of Y) per frequency band.
4. **Frames:**
   - Results come out in the global DQ frame. Keep `frequency_Hz` as the dq-frame frequency.
   - Also offer the αβ/sequence view, where a dq frequency f maps to f ± f0, through the modified-sequence form that `dqToSequence` already gives.
5. **Symbolic option:** for one device, substitute numeric parameters except a chosen few, and return
   `Z(s)` as a SymPy rational function, e.g. for showing the effect of `Kp_cc` or `Rv`. Expensive; single devices only.
6. **Sensitivities:** `dZ/dp` from finite differences on `p[...]["num"]`. This is cheap because parameter values do not
   trigger recompilation (`_structureKey`).

### 3b. Numerical: an EMT frequency sweep, model-agnostic, to validate 3a
1. **Injection hardware.**
   - Allow **several Current devices per node**, summing their currents in `RLGrid`. This is a small core change; the alternative is a dummy node behind a tiny branch.
   - Add a `PerturbationSource` Current device whose αβ output is driven by references: shunt current injection gives bus impedance.
   - For separating subsystems, add a `SeriesVoltagePerturbation` in a branch. It reuses the branch-voltage input from the transformer block-incidence change.
2. **Procedure for each frequency f_p (dq frame):**
   - Initialize. Run to steady state and keep that state.
   - Run two simulations from it: one injecting along d, one along q. Two independent injections are needed for a 2×2 matrix.
   - Window over an integer number of periods of both f_p and f0. Fourier-extract Δv and Δi at f_p in DQ.
   - Solve `Z = [Δv1 Δv2][Δi1 Δi2]^-1`.
   - Subtract a no-injection baseline run with the same solver settings, to remove numerical drift.
   - Use an injection amplitude of 1–5 %. Check linearity by repeating at two amplitudes.
3. **Speed:**
   - Inject several frequencies at once (multi-tone, with non-harmonically related tones).
   - Run frequencies in parallel with `multiprocessing`. Each process rebuilds the model once; the model cache is per process.
   - Set `max_step` from the highest injected frequency.
4. **Uses:** validate 3a, measure limiters and nonlinear operating regimes, and treat black-box devices.

### 3c. Integration
- System methods: `scanImpedanceAnalytic(...)` and `scanImpedanceEMT(...)`.
- A CLI method in `api/cli.py`, and an `impedanceScan` section in `OutputJsonFactory` (Bode and Nyquist data) for the frontend.

---

## 4. Static voltage stability: PV and QV curves

This tool sits next to the impedance scanner. The scanner answers small-signal questions at one
operating point. This tool answers how far the operating point can move before the power flow has
no solution (the nose of the PV curve), and how much reactive margin each bus has.

### 4a. What exists and what is missing
- `RLGrid.solvePowerFlow` (`rl_grid.py:440`) builds `Y` from the RL branches and solves the
  device power-flow equations with `scipy.optimize.root` (`hybr`) from a flat start.
- `System.scanPQ` (`system.py:100`) already scans the power surface of one subsystem over a grid
  of `(V, δ)` at a cut. That is a brute-force picture of the same nose. It is not a continuation method.
- Missing:
  - a load/generation **loading parameter λ** in the power-flow equations;
  - warm starts and an analytic power-flow Jacobian (`hybr` from a flat start fails near the nose,
    where the Jacobian is singular);
  - device **reactive and current limits** in power-flow form (generator Q limits, converter current limits);
  - a way to fix the voltage of any bus while leaving its Q free (for QV curves).

### 4b. PV curves: continuation power flow (CPF)
1. **Loading direction.**
   - λ scales a chosen set of loads and generators: `P_L,k(λ) = P_L,k0 (1 + λ d_k)`, the same for Q at
     constant power factor. Generator dispatch follows participation factors; the slack takes the rest.
   - Implement it through references: devices already read `Pref`, `Qref` and similar from `uref`, which
     `solvePowerFlow` substitutes from `vPref`. A `LoadingDirection` helper maps λ to those references,
     so no device code changes.
2. **Analytic Jacobian.**
   - Build `F(z, λ)` symbolically with the same setup as `solvePowerFlow`, and take `dF/dz` and `dF/dλ`
     with the existing `cseJacobian` (`core/symbolic.py`). Cache it in the model entry under
     `_structureKey`, like `A_cse`.
3. **Predictor–corrector.**
   - Predictor: the tangent `[dF/dz  dF/dλ] t = 0`, normalized, with a step σ.
   - Corrector: Newton on `F(z, λ) = 0` plus a **local (or pseudo-arclength) parameterization**
     equation. Switch the continuation parameter from λ to the fastest-changing bus voltage near the nose.
   - Adapt σ to the Newton iteration count. Stop at a voltage floor (e.g. 0.5 pu) or at λ < 0 on the lower branch.
4. **Nose and limits.**
   - The nose is where `dλ` in the tangent changes sign. Refine it with a few bisection steps, or solve
     the extended system `F = 0, J^T w = 0, ‖w‖ = 1` for the exact saddle-node point.
   - Device limits: each device may declare a power-flow limit (`Qmax`, `Qmin`, `Imax`). When one is
     hit, swap the device equation (PV → PQ at a Q limit; constant current at `Imax` for converters) and
     restart the corrector from the last point. Record a **limit-induced bifurcation** if λ then decreases.
   - Report: λ_max (the loading margin in MW), the critical bus, the binding limits, and the full curve
     for the monitored buses.

### 4c. QV curves
- At the scanned bus k, add a fictitious synchronous condenser: a `ScanQV` device whose power-flow
  equations are `V = V_set, P = 0`, with Q free. This is the static counterpart of `ScanSource` in 3a.
- Sweep `V_set` downward from the base-case voltage in small steps, with warm starts. Each point is one
  ordinary power flow, so no continuation is needed until the bottom of the curve; use CPF there if Newton fails.
- Report: the reactive margin `Q_margin = -min_V Q` (negative means the bus is already unstable
  without support), the voltage at the minimum, and `dQ/dV` along the curve (a negative slope means
  the operating point is on the unstable side).
- Run QV curves for many buses in parallel with `multiprocessing`, as in 3b (or on the GPU engine in part 5).

### 4d. Modal analysis and the link to dynamics
- **V–Q sensitivity:** from the power-flow Jacobian, take the reduced matrix
  `J_R = J_QV - J_Qθ J_Pθ^-1 J_PV`. Its eigenvalues give the voltage-stability margin per mode, and bus
  participation factors show which buses drive the weakest mode. Compute them at every CPF point,
  so the user can see the weak area form as load grows.
- **Cross-check with the dynamic model:** at the nose, one eigenvalue of the RMS model
  (`getRMSEquationsSym`) or of `analyzeSmallSignal` should reach zero (a saddle-node bifurcation) when
  loads are constant-power. With dynamic loads, controller limits or ZIP loads, the dynamic limit can come
  earlier (e.g. a Hopf bifurcation). Flag the case where the small-signal model goes unstable before the static nose.
- Run the impedance scanner (3a) at points along the PV curve: the Nyquist margin should shrink as λ approaches λ_max.

### 4e. Integration
- System methods: `scanPV(direction, ...)`, `scanQV(bus, vRange, ...)` and `voltageStabilityModes(...)`.
- A CLI method in `api/cli.py`, and a `voltageStability` section in `OutputJsonFactory` (PV and QV
  curves, nose points, margins, participation factors) for the frontend.
- Unbalanced systems: once the three-phase power flow from part 2, stage 3 exists, CPF runs on it with
  the same structure. Monitor the positive-sequence voltage or the lowest phase voltage.

---

## 5. Parallel computing engine: CUDA-GPU EMT

### 5a. Where the time goes today
- `System.integrate` (`system.py:264`) calls `scipy.integrate.solve_ivp` once per system. Each RHS call
  goes through a lambdified NumPy function (`_lambdify` with `cse=True`), or a Numba-JIT version after
  `enableNumba` (`rl_grid.py:367`). Per-call Python overhead dominates for the small and medium
  systems CodeSim builds today.
- A single benchmark-sized system (tens of nodes, a few hundred states) is **too small to fill a GPU**.
  The GPU pays off when many simulations run together: the EMT frequency sweep (3b), parameter sweeps,
  Monte Carlo, contingency and fault screening, time-domain checks of PV/QV results (4d), and training-data generation.

So the engine has two levels, built in this order:
1. **Batch (ensemble) parallelism:** many copies of one model with different parameters, references,
   injections or faults, integrated in lockstep. This is the main target.
2. **Intra-model parallelism:** one large network split across GPU threads. This only matters for
   networks with thousands of nodes, and comes later.

### 5b. Code generation from the existing symbolic model
- The model is already a SymPy expression in CSE form (`f`, `g`, `A_cse`), with parameters passed as
  values at call time rather than compiled in (`_structureKey`). Keep this: **one compiled kernel per
  structure, and a parameter matrix of shape `(n_batch, n_p)`**. Parameter sweeps then need no recompilation.
- Recommended backend: **JAX** first.
  - `sp.lambdify(..., modules="jax")` reuses the existing pipeline. `jax.vmap` adds the batch dimension,
    `jax.jit` compiles for CUDA, and `jax.jacfwd` gives the batched Jacobian for implicit solvers (or use
    the CSE Jacobian from `A_cse` if it is faster).
  - The same code runs on CPU, so tests run without a GPU.
- Alternative for the hot path: emit CUDA C from the CSE form with `sympy.printing.ccode` and compile
  it with CuPy `RawKernel` (NVRTC), one thread per batch member. Choose this only if profiling shows
  that JAX overhead matters.
- Add a `Backend` option (`"scipy"`, `"numba"`, `"jax-cpu"`, `"cuda"`) next to `enableNumba`.

### 5c. Integrator
- EMT models here are stiff (fast network RL modes next to slow controller and mechanical modes), and
  batch members must stay in lockstep. So use a **fixed-step implicit integrator**:
  - implicit trapezoidal (the EMTP standard), with a backward-Euler step after each discontinuity to damp
    numerical oscillation;
  - Newton on each step, with **batched dense LU** of `I - (h/2) J` (`jax.numpy.linalg.solve` under `vmap`,
    or cuSOLVER batched `getrf`/`getrs`). Reuse the factorization for several steps (modified Newton),
    and refresh it only when convergence slows or after a switching event.
- Optional: an adaptive-step mode where all members share the smallest accepted step. It is simpler
  than per-member steps and is enough for sweeps where all members behave similarly.
- **Precision:** use float64. Per-unit states, stiff dynamics and long runs make float32 unreliable.
  Data-center GPUs (A100/H100 class) have full-rate FP64; consumer GPUs run FP64 at 1/32–1/64 rate.
  Say so in the documentation, and benchmark both.

### 5d. Events, schedules and output
- Breakpoints and the parameter schedule from 1a (switches, faults, tap changes) are known in advance:
  align them to the fixed step and apply them as per-member masks on the parameter matrix.
- Threshold events (`eventHandler`, load-shedding relays): evaluate the event function for all members
  at every step and switch each member with a mask. Do not stop the batch.
- Keep states on the GPU. Decimate on the device and copy chunks to the host only at the output
  interval, then pass them to `OutputJsonFactory.writeSimulationChunk` as today. For sweeps, compute
  the reduction on the GPU (for example the Fourier extraction of Δv and Δi in 3b), so only the result comes back.

### 5e. Intra-model parallelism for large networks (later)
- Split the network at long lines. A distributed-parameter (Bergeron / travelling-wave) line model
  decouples its two ends by the travel time, which is at least one time step. Each subnetwork is then
  solved independently per step; this is the standard parallel-EMT approach.
- This needs the distributed-parameter line model as a new branch type. It ties in with
  `NetworkBuilder` (1a), which already knows line lengths.
- Within a subnetwork, use a sparse batched solve for the network part and one thread block per device group.

### 5f. Integration
- `System.integrateBatch(time_s, initialStates, references, parameterSets, solverSettings, Backend=...)`
  returns arrays of shape `(n_batch, n_t, n_x)`, or reduced outputs.
- `scanImpedanceEMT` (3b) and the time-domain checks in 4d call `integrateBatch` when a GPU backend is
  selected, instead of `multiprocessing`.
- A CLI option selects the backend and device, with a capability check that falls back to CPU.

---

## Suggested order of work

1. **NetworkBuilder, transformers (block incidence), switches/schedule, `ImpedanceLoad`/`ShuntRL`, `gridEquivalent`**
   (1a). This unlocks all balanced benchmarks and both scanner modes.
2. **Analytical subsystem scanner (3a, items 1–3)**, validated on the existing SMIB systems
   (`smib_gfl_c_f_lc.py`, `smib_gfm_lc.py`): poles of `Z_bus` = eigenvalues of `analyzeSmallSignal`.
3. **EMT sweep (3b)**, validated against item 2 on the same SMIBs.
4. **PV/QV tool (4b–4d)**: CPF with an analytic Jacobian first, then QV curves, device limits and
   modal analysis. It needs only the power flow, so it can start in parallel with item 3.
5. **GPU batch engine (5b–5d)**: the JAX backend and a fixed-step trapezoidal integrator, validated
   against `solve_ivp`, then used by the EMT sweep (3b).
6. **SG6, then the HV benchmark; the resource-side benchmark; MV islanding** (1b/1c). Run PV/QV
   curves on the HV benchmark once it exists.
7. **Unbalanced stages 1 → 2 → 3** (part 2), then the NA MV and LV benchmarks and LTP analysis.
8. **Intra-model GPU parallelism (5e)** with distributed-parameter lines, once networks are large enough to need it.

Core changes go in the `codesimpy` repo (installed from GitHub, not editable). Clone it and use
`pip install -e` before starting. Benchmark systems and new devices can live in `test261113/`
as flat modules with `@system`, as now.

## Verification

- **Power flow** against the TB 575 Appendix 9.2 tables:
  - 9.1/9.2 for HV;
  - 9.6 for EU MV, including the transformer taps in Table 9.7;
  - 9.3/9.4/9.8/9.9 for unbalanced NA MV and LV.
  Report the maximum |ΔV| and Δθ per bus.
- **Builder regression:** `cigre14.py` rebuilt on `NetworkBuilder` must give the same operating point and eigenvalues as now.
- **Scanner consistency:**
  - analytic `Z_bus` poles equal the full-model eigenvalues;
  - the EMT sweep matches the analytic scan within a few % in magnitude and degrees in phase, away from resonances;
  - the GNC verdict agrees with the sign of the real part of the dominant eigenvalue in a parameter sweep that crosses instability (e.g. SCR or `Kp`).
- **Unbalanced:** the SLG fault current on a simple radial feeder matches a hand calculation with sequence networks. A periodic steady state found by shooting returns to itself within 1e-8 after one period.
- **Dynamic studies:** reproduce the qualitative results of TB 575 Figs 5.7 (DFIG LVRT), 7.11–7.13 (microgrid islanding), 8.6–8.8 (droop VSIs, fault, reclose) and 8.16 (MV island transfer).
- **Voltage stability:**
  - a two-bus system (a source behind X feeding a constant-PQ load at unity power factor) has its nose at
    `P_max = V²/(2X)`, and the CPF nose matches it within 1e-6 pu;
  - CPF λ_max matches a brute-force `scanPQ` sweep at the same cut;
  - the QV minimum at a bus equals the CPF result when only that bus is loaded with Q;
  - with constant-power loads, a small-signal eigenvalue reaches zero at the CPF nose.
- **GPU engine:**
  - a batch of one matches `solve_ivp` (tight tolerances) within 1e-6 pu on the SMIB systems and `cigre14.py`;
  - each member of a parameter-sweep batch matches its own CPU run;
  - the GPU EMT sweep gives the same impedance as the CPU sweep in 3b;
  - report throughput (simulated seconds × members per wall-clock second) against batch size, CPU vs GPU.
