---
title: Validation against Simscape Electrical
---

CodeSim results are checked against an independent EMT tool: **Simscape Electrical, Specialized Power Systems (SPS), MATLAB R2024b**. For each case, an equivalent SPS model is built from the CodeSim system and both are simulated through the same disturbance.

(val-smib-gfm-a-c-l)=
## SMIB · GFM_A_C_L

The case is `systems/SMIB_NL__GFM_A_C_L.cir` (the example of the [guide](#guide-netlist)): a `GFM_A_C_L` converter without a PCC capacitor feeds an infinite bus through the grid branch $R_g + jX_g$, with SCR = 2 and $R/X = 0.1$. From 0.2 s to 1.0 s the grid voltage drops to 0.5 pu (1 ms ramps).

### The SPS model

`tests/simulink/build_smib_gfm_a_c_l.m` builds `SMIB_GFM_A_C_L_SPS.slx` programmatically. The model holds no numbers of its own: every value comes from the struct `smib`, which `compare_smib_gfm_a_c_l.py` derives from the CodeSim system.

* **Circuit (SPS, physical units).** The converter is an average model: three Controlled Voltage Sources with a grounded neutral. It connects through $L_f$, a Three-Phase V-I Measurement at the PCC, and a Three-Phase Series RLC Branch ($R_g$, $L_g$) to three Controlled Voltage Sources for the infinite bus. powergui runs in continuous mode and starts from the AC steady state of the initialized sources.
* **Infinite bus.** As codesimpy `Infbus`: $\theta = \theta_0 + \omega_b w_0 t$, $\dot A = \omega_b k_A (A_{ref}(t) - A)$, $\bm v = A\,[\cos\theta, \sin\theta]$. $A_{ref}$ is the PWL of the netlist.
* **Control.** The `GFM_A_C_L` equations ([component reference](./components.md)), term by term, in a MATLAB Function block. Its inputs are the measured PCC voltage and current, in per unit and in the converter's $dq$ frame $\theta$. The states $i_{ref}$, $x_{cc}$, $\delta$ and $\theta$ are Simulink integrators. The filter current $i_L$ is the current of the physical inductor $L_f$.
* **Converter voltage.** CodeSim writes the filter as $L_f\,\dot{\bm i}_L = \bm v_{ctrl} - \bm v_{PCC}/2$ in the $dq$ frame, with ideal decoupling. The SPS converter therefore applies $\bm v_{conv} = \bm v_{ctrl} + \bm v_{PCC}/2 + w L_f \bm J \bm i_L$, so the physical inductor follows exactly that equation. The PCC voltage feed-forward closes an algebraic loop through the inductive divider (loop gain ≈ 0.44), which Simulink solves.
* **Per unit.** As in codesimpy: amplitude-invariant Clarke transform, voltage base $V_b\sqrt{2/3}$ (peak phase), current base $S_b/(\sqrt{3/2}\,V_b)$, $\tau = \omega_b t$. The SPS model uses $V_b = 400$ V and the netlist's $S_b = 8.1$ kVA. Per-unit results do not depend on the voltage base.
* **Initial state.** The control integrators start from CodeSim's operating point, and the source phasors are set from it. Both simulations start in the same steady state: $|v_{PCC}| = 1.05$, $P = 1$, $Q = 0.245$ pu.

### Results

:::{figure} figures/smib-gfm-a-c-l-sps-comparison.png
:alt: |v_PCC|, P, Q and δ of CodeSim (solid blue) and Simscape SPS (dashed orange), 0 to 1.5 s and a zoom on the fault onset; the curves coincide.
:label: fig-smib-gfm-sps

CodeSim (solid) and Simscape SPS (dashed) through the voltage dip. The converter loses synchronism twice during the dip and after its clearing ($\delta$ wraps at ±π) and resynchronizes; both tools follow the same trajectory. Right: the first 60 ms after the fault.
:::

Deviation of CodeSim from SPS over 0–1.5 s, on CodeSim's 0.1 ms output grid ($\delta$ modulo $2\pi$):

:::{list-table}
:header-rows: 1

* - Signal
  - Max deviation
  - RMS deviation
* - $|v_{PCC}|$ (pu)
  - 7.2e-4
  - 7.2e-5
* - $P$ (pu)
  - 3.3e-3
  - 2.6e-4
* - $Q$ (pu)
  - 3.4e-3
  - 2.5e-4
* - $i_{Ld}$ (pu)
  - 2.7e-3
  - 2.4e-4
* - $i_{Lq}$ (pu)
  - 2.2e-3
  - 1.8e-4
* - $\delta$ (rad)
  - 8.0e-4
  - 2.1e-4
:::

For every signal, the largest deviation occurs at about 1.22 s. That is the fast transient in which the converter resynchronizes after its second pole slip, where $P$ and $Q$ change by more than 1 pu within a few milliseconds. The rest of the run, including the voltage ramps, stays well below these maxima, as the RMS values show.

**Solvers.** CodeSim uses the netlist's settings (`method=radau`, `reltol=1e-6`) and ran 1.5 s in 2.5 s. SPS uses `ode23tb` with RelTol 1e-7 and a 20 µs maximum step, and ran in 24 s (MATLAB start-up and model build not included).

### Reproduce

```bash
python tests/simulink/compare_smib_gfm_a_c_l.py      # redraws the figure, prints the deviations
python -m pytest tests/test_simulink_smib.py          # the same comparison as a test
```

Both need MATLAB R2024b with Simscape Electrical; set `CODESIM_MATLAB` to `matlab.exe` if it is not in `C:\Program Files\MATLAB\R2024b\bin`. The test is skipped without MATLAB. The script also writes `tests/simulink/smib_gfm_a_c_l_params.mat`, so `SMIB_GFM_A_C_L_SPS.slx` can be opened and run by hand, and the deviations to `docs/figures/smib-gfm-a-c-l-sps-comparison.json`.
