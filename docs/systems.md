---
title: System reference
---

How CodeSim assembles devices into a network, and every runnable system in `codesimpy/systems`: topology, network data, devices and their settings, references, disturbances and fault-clearing logic. Device equations are on the [component reference](./components.md).

(sys-network)=
## 1. Network model

Every system is a `System` subclass that builds a list of devices and an RL network and hands them to `RLGrid`. Units, frames and port conventions are those of the [component reference](#cmp-conventions): per unit of the system base, normalized time $\tau=\omega_b t$, stationary $\alpha\beta$ frame, generator convention.

(sys-rlgrid)=
### 1.1 Branch and bus equations

The network has $n$ buses and $b$ series RL branches. The incidence matrix $\bm A\in\mathbb R^{n\times b}$ has $A_{mk}=+1$ at the bus where branch $k$ leaves and $-1$ where it enters. Each bus $m$ holds exactly one voltage port $V_m$ and one current port $I_m$. The branch currents are the first $2b$ states (`iga1, igb1, …`), followed bus by bus by the states of $V_m$ and $I_m$.

$$
\begin{aligned}
            \bm v_m &= \bm g_{V_m}\!\left(\bm x_{V_m}\right) &&\text{bus voltage from the voltage port}\\
            \bm i^{I}_m &= \bm g_{I_m}\!\left(\bm x_{I_m},\bm v_m\right) &&\text{current-port injection}\\
            L_k\,\dot{\bm i}_k &= \sum_m A_{mk}\,\bm v_m-R_k\,\bm i_k &&\text{branch }k\;(\alpha\beta)\\
            \bm i^{V}_m &= \sum_k A_{mk}\,\bm i_k-\bm i^{I}_m &&\text{current delivered by the voltage port}
          \end{aligned}
$$

so each voltage port supplies whatever its bus sends into the branches minus what the current port injects (KCL), and its state equations receive $\bm i^{V}_m$. In compact form, with $\bar{\bm A}=\bm A\otimes\bm I_2$, $\dot{\bm i}_b=\bm L^{-1}(\bar{\bm A}^{\top}\bm v-\bm R\,\bm i_b)$. Since the frame is stationary, the branches carry no $\omega L\bm J$ terms. When a device base $(V_b,S_b)$ differs from the system base, its port input and output are scaled by $V_b/V_{b,sys}$ and $I_b/I_{b,sys}$ (`Inputscale`, `Outputscale`).

The output vector of the grid is $[\bm i_b;\bm v;\bm i^{I}]$, from which `calculateNetworkSignals` derives the bus injections $\bar{\bm A}\bm i_b$, the bus powers $\underline v_m(\sum_k A_{mk}\underline i_k)^{*}$ and the device powers.

(sys-kron)=
### 1.2 Passive buses

A bus with `None` as both its voltage and current device is removed by Kron reduction before the model is built: a degree-2 bus merges its two branches in series ($R=R_1+R_2$, $L=L_1+L_2$); a bus of degree $d\gt 2$ is replaced by a mesh between its neighbours (star–mesh), $Z_{ij}=Z_iZ_j\sum_k Z_k^{-1}$, split into $R=\operatorname{Re}Z_{ij}$, $L=\operatorname{Im}Z_{ij}$.

(sys-pf)=
### 1.3 Power flow and initialization

The unknowns are every bus voltage $V_m\angle\theta_m$ and the $P_d, Q_d$ of every device. The equations are the bus balances and two device equations each ([component reference](#cmp-init)):

$$
\bm Y=\bm A\,\operatorname{diag}\!\left(\frac{1}{R_k+jL_k}\right)\bm A^{\top},\qquad
          \underline V_m\left(\bm Y\underline{\bm V}\right)_m^{*}=\sum_{d\in m}\left(P_d+jQ_d\right),\qquad
          F_d(P_d,Q_d,V_m,\theta_m)=0 .
$$

They are solved from a flat start with `scipy.optimize.root` (hybrid method). Branch currents are then $(\underline V_{from}-\underline V_{to})/(R_k+jL_k)$, each device builds its initial guess, and the steady state is refined by least squares on $\bm f_{\text{init}}(\bm x)=\bm 0$ (every state equation except the frame angles). The system raises an error if either residual exceeds `residualTolerance`.

(sys-faults)=
### 1.4 Disturbances and clearing

All systems are disturbed through the amplitude reference $A_{ref}(t)$ of the infinite bus. Two schemes are used.

#### Time-based profile

Piecewise-linear through the breakpoints (used by every system in `"time"` mode):

$$
A_{ref}(t)=\operatorname{interp}\Big(t;\;
          \big(0,1\big),\big(t_{fs},1\big),\big(t_{fc},V_f\big),\big(t_{rs},V_f\big),\big(t_{rc},1\big)\Big),
$$

with $t_{fs}$=`faultStart_s`, $t_{fc}$=`faultComplete_s`, $t_{rs}$=`recoveryStart_s`, $t_{rc}$=`recoveryComplete_s`, $V_f$=`faultVoltage`. The breakpoints are passed to the integrator so it restarts at each kink.

#### Angle-based clearing

With `clearMode = "angle"` the fault is cleared when the GFM load angle reaches a target, found by the integrator's event detection after an eligibility time; the detected time $t_{clr}$ is stored as `eventTime_s` and returned as `clearingTime_s`.

:::{list-table}
:header-rows: 1

* - Family
  - Event function $h(\bm x)$
  - Direction
  - Eligible from
  - Voltage after detection
* - SMIB GFMs
  - $\sin\!\big((x_{clr}-\theta_{clr})/2\big)$, $x_{clr}$ = `clearingState` (default $\delta$), $\theta_{clr}$ = `clearAngle_rad`
  - either
  - $t_{fs}$ + `minimumFaultDuration_s`
  - ramps back to 1 over $t_{rc}-t_{rs}$
* - CIGRE14 / CIGRE4
  - $0$ if $\delta_w\in(\theta_{lo},\theta_{up})$, else $\min(\delta_w-\theta_{lo},\theta_{up}-\delta_w)$; $\delta_w$ = $\delta$ wrapped to $(-\pi,\pi]$
  - rising
  - `angleClearEligible_s`
  - steps back to 1
:::

For the SMIB GFMs the reference is $A_{ref}=1-(1-V_f)\,\operatorname{clip}\!\big(\tfrac{t-t_{fs}}{t_{fc}-t_{fs}},0,1\big)+(1-V_f)\,\operatorname{clip}\!\big(\tfrac{t-t_{clr}}{t_{rc}-t_{rs}},0,1\big)$ with $t_{clr}=t_{rs}$ in time mode. For CIGRE in angle mode, $A_{ref}=V_f$ for $t_{fs}\le t\lt t_{clr}$ and 1 otherwise, with defaults $t_{fs}=0.2$ s, $V_f=0$, window $(-\pi/3,0)$, eligible from 0.25 s.

(sys-list)=
### 1.5 System list

:::{list-table}
:header-rows: 1

* - System
  - Buses (after reduction)
  - States
  - Units
* - [`SMIB__GFM_A_C_L`](#sys-smib-gfm-acl)
  - 1
  - 10
  - GFM, embedded grid branch
* - [`SMIB__GFL_C_F_L`](#sys-smib-gfl-l)
  - 1
  - 8
  - GFL, embedded grid impedance
* - [`SMIB__SG`](#sys-smib-sg)
  - 2
  - 15
  - synchronous generator + capacitor
* - [`CIGRE14__GFM_A_C_LC__GFL_C_F_L`](#sys-cigre14)
  - 13
  - 92
  - GFM bus 9, GFL bus 7
* - [`CIGRE4__GFM_A_C_LC__GFL_C_F_L`](#sys-c4-acl-lc)
  - 4
  - 30
  - GFM bus 9, GFL bus 7
:::

(sys-smib)=
## 2. SMIB — single unit on an infinite bus

All SMIB systems use one infinite bus behind a grid impedance set by the short-circuit ratio and the R/X ratio $\rho$ (on the unit base $S_b = 8100$ VA, $V_b = 1$):

$$
\begin{aligned}
L_g &= \frac{1}{\mathrm{SCR}\sqrt{1+\rho^2}},\qquad R_g=\rho\,L_g\\
|R_g+jL_g| &= \frac{1}{\mathrm{SCR}}
\end{aligned}
$$

Depending on the unit, the grid impedance is either a network branch (`SMIB__SG`, whose terminal capacitor is a voltage port) or embedded in the unit (the unit is a current port and the network is a single bus with no branches).

(sys-smib-gfm-acl)=
### 2.1 SMIB\_\_GFM\_A\_C\_L

`systems/smib_gfm_a_c_l.py`

[GFM\_A\_C\_L](#cmp-GFM_A_C_L) has no capacitor, so it is a current port with $R_g+jL_g$ embedded; the network is a single bus with no branch.

```{code} text
bus 1 "Infinite bus":  Infbus (V-port)  +  GFM_A_C_L (I-port, embeds Rg + jLg)
```

#### Parameters

:::{list-table}
:header-rows: 1

* - Name
  - Default
  - Meaning
* - `SCR`, `rho`
  - 2, 0.1
  - grid strength and R/X ($L_g=0.4975$, $R_g=0.0498$)
* - `Pref`, `Vref`, `wref`
  - 1, 1.1, 1
  - GFM references
* - `Lf`, `Rv`, `Lv`
  - 0.07, 0.05, 0.01
  - filter and virtual impedance (override the device)
* - `Imax`, `Kp`, `Kp_cc`, `Ki_cc`
  - 1.5, 0.05, 20, 5
  - current limit, droop, current PI
* - `clearMode`
  - `"time"`
  - `"time"` or `"angle"`
* - `faultStart_s`, `faultComplete_s`
  - 0.2, 0.201
  - fault ramp
* - `recoveryStart_s`, `recoveryComplete_s`
  - 1.0, 1.001
  - recovery ramp (time mode)
* - `faultVoltage`
  - 0.5
  - residual voltage
* - `clearAngle_rad`, `clearingState`, `minimumFaultDuration_s`
  - −π/3, delta, 0.5
  - angle clearing
* - `validationTime_s`, `outputStep_s`
  - 2, 1e-4
  - run length and output step
:::

Solver: RK45, rtol 1e-3, atol 1e-5, maximum step $5\times10^{-3}/\omega_b$. The reported PCC voltage is the reconstructed $\bm v_{pcc}=\bm d+L_g\dot{\bm i}$ of the device.

(sys-smib-gfl-l)=
### 2.2 SMIB\_\_GFL\_C\_F\_L

`systems/smib_gfl_c_f_l.py`

LVRT benchmark: [GFL\_C\_F\_L](#cmp-GFL_C_F_L) with the grid impedance embedded (its PLL measures the reconstructed terminal voltage), on a single infinite bus.

```{code} text
bus 1 "PCC / Infinite bus":  Infbus (V-port)  +  GFL_C_F_L (I-port, embeds Rg + jω0Lg)
```

:::{list-table}
:header-rows: 1

* - Name
  - Default
  - Meaning
* - `SCR`, `rho`
  - 2, 0.25
  - $L_g=0.4851$, $R_g=0.1213$
* - `Pref`, `Qref`
  - 1, −0.1
  - $I_{d,ref}=$`Pref`, $I_{q,ref}=-$`Qref` (so $Q=V\cdot$`Qref`; the default absorbs reactive power)
* - `K1`, `Id2`
  - 1, 1.0
  - FRT gain and active current
* - device overrides
  - KpPll = KiPll = 0.2
  -  
* - fault
  - 0.3→0.301 to 0.5; 0.6→0.601 back to 1
  - time profile
* - `validationTime_s`, `outputStep_s`
  - 1, 1e-4
  -  
:::

Solver: DOP853, rtol 1e-3, atol 1e-5, max step $5\times10^{-3}/\omega_b$. Signals include the reconstructed $v_d,v_q$, P, Q, the FRT blend $m$, the FRT current references, the raw and limited PLL frequency and the PLL angle error $\theta-\theta_{grid}$.

(sys-smib-sg)=
### 2.3 SMIB\_\_SG

`systems/smib_sg.py`

[Synchronous generator](#cmp-SG) with a terminal capacitor (default $C=0.068$) on an infinite bus.

```{code} text
[Capacitor + SG]  bus 1 "SG PCC" ──── Rg + jLg ──── bus 2 "Infinite bus"  [Infbus + ZeroCurrent]
```

:::{list-table}
:header-rows: 1

* - Name
  - Default
  - Meaning
* - `SCR`, `rho`
  - 2, 0.25
  - grid impedance
* - `Ka`, `Vf0`
  - 0.5, 1.0
  - AVR gain, field offset (override the device)
* - `Vdref`
  - 1.2812
  - AVR set point; $P_{ref}=0$, $V_{q,ref}=0$, $\omega_{ref}=1$
* - fault
  - 0.10→0.101 to 0.2; 0.20→0.201 back
  - time profile
* - `validationTime_s`, `outputStep_s`, `residualTolerance`
  - 0.4, 2e-3, 1e-7
  -  
:::

**Initialization.** The standard power flow and initialization ([1.3](#sys-pf)), using the SG power-flow equations from the [component reference](#cmp-SG). The generator starts at synchronous speed.

(sys-cigre14)=
## 3. CIGRE14 — full CIGRE MV feeder

`systems/cigre14.py` · *base class _CIGRE14*

`CIGRE14__GFM_A_C_LC__GFL_C_F_L`: the CIGRE medium-voltage benchmark feeder (buses 1–11) fed from an infinite bus, with a GFM at bus 9 and a GFL at bus 7. System base $S_b=25$ MVA, $V_{LL}=20$ kV, 50 Hz; GFM base 5 MVA; GFL base 1.5 MVA.

```{code} text
source ─ 1 ─ 2 ─ 3 ─ 4 ─ 5 ─ 6
                   └ 8 ─ 9 ─ 10 ─ 11        GFM_A_C_LC at 9,  GFL at 7
                     └ 7                     loads: RL branches from 1, 3…11 to "ground"
```

#### Network

Lines are series RL branches; each line's charging capacitance is split half to each end and lumped as a [Capacitor](#cmp-Capacitor) voltage port at every bus (with a ZeroCurrent current port). Loads are series RL branches from their bus to a node `ground` held at zero by a second Infbus with $A_{ref}=0$. The source bus holds the Infbus that carries the fault. Bus 9 has the GFM as its voltage port, with the bus-9 line capacitance as the GFM capacitor (converted to GFM base, $C=C_9\cdot S_b/S_{b,GFM}$); bus 7 has the GFL as current port next to its line capacitor.

:::{list-table}
:header-rows: 1

* - Branch
  - R (pu)
  - L (pu)
  - C (pu)
  - Branch
  - R (pu)
  - L (pu)
  - C (pu)
* - source–1
  - 0.0014975
  - 0.1249752
  - 0
  - 5–6
  - 0.0482213
  - 0.068915
  - 0.0011702
* - 1–2
  - 0.0883013
  - 0.126195
  - 0.0021429
  - 7–8
  - 0.0522919
  - 0.0747325
  - 0.0012690
* - 2–3
  - 0.1384013
  - 0.197795
  - 0.0033587
  - 8–9
  - 0.01002
  - 0.01432
  - 0.0002432
* - 3–4
  - 0.0191006
  - 0.0272975
  - 0.0004635
  - 9–10
  - 0.0241106
  - 0.0344575
  - 0.0005851
* - 3–8
  - 0.0407063
  - 0.058175
  - 0.0009879
  - 10–11
  - 0.0103331
  - 0.0147675
  - 0.0002508
* - 4–5
  - 0.017535
  - 0.02506
  - 0.0004255
  -  
  -  
  -  
  -  
:::

:::{list-table}
:header-rows: 1

* - Load bus
  - R (pu)
  - L (pu)
  - Load bus
  - R (pu)
  - L (pu)
* - 1
  - 1.19486
  - 0.27929
  - 7
  - 236.111
  - 146.329
* - 3
  - 42.4688
  - 17.6818
  - 8
  - 40.0826
  - 10.0457
* - 4
  - 54.4944
  - 13.6576
  - 9
  - 31.4815
  - 19.5105
* - 5
  - 32.3333
  - 8.10350
  - 10
  - 42.2892
  - 12.5524
* - 6
  - 42.9204
  - 10.7569
  - 11
  - 71.3235
  - 17.8754
:::

#### Units

- **GFM at bus 9**: [GFM\_A\_C\_LC](#cmp-GFM_A_C_LC) with $L_v=0.3$, $R_v=0.1$, $L_f=0.07$, $I_{max}=1.5$, $K_p=0.05$, $K_{p,cc}=20$, $K_{i,cc}=5$; references `VSGPref` 1, `VSGVref` 1.1, `VSGwref` 1.
- **GFL at bus 7**:[GFL\_C\_F\_L](#cmp-GFL_C_F_L) with `GFLLf` 0.07, `GFLKpCc` 0.3, `GFLKiCc` 1, `GFLKpPll` 0.2, `GFLKiPll` 0.02; `GFLPref` 1, `GFLQref` 0. The references are $I_{d,ref}=$`GFLPref`, $I_{q,ref}=-$`GFLQref` (so $Q=V\cdot$`GFLQref`). Optional `GFLVth`, `GFLK1`, `GFLId2`, `GFLImax`.

#### Disturbance

:::{list-table}
:header-rows: 1

* - Name
  - Default (time)
  - Default (angle)
* - `faultStart_s`, `faultComplete_s`
  - 0.10, 0.101
  - 0.20, 0.201
* - `faultVoltage`
  - 0.1
  - 0
* - `recoveryStart_s`, `recoveryComplete_s`
  - 0.25, 0.2501
  - –
* - `angleClearEligible_s`, window
  - –
  - 0.25, (−π/3, 0)
* - `validationTime_s`, `outputStep_s`
  - 1, 1e-4
  - 1, 1e-4
:::

Solver: DOP853, rtol 1e-3, atol 1e-5, max step $5\times10^{-3}/\omega_b$.

(sys-cigre4)=
## 4. CIGRE4 — four-node reduction

(sys-c4-reduction)=
### 4.1 Reduced corridor

`_CIGRE4GFMAdmittance.buildReducedNetwork` keeps buses 7, 8 and 9 of the CIGRE feeder and replaces everything upstream of bus 8 by the infinite bus behind one equivalent branch, the series sum of source–1–2–3–8. All loads and the other laterals are omitted.

```{code} text
Infinite bus ── Z_up ── 8 ── Z_78 ── 7
                        │
                        └── Z_89 ── 9
```

:::{list-table}
:header-rows: 1

* - Branch
  - R (pu)
  - L (pu)
  - R/X
  - C (pu)
* - upstream (source–8)
  - 0.268906
  - 0.507140
  - 0.5302
  - 0.006489
* - 7–8
  - 0.052292
  - 0.074733
  - 0.6997
  - 0.001269
* - 8–9
  - 0.010020
  - 0.014320
  - 0.6997
  - 0.000243
:::

Bus line-charging capacitances (half of each adjacent line): bus 7 0.000635, bus 8 0.004001, bus 9 0.000122 (system base 25 MVA). Two options reshape the upstream equivalent $Z_{up}=R+jL$:

$$
\texttt{gridImpedanceScale}=s:\;(R,L)\to(sR,\,sL);\qquad
          \texttt{gridRX}=\rho:\;L=\frac{|Z_{up}|}{\sqrt{1+\rho^2}},\;R=\rho L\;\text{(same }|Z_{up}|\text{)}.
$$

Common parameters of the CIGRE4 systems are those of [CIGRE14](#sys-cigre14) (bases, GFL settings, references, fault), with max step `maximumStep_tau`$/\omega_b$ = $2\times10^{-3}/\omega_b$.

(sys-c4-acl-lc)=
### 4.2 CIGRE4\_\_GFM\_A\_C\_LC\_\_GFL\_C\_F\_L

`systems/cigre4_gfm_a_c_lc.py`

The four buses of 4.1 with the CIGRE14 units: [GFM\_A\_C\_LC](#cmp-GFM_A_C_LC) as the voltage port of bus 9 (its capacitor is the bus-9 line capacitance, no extra filter capacitor) and [GFL\_C\_F\_L](#cmp-GFL_C_F_L) at bus 7 next to the bus-7 line capacitor; bus 8 has its line capacitor. This is the p261030 tier-A benchmark. Options `gridImpedanceScale` (default 1) and `gridRX` (default None) reshape the upstream equivalent as in 4.1.
