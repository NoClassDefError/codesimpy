---
title: Component reference
---

The equations of every device in `codesimpy/components`, transcribed from the code: states, references, parameters with their defaults, the state and port equations, the power-flow equations and how the initial state is built.

(cmp-conventions)=
## 1. Conventions

All components share the notation below. Every equation on this page is a direct transcription of `getStateAndOutputEquationsSym`, `getPowerFlowEquations` and `initFromPowerFlow` of the class; where the code simplifies algebraically, both forms are shown.

(cmp-units)=
### 1.1 Units and time

- Each device is written in per unit of its own base $(V_b, S_b)$ with current base $I_b = S_b / (\sqrt{3/2}\,V_b)$. `RLGrid` rescales the port input and output when the device base differs from the system base.
- Time is normalized: $\tau = \omega_b t$ with $\omega_b = 2\pi f$. Throughout, $\dot x \equiv \mathrm{d}x/\mathrm{d}\tau$. Inductances $L$ and capacitances $C$ are therefore per-unit reactances and susceptances at nominal frequency, and a frequency $\omega = 1$ is nominal.
- Each symbol carries the device index as a suffix in the code (`vCd_vsg`); the index is omitted here.

(cmp-frames)=
### 1.2 Frames and operators

Port quantities are in the stationary $\alpha\beta$ frame. A device with a frame angle $\theta$ works internally in its own rotating $dq$ frame. With

$$
\mathbf R(\theta)=\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix},\qquad
          \mathbf J=\begin{bmatrix}0 & -1\\ 1 & 0\end{bmatrix},\qquad
          \mathbf e(\phi)=\begin{bmatrix}\cos\phi\\ \sin\phi\end{bmatrix},
$$

a port vector is rotated in and out as $\mathbf x_{dq}=\mathbf R(\theta)^{\top}\mathbf x_{\alpha\beta}$ and $\mathbf x_{\alpha\beta}=\mathbf R(\theta)\,\mathbf x_{dq}$. $\mathbf J$ is multiplication by $j$: a term $\omega L\mathbf J\mathbf i$ is the rotational (speed) voltage of an inductor in the $dq$ frame, and $-\omega\mathbf J\mathbf x$ in a state equation is the frame-rotation term. Bold symbols are 2-vectors $[x_d, x_q]^\top$, $\mathbf a\cdot\mathbf b=a_d b_d + a_q b_q$, and $|\mathbf a| = \sqrt{\mathbf a\cdot\mathbf a}$.

**Frame angle and load angle.** In the GFMs the frame angle obeys $\dot\theta=\omega$ (the frequency reference), and the internal EMF sits at the angle $\delta$ *within* that frame. The absolute EMF angle is $\theta+\delta$ and its frequency is $\omega+\dot\delta$. Frame angles have no initialization residual; they are set by the power flow.

(cmp-ports)=
### 1.3 Ports and power

:::{list-table}
:header-rows: 1

* - Port type
  - Input $\mathbf u$ ($\alpha\beta$)
  - Output $\mathbf y$ ($\alpha\beta$)
* - `"Voltage"`
  - current $\mathbf i$ the device delivers into the network
  - bus voltage $\mathbf v$
* - `"Current"`
  - bus voltage $\mathbf v$
  - current $\mathbf i$ injected into the bus
:::

Both port types use the **generator convention**: the power the device delivers to the bus is

$$
P = \mathbf v\cdot\mathbf i = v_d i_d + v_q i_q,\qquad Q = v_q i_d - v_d i_q .
$$

A load therefore has negative $P$. In the power flow, $P$, $Q$, $V$ and $\theta$ are the device's injection and its bus voltage magnitude and angle.

(cmp-limiters)=
### 1.4 Limiter functions

The limiters are written with absolute values so that SymPy can differentiate them:

$$
\begin{aligned}
            \operatorname{pos}(x) &= \max(0,x) = \tfrac12\left(x+|x|\right),\\
            \operatorname{clip}(x,a,b) &= a+\operatorname{pos}(x-a)-\operatorname{pos}(x-b),\\
            \max(r,I) &= \sqrt{\tfrac12\left(r^2+I^2+|r^2-I^2|\right)}\quad(r,I\ge 0),\\
            \operatorname{sat}_{I}(\mathbf i) &= \frac{I\,\mathbf i}{\max(|\mathbf i|, I)}\qquad\text{(circular limit: unchanged inside the circle, scaled onto it outside).}
          \end{aligned}
$$

(cmp-init)=
### 1.5 Power flow and initialization

Each device contributes two power-flow equations $F(P,Q,V,\theta)=0$ at its bus (with its references substituted). After the power flow, `initFromPowerFlow` gives an initial guess of the device state in the frame of its bus voltage ($\mathbf v_{dq}=[V,0]^\top$, so $\theta$ equals the bus angle), and the system solves $\mathbf f_{\text{init}}(\mathbf x)=\mathbf 0$ by least squares, where $\mathbf f_{\text{init}}$ is every state equation except the frame angles listed in `thetaIndices` (`getFInit`). In the steady state $\mathbf i_{dq}$ at the terminal is $[P,-Q]^\top/V$.

**RMS model.** `getRMSEquationsSym` returns the EMT model with $\dot{\mathbf x}_f=0$ as a DAE: $\dot{\mathbf x}_s=\mathbf f_{\text{rms}}$ for the slow states (`thetaIndices` and `rmsDynamicIndices`) and $\mathbf 0=\mathbf g_{\text{rms}}$ for the others. The equations are written in a global frame rotating at 1 pu, so each frame-angle row becomes $\dot\theta=\omega-1$.

(cmp-overview)=
### 1.6 Overview

:::{list-table}
:header-rows: 1

* - Class
  - Port
  - States
  - References
  - Description
* - [`Infbus`](#cmp-Infbus)
  - V
  - 2
  - $A_{ref}$
  - Infinite bus, slack
* - [`ZeroCurrent`](#cmp-ZeroCurrent)
  - I
  - 0
  - –
  - Empty current port
* - [`Capacitor`](#cmp-Capacitor)
  - V
  - 3
  - –
  - Shunt capacitor with conductance
* - [`SG`](#cmp-SG)
  - I
  - 8
  - $P_{ref},V_{d,ref},V_{q,ref},\omega_{ref}$
  - Synchronous generator with AVR and governor
* - [`GFM_A_C_LC`](#cmp-GFM_A_C_LC)
  - V
  - 10
  - $P_{ref},V_{ref},\omega_{ref}$
  - Virtual admittance + current PI, circular limit, LC
* - [`GFM_A_C_L`](#cmp-GFM_A_C_L)
  - I
  - 8
  - $P_{ref},V_{ref},\omega_{ref}$
  - Virtual admittance + current PI, circular limit, no capacitor
* - [`GFL_C_F_L`](#cmp-GFL_C_F_L)
  - I
  - 6
  - $I_{d,ref},I_{q,ref}$
  - Current references + current PI, PLL, FRT priority
:::

Converter names follow `Role_Control_Limiter_Filter`: role `GFM`/`GFL`; control `A` virtual admittance, `C` current references; limiter `C` circular, `F` FRT current priority; filter `LC` (terminal capacitor in the device) or `L` (no capacitor).

(cmp-sources)=
## 2. Sources and passive devices

(cmp-Infbus)=
### Infbus

*Voltage port* · `components/infbus.py`

Ideal voltage source rotating at $\omega_0$ whose amplitude follows $A_{ref}$ with a first-order lag. It is the slack bus of the power flow; a fault is applied by changing $A_{ref}(t)$.

#### States, references, parameters

:::{list-table}
:header-rows: 1

* - Symbol
  - Code
  - Default
  - Meaning
* - $\theta$
  - `theta`
  - –
  - source angle (frame angle)
* - $A$
  - `A`
  - –
  - voltage amplitude
* - $A_{ref}$
  - `Aref`
  - reference
  - amplitude reference
* - $\omega_0$
  - `w0`
  - 1.0
  - source frequency
* - $k_A$
  - `kA`
  - 100
  - amplitude tracking gain
:::

#### Equations

$$
\dot\theta=\omega_0,\qquad \dot A = k_A\,(A_{ref}-A),\qquad \mathbf v_{\alpha\beta}=A\,\mathbf e(\theta).
$$

#### Power flow and initialization

$$
V-A_{ref}=0,\qquad \theta=0 \quad\text{(slack)};\qquad x_0=[\theta,\;V].
$$

Only the amplitude equation is an initialization residual. `convertToAnotherBase` does nothing: the source is always on the system base.

(cmp-ZeroCurrent)=
### ZeroCurrent

*Current port* · `components/zero_current.py`

Closes a bus that has only a voltage device. No states.

$$
\mathbf i_{\alpha\beta}=\mathbf 0,\qquad P=0,\quad Q=0.
$$

(cmp-Capacitor)=
### Capacitor

*Voltage port* · `components/capacitor.py`

Shunt capacitor with parallel conductance, written in its own frame rotating at $\omega$. Used as the line-charging capacitance of a bus, or as the terminal capacitor of a GFL.

#### States and parameters

:::{list-table}
:header-rows: 1

* - Symbol
  - Code
  - Default
  - Meaning
* - $\mathbf v_C$
  - `vCd, vCq`
  - –
  - capacitor voltage (dq)
* - $\theta_C$
  - `thetaC`
  - –
  - frame angle
* - $C$
  - `C`
  - 0.068
  - capacitance (susceptance at nominal frequency)
* - $G_d$
  - `Gd`
  - 0
  - parallel conductance
* - $\omega$
  - `w`
  - 1.0
  - frame speed
:::

#### Equations

$\mathbf i=\mathbf R(\theta_C)^{\top}\mathbf i_{\alpha\beta}$ is the current the capacitor delivers into the network:

$$
\dot{\mathbf v}_C=\frac{1}{C}\left(-\mathbf i-G_d\,\mathbf v_C\right)-\omega\mathbf J\mathbf v_C,\qquad
          \dot\theta_C=\omega,\qquad \mathbf v_{\alpha\beta}=\mathbf R(\theta_C)\,\mathbf v_C .
$$

#### Power flow and initialization

$$
P+G_dV^2=0,\qquad Q-\omega C V^2=0;\qquad \mathbf v_C=[V,0]^\top,\;\theta_C=\theta.
$$

Base conversion scales $C$ by $1/k_Z$ and $G_d$ by $k_Z$, with $k_Z=(V_{b,new}^2/S_{b,new})/(V_b^2/S_b)$.

(cmp-machines)=
## 3. Rotating machines

(cmp-SG)=
### SG — synchronous generator

*Current port* · `components/sg.py`

Eighth-order machine: stator and rotor flux linkages in the rotor $dq$ frame (one d-axis damper, one field winding, one q-axis damper), a first-order AVR with reactive droop, and a speed governor lumped with the damping.

#### States and references

:::{list-table}
:header-rows: 1

* - Symbol
  - Code
  - Meaning
* - $\psi_d,\psi_{kd},\psi_{fd}$
  - `psi_d, psi_kd, psi_fd`
  - d-axis stator, damper and field flux
* - $\psi_q,\psi_{kq}$
  - `psi_q, psi_kq1`
  - q-axis stator and damper flux
* - $v_r$
  - `vr`
  - AVR output
* - $\theta$
  - `theta`
  - rotor (frame) angle
* - $\omega$
  - `w`
  - rotor speed
* - $P_{ref}, V_{d,ref}, V_{q,ref}, \omega_{ref}$
  - `Pref, Vdref, Vqref, wref`
  - mechanical power, AVR voltage set point, (power flow only), speed reference
:::

#### Parameters

:::{list-table}
:header-rows: 1

* - Code
  - Default
  - Meaning
  - Code
  - Default
  - Meaning
* - `Rs`
  - 0.003
  - stator resistance
  - `Xl`
  - 0.15
  - stator leakage
* - `Rfd`
  - 0.000605
  - field resistance
  - `Xad`, `Xaq`
  - 1.66, 1.61
  - d/q magnetizing reactance
* - `R1d`
  - 0.0284
  - d damper resistance
  - `Xfd`
  - 0.1649
  - field leakage
* - `R1q`
  - 0.0650
  - q damper resistance
  - `Xf1d`
  - 0
  - field–damper mutual leakage
* - `Ka`, `Ta`
  - 120, 0.15
  - AVR gain and time constant
  - `X1d`, `X1q`
  - 0.1714, 0.1066
  - damper leakages
* - `Kq`
  - 0.1
  - reactive droop of the AVR
  - `Vf0`
  - 0
  - exciter output offset (in $E_{fd}$ units)
* - `M`
  - 2
  - inertia
  - `D`, `Kg`
  - 1, 100
  - damping and governor gain
* - `w0`
  - 1
  - initial speed
  - `Rsnub`, `Csnub`
  - 1e4, 1e-4
  - declared, not used in the equations
:::

The electrical defaults are Kundur's Example 4.1 machine (555 MVA, 60 Hz; *Power System Stability and Control*, Sec. 4.2), given as standard parameters in `KUNDUR_EXAMPLE_4_1` and converted with the classical definitions. The q axis uses the salient-pole form (one damper: $X_q, X''_q, T''_{q0}$). The solver integrates in per-unit time $\tau=\omega_b t$, so converting time constants in seconds to resistances needs the rated frequency: `SG(Vb, Sb, index, standard=None, frequency_Hz=60)`.

- `SG.standardToFundamental(Xd, Xq, Xl, Xdp, Xdpp, Xqpp, Td0p, Td0pp, Tq0pp, Ra, frequency_Hz)` returns the fundamental parameters (time constants in seconds).
- `sg.standardParameters(frequency_Hz)` returns the standard parameters implied by the current fundamental ones (Kundur Eqs. 4.41–4.42).
- `sg.setStandardParameters(frequency_Hz, H=None, **standard)` replaces some or all of them in place and, if `H` is given, sets $M = 2H\omega_b$.

#### Flux–current relation

$$
\mathbf X_d=\begin{bmatrix}X_l+X_{ad} & X_{ad} & X_{ad}\\ X_{ad} & X_{1d}+X_{ad} & X_{ad}+X_{f1d}\\ X_{ad} & X_{ad}+X_{f1d} & X_{fd}+X_{ad}\end{bmatrix},\qquad
          \mathbf X_q=\begin{bmatrix}X_l+X_{aq} & X_{aq}\\ X_{aq} & X_{aq}+X_{1q}\end{bmatrix},
$$

$$
\begin{bmatrix}i'_d\\ i_{kd}\\ i_{fd}\end{bmatrix}=\mathbf X_d^{-1}\begin{bmatrix}\psi_d\\ \psi_{kd}\\ \psi_{fd}\end{bmatrix},\qquad
          \begin{bmatrix}i'_q\\ i_{kq}\end{bmatrix}=\mathbf X_q^{-1}\begin{bmatrix}\psi_q\\ \psi_{kq}\end{bmatrix},\qquad
          i_d=-i'_d,\quad i_q=-i'_q .
$$

The inverses are stored as parameters `Gd11…Gd33` and `Gq11…Gq22` and are recomputed on a base change. $i_d, i_q$ are the stator currents injected into the bus.

#### Equations

With $\mathbf v=\mathbf R(\theta)^{\top}\mathbf v_{\alpha\beta}$, $P=\mathbf v\cdot\mathbf i$, $Q=v_q i_d-v_d i_q$:

$$
\begin{aligned}
            \dot\psi_d &= v_d+R_s i_d+\omega\psi_q, & \dot\psi_q &= v_q+R_s i_q-\omega\psi_d,\\
            \dot\psi_{kd} &= -R_{1d}\,i_{kd}, & \dot\psi_{kq} &= -R_{1q}\,i_{kq},\\
            \dot\psi_{fd} &= \tfrac{R_{fd}}{X_{ad}}(V_{f0}+v_r)-R_{fd}\,i_{fd}, & \dot v_r &= \frac{-v_r+K_a\left(V_{d,ref}-|\mathbf v|-K_qQ\right)}{T_a},\\
            \dot\theta &= \omega, & \dot\omega &= \frac{P_{ref}-P-(D+K_g)(\omega-\omega_{ref})}{M},
          \end{aligned}
$$

$$
\mathbf i_{\alpha\beta}=\mathbf R(\theta)\,[i_d,\;i_q]^\top .
$$

#### Power flow and initialization

The power flow is the equilibrium of the dynamic model at synchronous speed $\omega=\omega_0$. The governor ($\dot\omega=0$) fixes the active power. The AVR is proportional, so $\dot v_r=0$ and $\dot\psi_{fd}=0$ give $X_{ad}i_{fd}=V_{f0}+K_a e$ with a non-zero steady-state error $e$: the terminal voltage does not settle at $V_{d,ref}$.

$$
P-P_{ref}-(D+K_g)(\omega_{ref}-\omega_0)=0,\qquad
          X_{ad}\,i_{fd}(V,P,Q)-V_{f0}-K_a\left(V_{d,ref}-V-K_qQ\right)=0 .
$$

$i_{fd}(V,P,Q)$ is the steady-state field current from the classical phasor diagram with $X_q=X_l+X_{aq}$ at $\omega_0$. The two candidate load angles $\delta$ and $\delta+\pi$ differ only in the sign of every $dq$ quantity; the one with positive field current is taken. Initialization builds the fluxes from that solution and sets $v_r=X_{ad}i_{fd}-V_{f0}$, $\theta=\theta_{bus}-\delta$, $\omega=\omega_0$. Every equation except $\dot\theta$ is an initialization residual. $V_{q,ref}$ does not enter the model.

#### RMS model

$v_r$ and $\omega$ stay differential; every flux is algebraic. The damper currents then vanish and $X_{ad}i_{fd}=E_{fd}$, so the machine is the AVR-driven voltage $E_{fd}$ behind $X_d/X_q$, with the swing equation in $\omega$.

(cmp-gfm)=
## 4. Grid-forming converters

All GFMs share the synchronization law: a frame rotating at the frequency reference ($\dot\theta=\omega$, $\omega\equiv\omega_{ref}$) and an internal EMF $\mathbf E=V_{ref}\,\mathbf e(\delta)$ whose angle integrates the active-power error,

$$
\dot\delta = K_p\left(P_{ref}-P_{meas}\right),
$$

so $K_p$ is a P–f droop gain (frequency deviation $\dot\delta$ per unit power error). What differs is where $P_{meas}$ is measured, how the EMF is turned into a current, and how the current is limited. In the RMS model $\delta$ stays differential.

(cmp-GFM_A_C_LC)=
### GFM\_A\_C\_LC

*Voltage port* · `components/gfm_a_c_lc.py`

Virtual-admittance GFM: the EMF drives a virtual $R_v+jL_v$ branch whose current is the reference of an inner PI current loop on the filter inductor $L_f$; the reference is limited to a circle of radius $I_{max}$. The terminal capacitor $C$ is part of the device, so the output is its voltage.

#### States, references, parameters

:::{list-table}
:header-rows: 1

* - Symbol
  - Code
  - Default
  - Meaning
* - $\mathbf v_C$
  - `vCd, vCq`
  - –
  - capacitor (terminal) voltage
* - $\mathbf i_{ref}$
  - `irefd, irefq`
  - –
  - virtual-admittance current (unlimited reference)
* - $\mathbf i_L$
  - `iLd, iLq`
  - –
  - filter-inductor current
* - $\mathbf x_{cc}$
  - `xccd, xccq`
  - –
  - current-PI integrator
* - $\delta,\theta$
  - `delta, theta`
  - –
  - EMF angle in the frame; frame angle
* - $P_{ref},V_{ref},\omega$
  - `Pref, Vref, wref`
  - references
  - power set point, EMF magnitude, frequency
* - $C$
  - `C`
  - 0.068
  - terminal capacitor
* - $L_f$
  - `Lf`
  - 0.07
  - filter inductor
* - $R_v, L_v$
  - `Rv, Lv`
  - 0.1, 0.01
  - virtual impedance
* - $I_{max}$
  - `Imax`
  - 1.5
  - current-reference limit
* - $K_p$
  - `Kp`
  - 0.05
  - P–f droop gain
* - $K_{p,cc}, K_{i,cc}$
  - `Kp_cc, Ki_cc`
  - 20, 5
  - current PI gains
:::

#### Equations

$\mathbf i_g=\mathbf R(\theta)^{\top}\mathbf i_{\alpha\beta}$ is the current delivered to the network.

$$
\begin{aligned}
            L_v\,\dot{\mathbf i}_{ref} &= \mathbf E-\mathbf v_C-R_v\,\mathbf i_{ref}-\omega L_v\mathbf J\,\mathbf i_{ref} &&\text{virtual admittance}\\
            \boldsymbol\varepsilon &= \operatorname{sat}_{I_{max}}(\mathbf i_{ref})-\mathbf i_L,\qquad \dot{\mathbf x}_{cc}=\boldsymbol\varepsilon &&\text{limited current error}\\
            \mathbf v_{br} &= \tfrac12\mathbf v_C+\omega L_f\mathbf J\,\mathbf i_L+K_{p,cc}\boldsymbol\varepsilon+K_{i,cc}\mathbf x_{cc} &&\text{bridge voltage}\\
            L_f\,\dot{\mathbf i}_L &= \mathbf v_{br}-\mathbf v_C-\omega L_f\mathbf J\,\mathbf i_L = -\tfrac12\mathbf v_C+K_{p,cc}\boldsymbol\varepsilon+K_{i,cc}\mathbf x_{cc}\\
            C\,\dot{\mathbf v}_C &= \mathbf i_L-\mathbf i_g-\omega C\mathbf J\,\mathbf v_C\\
            \dot\delta &= K_p\left(P_{ref}-\mathbf v_C\cdot\mathbf i_L\right),\qquad \dot\theta=\omega,\qquad \mathbf v_{\alpha\beta}=\mathbf R(\theta)\,\mathbf v_C .
          \end{aligned}
$$

The bridge feeds forward half of the capacitor voltage and decouples the inductor's speed voltage; the current integrator supplies the other half in the steady state ($\mathbf x_{cc}=\mathbf v_C/(2K_{i,cc})$). The power is measured at the capacitor with the filter current.

#### Power flow and initialization

$$
\mathbf i_L=\frac{[P,-Q]^\top}{V}+\omega C\mathbf J\,[V,0]^\top,\quad
          \mathbf E_{ss}=[V,0]^\top+\left(R_v\mathbf I+\omega L_v\mathbf J\right)\mathbf i_L;\qquad
          P_{ref}-P=0,\quad |\mathbf E_{ss}|^2-V_{ref}^2=0 .
$$

Initial state: $\mathbf v_C=[V,0]^\top$, $\mathbf i_{ref}=\mathbf i_L$, $\mathbf x_{cc}=\mathbf v_C/(2K_{i,cc})$, $\delta=\angle\mathbf E_{ss}$, $\theta$ = bus angle.

(cmp-GFM_A_C_L)=
### GFM\_A\_C\_L

*Current port* · `components/gfm_a_c_l.py`

GFM\_A\_C\_LC without the terminal capacitor. Without a capacitor the converter cannot be a voltage port, so the grid branch $R_g+jL_g$ between the converter and the port bus is embedded in the device: the filter and grid currents are the same current $\mathbf i$, the grid inductor is eliminated exactly, and the converter-side PCC voltage is reconstructed.

:::{list-table}
:header-rows: 1

* - Symbol
  - Code
  - Default
  - Meaning
* - $\mathbf i_{ref}$
  - `irefd, irefq`
  - –
  - virtual-admittance current
* - $\mathbf i$
  - `iLd, iLq`
  - –
  - filter = grid current
* - $\mathbf x_{cc}$
  - `xccd, xccq`
  - –
  - current-PI integrator
* - $\delta,\theta$
  - `delta, theta`
  - –
  - EMF angle; frame angle
* - $L_f$
  - `Lf`
  - 0.07
  - filter inductor
* - $R_v, L_v$
  - `Rv, Lv`
  - 0.1, 0.01
  - virtual impedance
* - $R_g, L_g$
  - `Rg, Lg`
  - 0, 0
  - embedded grid branch
* - $I_{max}, K_p, K_{p,cc}, K_{i,cc}$
  - `Imax, Kp, Kp_cc, Ki_cc`
  - 1.5, 0.05, 20, 5
  - as GFM\_A\_C\_LC
:::

#### Equations

With the port voltage $\mathbf v_g=\mathbf R(\theta)^{\top}\mathbf v_{\alpha\beta}$, the bridge law of GFM\_A\_C\_LC ($\tfrac12$ feed-forward of the PCC voltage) and $\mathbf v_{pcc}=\mathbf d+L_g\dot{\mathbf i}$:

$$
\begin{aligned}
            \mathbf d &= \mathbf v_g+R_g\,\mathbf i+\omega L_g\mathbf J\,\mathbf i, \qquad
            \boldsymbol\varepsilon = \operatorname{sat}_{I_{max}}(\mathbf i_{ref})-\mathbf i,\qquad
            \mathbf u_{cc} = K_{p,cc}\boldsymbol\varepsilon+K_{i,cc}\mathbf x_{cc},\\
            \left(L_f+\tfrac12L_g\right)\dot{\mathbf i} &= \mathbf u_{cc}-\tfrac12\mathbf d,\qquad
            \mathbf v_{pcc} = \mathbf d+L_g\,\dot{\mathbf i},\qquad \dot{\mathbf x}_{cc}=\boldsymbol\varepsilon,\\
            L_v\,\dot{\mathbf i}_{ref} &= \mathbf E-\mathbf v_{pcc}-R_v\,\mathbf i_{ref}-\omega L_v\mathbf J\,\mathbf i_{ref},\\
            \dot\delta &= K_p\left(P_{ref}-\mathbf v_{pcc}\cdot\mathbf i\right),\qquad \dot\theta=\omega,\qquad \mathbf i_{\alpha\beta}=\mathbf R(\theta)\,\mathbf i .
          \end{aligned}
$$

(From $L_f\dot{\mathbf i}=-\tfrac12\mathbf v_{pcc}+\mathbf u_{cc}$ and $\mathbf v_{pcc}=\mathbf d+L_g\dot{\mathbf i}$.)

#### Power flow and initialization

$$
\mathbf i=\frac{[P,-Q]^\top}{V},\;
          \mathbf v_{pcc}=[V,0]^\top+(R_g\mathbf I+\omega L_g\mathbf J)\,\mathbf i,\;
          \mathbf E_{ss}=\mathbf v_{pcc}+(R_v\mathbf I+\omega L_v\mathbf J)\,\mathbf i;\quad
          P_{ref}-\mathbf v_{pcc}\cdot\mathbf i=0,\;|\mathbf E_{ss}|^2-V_{ref}^2=0 .
$$

Note that $P_{ref}$ is the power at the converter side of the embedded branch, not at the port bus. Initial state: $\mathbf i_{ref}=\mathbf i$, $\mathbf x_{cc}=\mathbf v_{pcc}/(2K_{i,cc})$, $\delta=\angle\mathbf E_{ss}$.

(cmp-gfl)=
## 5. Grid-following converter

(cmp-GFL_C_F_L)=
### GFL\_C\_F\_L — grid-following converter

*Current port* · `components/gfl_c_f_l.py`

The only grid-following converter in the library. It is fed directly with current references (no outer power or voltage loop) and has a PLL, an inner current PI, FRT/LVRT reactive-current priority and a filter inductor. A terminal capacitor, if needed, is a separate [Capacitor](#cmp-Capacitor) at the same bus. An optional grid impedance $R_g+jL_g$ can be embedded so the device can sit directly on an infinite bus. Everything is written in the PLL frame: $\theta$ is the PLL angle.

#### States, references, parameters

:::{list-table}
:header-rows: 1

* - Symbol
  - Code
  - Default
  - Meaning
* - $\mathbf i=[i_d,i_q]$
  - `id, iq`
  - state
  - filter current (PLL frame)
* - $\mathbf x_{cc}$
  - `xPIccd, xPIccq`
  - state
  - current-PI integrators
* - $\theta$
  - `theta`
  - state
  - PLL angle
* - $x_{PLL}$
  - `xPLL`
  - state
  - PLL integrator
* - $I_{d,ref}, I_{q,ref}$
  - `Idref, Iqref`
  - reference
  - current references
* - $L$
  - `L`
  - 0.07
  - filter inductor
* - $K_{p,cc}, K_{i,cc}$
  - `Kpcc, Kicc`
  - 0.1, 5
  - current PI
* - $R_g, L_g$
  - `Rg, Lg`
  - 0, 0
  - embedded grid impedance (0 = none)
* - $K_{p,PLL}, K_{i,PLL}$
  - `KpPll, KiPll`
  - 0.01, 0.5
  - PLL PI
* - $\omega_{min},\omega_{max}$
  - `wPllMin, wPllMax`
  - 0.8, 1.2
  - PLL frequency limits
* - $K_{aw}$
  - `KawPll`
  - 5
  - PLL anti-windup gain
* - $\omega_0$
  - `w0`
  - 1.0
  - nominal frequency
* - $I_{max}$
  - `Imax`
  - 1.5
  - current limit (circle radius)
* - $V_{th}$
  - `Vth`
  - 0.9
  - FRT voltage threshold
* - $K_1$
  - `K1`
  - 1.5
  - FRT reactive-current gain
* - $I_{d2}$
  - `Id2`
  - 1.2
  - active current allowed during FRT
:::

#### Equations

**Measured voltage.** The port voltage is rotated into the PLL frame. If a grid impedance is embedded, the converter-terminal voltage is reconstructed from it as a quasi-steady drop at $\omega_0$:

$$
\mathbf v=\mathbf R(\theta)^{\top}\mathbf v_{\alpha\beta}+R_g\,\mathbf i+\omega_0L_g\mathbf J\,\mathbf i .
$$

**PLL** with frequency limit and anti-windup:

$$
\omega_{raw}=\omega_0+K_{p,PLL}\,v_q+K_{i,PLL}\,x_{PLL},\qquad
          \omega=\operatorname{clip}(\omega_{raw},\omega_{min},\omega_{max}),\qquad
          \dot x_{PLL}=v_q+K_{aw}\left(\omega-\omega_{raw}\right),\qquad\dot\theta=\omega .
$$

**Current priority.** In normal operation the active current has priority and the reactive current takes what is left of the circle $I_{max}$. Below $V_{th}$ the reactive support current $-K_1(V_{th}-|\mathbf v|)$ has priority and the active current is capped at $I_{d2}$. The two are blended by a steep logistic switch $m$:

$$
\begin{aligned}
            \text{normal: }& i_d^{N}=\operatorname{clip}(I_{d,ref},0,I_{max}), &
              i_q^{N}&=\operatorname{clip}\!\left(I_{q,ref},-\ell_q,\ell_q\right),\;\ell_q=\sqrt{\operatorname{pos}\!\left(I_{max}^2-(i_d^{N})^2\right)},\
            \text{FRT: }& i_q^{F}=\operatorname{clip}\!\left(-K_1(V_{th}-|\mathbf v|),-I_{max},I_{max}\right), &
              i_d^{F}&=\operatorname{clip}\!\left(I_{d2},0,\ell_d\right),\;\ell_d=\sqrt{\operatorname{pos}\!\left(I_{max}^2-(i_q^{F})^2\right)},\
            &m=\tfrac12\left(1+\tanh\!\left(50\,(V_{th}-|\mathbf v|)\right)\right), &
              \mathbf i^{\star}&=(1-m)\,\mathbf i^{N}+m\,\mathbf i^{F}.
          \end{aligned}
$$

A negative $i_q$ injects reactive power ($Q=-v_d i_q$ when $v_q=0$).

**Current loop and filter.** The bridge voltage is $\mathbf v_{conv}=\mathbf v+\omega L\mathbf J\mathbf i+K_{p,cc}\boldsymbol\varepsilon+K_{i,cc}\mathbf x_{cc}$ with $\boldsymbol\varepsilon=\mathbf i^{\star}-\mathbf i$. The decoupling term cancels the rotation of the PLL frame, so the filter equation reduces to a PI-driven integrator:

$$
L\,\dot{\mathbf i}=\mathbf v_{conv}-\mathbf v-\omega L\mathbf J\,\mathbf i=K_{p,cc}\,\boldsymbol\varepsilon+K_{i,cc}\,\mathbf x_{cc},\qquad
          \dot{\mathbf x}_{cc}=\boldsymbol\varepsilon,\qquad \mathbf i_{\alpha\beta}=\mathbf R(\theta)\,\mathbf i .
$$

#### Power flow and initialization

$$
P-V\,I_{d,ref}=0,\qquad Q+V\,I_{q,ref}=0 .
$$

Initial state: $\mathbf i=[P,\,-Q]^\top/V$, $\mathbf x_{cc}=\mathbf 0$, $\theta=\theta_{bus}$, $x_{PLL}=0$.

#### RMS model

$x_{PLL}$ stays differential; the currents and current-PI integrators are algebraic.
