---
title: Building a component and a system
---

How a device model (`Device`) and a study (`System`) are written, step by step, using the existing grid-forming converter `GFM_A_C_L` and its system `SMIB__GFM_A_C_L`.

(guide-overview)=
## 1. Overview

CodeSim models a power network as **devices** (converters, machines, sources, capacitors) connected by an **RL network**. You write two classes:

:::{list-table}
:header-rows: 1

* - Class
  - Base class
  - Contains
  - Registration
* - Component
  - `Device`
  - the equations of one device
  - none: a system imports and creates it
* - System
  - `System`
  - one study: devices on buses, network, references, parameters, output signals
  - the `@system` decorator
:::

Initialization, EMT and RMS simulation, small-signal and interaction analysis and the VS Code integration then work without further code.

(guide-concepts)=
## 2. Concepts

- **Per unit and time.** Each device writes its equations in per unit of its own base $(V_b, S_b)$. Derivatives are with respect to $\tau=\omega_b t$, $\omega_b=2\pi f$, so inductances and capacitances are per-unit reactances and susceptances. Times you pass in (simulation time, breakpoints, references) are in seconds.
- **Frames.** Device ports and the network use the stationary $\alpha\beta$ frame. Inside a device you work in its own rotating $dq$ frame, whose angle $\theta$ is a state ([component step 4](#guide-c-theta)). With $\bm R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$ and $\bm J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$, a port vector is rotated in as $\bm x_{dq}=\bm R(\theta)^{\top}\bm x_{\alpha\beta}$ and out as $\bm x_{\alpha\beta}=\bm R(\theta)\,\bm x_{dq}$; $\omega L\bm J\bm i$ is the speed voltage of an inductor in $dq$.
- **Buses and ports.** Bus $i$ has one **voltage port** `Vdev[i]`, which sets the bus voltage, and one **current port** `Idev[i]`, which injects a current. An unused current port is filled with `ZeroCurrent`; a bus with only a current device gets its line `Capacitor` as voltage port.

:::{list-table}
:header-rows: 1

* - Port type
  - Receives $\bm u$ ($\alpha\beta$)
  - Returns $\bm g$ ($\alpha\beta$)
  - Examples
* - `"Voltage"`
  - current it delivers
  - bus voltage
  - `Infbus`, `Capacitor`, `GFM_A_C_LC`
* - `"Current"`
  - bus voltage
  - current it injects
  - `GFM_A_C_L`, `GFL_C_F_L`, `SG`, `ZeroCurrent`
:::

The output $\bm g$ must be algebraic in the states, so a device whose terminal is an inductor is a current port. Every symbol gets the device index as suffix: state `delta` of the device `"vsg"` is `delta_vsg`.

(guide-component)=
## 3. Write a component

The example is `GFM_A_C_L` (`codesimpy/components/gfm_a_c_l.py`): a virtual-admittance grid-forming converter with a current PI loop, a circular current limit and a filter inductor, with the grid branch $R_g+jL_g$ embedded. The whole file is at the end of the section.

(guide-c-plan)=
### Step 1 — Plan the model

Without a terminal capacitor the converter's terminal is an inductor, so it is a **current port**. Its states, references and parameters:

:::{list-table}
:header-rows: 1

* - Math
  - Code
  - Kind
  - Meaning
* - $\bm i_u$
  - `irefd, irefq`
  - state 0–1
  - virtual-admittance current: the unlimited reference (the limited one is $\bm i_{ref}$, 3c)
* - $\bm i_L$
  - `iLd, iLq`
  - state 2–3
  - filter current = grid current
* - $\bm x_{cc}$
  - `xccd, xccq`
  - state 4–5
  - current-PI integrator
* - $\delta$
  - `delta`
  - state 6
  - internal-voltage angle inside the frame
* - $\theta$
  - `theta`
  - state 7
  - frame angle
* - $P_{ref}, V_{ref}, \omega$
  - `Pref, Vref, wref`
  - references
  - power set point, internal-voltage magnitude, frequency
* - $L_f, R_v, L_v$
  - `Lf, Rv, Lv`
  - parameters
  - filter inductor, virtual impedance
* - $R_g, L_g$
  - `Rg, Lg`
  - parameters
  - embedded grid branch
* - $I_{max}, K_p, K_{pc}, K_{ic}$
  - `Imax, Kp, Kp_cc, Ki_cc`
  - parameters
  - current limit, droop gain, current-PI gains
:::

The control structure (all vectors in the device $dq$ frame):

:::{figure} figures/gfm-a-c-l-control.png
:alt: Circuit and control diagram of GFM_A_C_L

Top: the bridge voltage $\bm v^{*}$ drives the filter $L_f$; the terminal voltage $\bm v_t$ connects through the grid branch $R_g+jL_g$ to the grid voltage $\bm v_g$. Without a terminal capacitor $\bm i_g=\bm i_L$. Bottom: the droop (APC) sets the angle $\delta$ of the internal voltage $\bm v_s$ of magnitude $V_{ref}$ ($V$ in the figure); the voltage difference $\bm e=\bm v_s-\bm v_t$ drives the virtual admittance $\bm Z_v^{-1}(s)$ to give $\bm i_u$, the circular limiter gives $\bm i_{ref}$, and the current loop gives $\bm v^{*}$.
:::

(guide-c-constructor)=
### Step 2 — Class and constructor

Subclass `Device` and pass the port type, the parameters with their defaults (per unit of the device base), the state names and the reference names. The state order chosen here is the order of `f`, of the initial state and of the indices in step 4.

```{code} python
:caption: class and constructor
class GFM_A_C_L(Device):
    """GFM · A: virtual admittance + current PI · C: circular limit on the current reference · L: no terminal capacitor (current port; the grid branch Rg/Lg is in the device)."""

    def __init__(self, Vb: float, Sb: float, index: str) -> None:
        super().__init__(
            Vb, Sb, "Current", index,
            {
                "Lf": 0.07, "Rv": 0.1, "Lv": 0.01,
                "Rg": 0.0, "Lg": 0.0,
                "Imax": 1.5, "Kp": 0.05,
                "Kp_cc": 20.0, "Ki_cc": 5.0,
            },
            ("irefd", "irefq", "iLd", "iLq", "xccd", "xccq",
             "delta", "theta"),
            ("Pref", "Vref", "wref"),
        )
```

`Device.__init__` creates the symbols, suffixed with the device index: `self.x` (states), `self.u` (port input, here the bus voltage $\bm v_{\alpha\beta}$), `self.uref` (references) and `self.p[name]["sym"]` (parameters; the value is in `self.p[name]["num"]`). It then calls `getStateAndOutputEquationsSym` (step 3) once and stores `f` and `g`.

(guide-c-equations)=
### Step 3 — State and output equations

`getStateAndOutputEquationsSym` returns `f` ($d\bm x/d\tau$, one row per state, in state order) and `g` (the port output in $\alpha\beta$). Write parameters as symbols so they can be changed without rebuilding, and use only differentiable functions (`sp.Abs`, `sp.sqrt`, `sp.tanh`; no `Piecewise`, `Min`/`Max` or `if`). The method is built in six parts.

#### 3a. Symbols and rotation

Unpack states and references and build $\bm J$ and $\bm R(\theta)$.

```{code} python
:caption: getStateAndOutputEquationsSym — symbols
def getStateAndOutputEquationsSym(self):
    iref = sp.Matrix(self.x[0:2])
    current = sp.Matrix(self.x[2:4])
    xcc = sp.Matrix(self.x[4:6])
    delta, theta = self.x[6:8]
    Pref, Vref, w = self.uref
    gridVoltageStationary = sp.Matrix(self.u)

    p = lambda name: self.p[name]["sym"]
    J = sp.Matrix([[0, -1], [1, 0]])
    rotation = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta), sp.cos(theta)],
    ])
```

#### 3b. Port voltage and the embedded grid branch

The port voltage (the infinite bus) is rotated into the frame. Filter and grid carry the same current $\bm i_L$, so the terminal voltage $\bm v_t$ follows from the grid branch:

$$
\begin{aligned}
\bm v_g &= \bm R(\theta)^{\top}\bm v_{\alpha\beta}\\
\bm v_t &= \bm v_g+R_g\,\bm i_L+\omega L_g\bm J\,\bm i_L+L_g\,\frac{d\bm i_L}{d\tau}
\end{aligned}
$$

The code computes the first three terms here (`gridSideDrop`); the $L_g\,d\bm i_L/d\tau$ term is added in 3d, once $d\bm i_L/d\tau$ is known.

```{code} python
:caption: getStateAndOutputEquationsSym — port voltage
# The external voltage port is the infinite bus. With no PCC
# capacitor, the filter and grid current are identical, so the grid
# inductor can be eliminated exactly and embedded in this current
# device.
gridVoltage = rotation.T * gridVoltageStationary
gridSideDrop = (
    gridVoltage + p("Rg") * current
    + w * p("Lg") * J * current
)
```

#### 3c. Circular current limit

The unlimited current $\bm i_u$ from the virtual admittance (3e) is scaled onto the circle $I_{max}$ when it is outside. The code writes the $\min$ with an absolute value so that it can be differentiated.

$$
\bm i_{ref}=\bm i_u\,\min\!\left(1,\;\frac{I_{max}}{|\bm i_u|}\right)
$$

```{code} python
:caption: getStateAndOutputEquationsSym — current limit
squaredMagnitude = iref.dot(iref)
denominator = sp.sqrt((
    squaredMagnitude + p("Imax")**2
    + sp.Abs(squaredMagnitude - p("Imax")**2)
) / 2)
limitedReference = p("Imax") * iref / denominator
```

#### 3d. Current loop and filter

The bridge voltage command feeds forward a part $K$ of the terminal voltage, decouples the filter's speed voltage and adds a PI on the current error:

$$
\begin{aligned}
\bm v^{*} &= K\,\bm v_t+\omega L_f\bm J\,\bm i_L+\left(K_{pc}+\frac{K_{ic}}{s}\right)\left(\bm i_{ref}-\bm i_L\right)\\
K &= 0.5
\end{aligned}
$$

The integral part is the state $\bm x_{cc}$ (`xccd, xccq`). The code puts $\bm v^{*}$ into the filter inductor and, because $\bm v_t$ contains $L_g\,d\bm i_L/d\tau$ (3b), solves for $d\bm i_L/d\tau$ (`dcurrent`) before completing $\bm v_t$.

```{code} python
:caption: getStateAndOutputEquationsSym — current loop
currentError = limitedReference - current
controllerVoltage = (
    p("Kp_cc") * currentError + p("Ki_cc") * xcc
)
dcurrent = (
    controllerVoltage - gridSideDrop / 2
) / (p("Lf") + p("Lg") / 2)
pccVoltage = gridSideDrop + p("Lg") * dcurrent
```

#### 3e. Droop and virtual admittance

The droop (APC) moves the angle $\delta$ of the internal voltage $\bm v_s$ with the power error. The voltage difference $\bm e$ drives the virtual admittance (AVC), whose current $\bm i_u$ is the unlimited reference:

$$
\begin{aligned}
\dot\delta &= K_p\left(P_{ref}-\bm v_t\cdot\bm i_L\right)\\
\bm v_s &= V_{ref}\begin{bmatrix}\cos\delta\\\sin\delta\end{bmatrix}\\
\bm e &= \bm v_s-\bm v_t\\
\bm i_u &= \bm Z_v^{-1}(s)\,\bm e,\qquad \bm Z_v(s)=R_v\bm I+L_v\left(s\bm I+\omega\bm J\right)
\end{aligned}
$$

The code writes $\bm Z_v^{-1}(s)$ as the state equation of $\bm i_u$ (`diref`).

```{code} python
:caption: getStateAndOutputEquationsSym — droop and virtual admittance
ddelta = p("Kp") * (
    Pref - pccVoltage.dot(current)
)
internalVoltage = Vref * sp.Matrix([
    sp.cos(delta), sp.sin(delta),
])
diref = (
    internalVoltage - pccVoltage - p("Rv") * iref
    - w * p("Lv") * J * iref
) / p("Lv")
```

#### 3f. Return f and g

The rows of `f` follow the state order of step 2; the last row is the frame angle. The output is the current rotated back to $\alpha\beta$.

$$
\begin{aligned}
\bm f &= \left[\frac{d\bm i_u}{d\tau},\;\frac{d\bm i_L}{d\tau},\;\frac{d\bm x_{cc}}{d\tau},\;\dot\delta,\;\dot\theta\right],\qquad
\frac{d\bm x_{cc}}{d\tau}=\bm i_{ref}-\bm i_L,\quad \dot\theta=\omega\\
\bm g &= \bm i_{\alpha\beta}=\bm R(\theta)\,\bm i_L
\end{aligned}
$$

```{code} python
:caption: getStateAndOutputEquationsSym — return
return (
    list(diref) + list(dcurrent) + list(currentError)
    + [ddelta, w],                    # f
    list(rotation * current),         # g
)
```

(guide-c-theta)=
### Step 4 — The frame angle θ and the RMS states

Two class attributes, given as positions in the state list:

```{code} python
:caption: class attributes
thetaIndices = (7,)
rmsDynamicIndices = (6,)  # delta
```

**What $\theta$ is.** $\theta$ is the angle of the device's own $dq$ frame measured in the stationary $\alpha\beta$ frame. Every vector inside the device ($\bm i_L$, $\bm i_u$, $\bm x_{cc}$, $\bm v_s$) is written in this frame, and $\bm R(\theta)$ connects it to the port. In `GFM_A_C_L` the frame turns at the frequency reference, $\dot\theta=\omega$, and the internal voltage sits at $\delta$ *inside* it, so its absolute angle is $\theta+\delta$ and its frequency $\omega+\dot\delta$. $\theta$ is the clock of the frame; $\delta$ is the load angle that the droop moves.

**Why it is special.** In steady state every other state is constant, but $\theta$ keeps growing ($\dot\theta=\omega\ne0$). So:

- **Initialization.** The states in `thetaIndices` have no steady-state equation. All other rows must satisfy $f=0$ and are solved; $\theta$ is held at the value from `initFromPowerFlow` (step 6), normally the bus angle so the frame is aligned with the bus voltage.
- **RMS model.** The RMS model is written in a common $DQ$ frame turning at 1 pu, and each $\theta$ is measured relative to it: its row becomes $\dot\theta=\omega-1$, which is zero in steady state at nominal frequency. $\theta$ always stays differential.
- **Absolute angle.** Only angle differences matter physically. One device, the infinite bus, fixes the reference angle in the power flow.

$\theta$ in each existing device:

:::{list-table}
:header-rows: 1

* - Device
  - $\theta$ state
  - $\dot\theta$
  - Meaning
  - Initial value
* - `Infbus`
  - `theta` (0)
  - $\omega_0=1$
  - angle of the grid voltage $A\,[\cos\theta,\sin\theta]$
  - bus angle (0: the slack fixes it)
* - `Capacitor`
  - `thetaC` (2)
  - $\omega=1$
  - a frame at nominal speed in which the capacitor voltage is written
  - bus angle, so $\bm v_C=[V,0]$
* - `GFM_A_C_L`, `GFM_A_C_LC`
  - `theta`
  - $\omega_{ref}$
  - control frame at the frequency reference; the internal voltage is at $\delta$ inside it
  - bus angle
* - `GFL_C_F_L`
  - `theta` (4)
  - $\omega_{PLL}$
  - PLL angle: the PLL turns the frame until $v_q=0$
  - bus angle
* - `SG`
  - `theta` (6)
  - $\omega$ (rotor speed, a state)
  - rotor angle: the frame is fixed to the rotor's $d$ axis
  - bus angle − load angle
* - `ZeroCurrent`
  - –
  - –
  - no states
  - –
:::

**RMS states.** `rmsDynamicIndices` lists the slow states that stay differential in the RMS model besides $\theta$: here $\delta$. The fast states (currents, integrators of the current loop) become algebraic, $0=f$.

(guide-c-powerflow)=
### Step 5 — Power-flow equations

Initialization first solves a power flow in the bus voltage $V$, the bus angle and the injected power $P, Q$. The device returns exactly two equations $=0$. In steady state $\bm i_L=[P,-Q]/V$ in a frame aligned with the bus voltage, so:

$$
\begin{aligned}
\bm v_t &= \begin{bmatrix}V\\0\end{bmatrix}+(R_g\bm I+\omega L_g\bm J)\,\bm i_L\\
\bm v_s &= \bm v_t+(R_v\bm I+\omega L_v\bm J)\,\bm i_L\\
0 &= P_{ref}-\bm v_t\cdot\bm i_L\\
0 &= |\bm v_s|^2-V_{ref}^2
\end{aligned}
$$

```{code} python
:caption: getPowerFlowEquations
def getPowerFlowEquations(self, V, theta, P, Q):
    del theta
    w = self.uref[2]
    current = sp.Matrix([P / V, -Q / V])
    J = sp.Matrix([[0, -1], [1, 0]])
    pccVoltage = (
        sp.Matrix([V, 0]) + self.p["Rg"]["num"] * current
        + w * self.p["Lg"]["num"] * J * current
    )
    internalVoltage = (
        pccVoltage + self.p["Rv"]["num"] * current
        + w * self.p["Lv"]["num"] * J * current
    )
    return [
        self.uref[0] - pccVoltage.dot(current),
        internalVoltage.dot(internalVoltage) - self.uref[1] ** 2,
    ]
```

References appear as symbols (`self.uref`; their values are substituted at the initialization time) and parameters as numbers (`self.p[name]["num"]`).

(guide-c-init)=
### Step 6 — Initial state

From the power-flow result, return a value for every state, in state order. The guess needs to be on the right branch; the exact steady state is then solved from $f=0$.

$$
\begin{aligned}
\bm i_u &= \bm i_L=\frac{[P,-Q]^\top}{V}, & \bm x_{cc} &= \frac{\bm v_t}{2K_{ic}}\\
\delta &= \angle\bm v_s, & \theta &= \theta_{bus}
\end{aligned}
$$

($\bm x_{cc}$ supplies the half of $\bm v_t$ that the bridge does not feed forward.)

```{code} python
:caption: initFromPowerFlow
def initFromPowerFlow(self, V, theta, P, Q):
    w = 1.0
    current = np.array([P / V, -Q / V])
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    pccVoltage = (
        np.array([V, 0.0]) + self.p["Rg"]["num"] * current
        + w * self.p["Lg"]["num"] * J @ current
    )
    internalVoltage = (
        pccVoltage + self.p["Rv"]["num"] * current
        + w * self.p["Lv"]["num"] * J @ current
    )
    xcc = 0.5 * pccVoltage / self.p["Ki_cc"]["num"]
    delta = np.arctan2(internalVoltage[1], internalVoltage[0])
    return np.r_[current, current, xcc, delta, theta]
```

:::{note}
**No registration.** A component is a plain `Device` subclass that a system imports and creates. Put your own in your workspace next to the system that uses it (the workspace folder is on the import path), or in `codesimpy/components/`.
:::

::::{dropdown} Full file: codesimpy/components/gfm_a_c_l.py
```{code} python
:caption: codesimpy/components/gfm_a_c_l.py
"""GFM_A_C_L: Virtual-admittance GFM without a PCC capacitor."""

import numpy as np
import sympy as sp

from ..core import Device


class GFM_A_C_L(Device):
    """GFM · A: virtual admittance + current PI · C: circular limit on the current reference · L: no terminal capacitor (current port; the grid branch Rg/Lg is in the device)."""
    thetaIndices = (7,)
    rmsDynamicIndices = (6,)  # delta

    def __init__(self, Vb: float, Sb: float, index: str) -> None:
        super().__init__(
            Vb, Sb, "Current", index,
            {
                "Lf": 0.07, "Rv": 0.1, "Lv": 0.01,
                "Rg": 0.0, "Lg": 0.0,
                "Imax": 1.5, "Kp": 0.05,
                "Kp_cc": 20.0, "Ki_cc": 5.0,
            },
            ("irefd", "irefq", "iLd", "iLq", "xccd", "xccq",
             "delta", "theta"),
            ("Pref", "Vref", "wref"),
        )

    def getStateAndOutputEquationsSym(self):
        iref = sp.Matrix(self.x[0:2])
        current = sp.Matrix(self.x[2:4])
        xcc = sp.Matrix(self.x[4:6])
        delta, theta = self.x[6:8]
        Pref, Vref, w = self.uref
        gridVoltageStationary = sp.Matrix(self.u)

        p = lambda name: self.p[name]["sym"]
        J = sp.Matrix([[0, -1], [1, 0]])
        rotation = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta)],
            [sp.sin(theta), sp.cos(theta)],
        ])

        # The external voltage port is the infinite bus. With no PCC
        # capacitor, the filter and grid current are identical, so the grid
        # inductor can be eliminated exactly and embedded in this current
        # device.
        gridVoltage = rotation.T * gridVoltageStationary
        gridSideDrop = (
            gridVoltage + p("Rg") * current
            + w * p("Lg") * J * current
        )

        squaredMagnitude = iref.dot(iref)
        denominator = sp.sqrt((
            squaredMagnitude + p("Imax")**2
            + sp.Abs(squaredMagnitude - p("Imax")**2)
        ) / 2)
        limitedReference = p("Imax") * iref / denominator
        currentError = limitedReference - current
        controllerVoltage = (
            p("Kp_cc") * currentError + p("Ki_cc") * xcc
        )
        dcurrent = (
            controllerVoltage - gridSideDrop / 2
        ) / (p("Lf") + p("Lg") / 2)
        pccVoltage = gridSideDrop + p("Lg") * dcurrent

        internalVoltage = Vref * sp.Matrix([
            sp.cos(delta), sp.sin(delta),
        ])
        diref = (
            internalVoltage - pccVoltage - p("Rv") * iref
            - w * p("Lv") * J * iref
        ) / p("Lv")
        ddelta = p("Kp") * (
            Pref - pccVoltage.dot(current)
        )

        outputCurrent = rotation * current
        return (
            list(diref) + list(dcurrent) + list(currentError)
            + [ddelta, w],
            list(outputCurrent),
        )

    def getPowerFlowEquations(self, V, theta, P, Q):
        del theta
        w = self.uref[2]
        current = sp.Matrix([P / V, -Q / V])
        J = sp.Matrix([[0, -1], [1, 0]])
        pccVoltage = (
            sp.Matrix([V, 0]) + self.p["Rg"]["num"] * current
            + w * self.p["Lg"]["num"] * J * current
        )
        internalVoltage = (
            pccVoltage + self.p["Rv"]["num"] * current
            + w * self.p["Lv"]["num"] * J * current
        )
        return [
            self.uref[0] - pccVoltage.dot(current),
            internalVoltage.dot(internalVoltage) - self.uref[1] ** 2,
        ]

    def initFromPowerFlow(self, V, theta, P, Q):
        w = 1.0
        current = np.array([P / V, -Q / V])
        J = np.array([[0.0, -1.0], [1.0, 0.0]])
        pccVoltage = (
            np.array([V, 0.0]) + self.p["Rg"]["num"] * current
            + w * self.p["Lg"]["num"] * J @ current
        )
        internalVoltage = (
            pccVoltage + self.p["Rv"]["num"] * current
            + w * self.p["Lv"]["num"] * J @ current
        )
        delta = np.arctan2(internalVoltage[1], internalVoltage[0])
        xcc = 0.5 * pccVoltage / self.p["Ki_cc"]["num"]
        return np.r_[current, current, xcc, delta, theta]
```
::::

(guide-system)=
## 4. Write a system

The example is `SMIB__GFM_A_C_L` (`codesimpy/systems/smib_gfm_a_c_l.py`): `GFM_A_C_L` on an infinite bus. The grid branch is inside the device, so the network has a single bus and no branch.

:::{figure} figures/smib-gfm-a-c-l-circuit.svg
:alt: Circuit of SMIB__GFM_A_C_L

One bus: the infinite bus is its voltage port, the converter its current port. The filter $L_f$ and the grid branch $R_g+jL_g$ are inside `GFM_A_C_L`; the terminal voltage $\bm v_t$ between them is reconstructed from the grid branch (component step 3b).
:::

(guide-s-file)=
### Step 1 — File, imports and the `@system` annotation

Your own system lives in a `.py` file in your VS Code workspace. Mark the class with `@system`: the backend imports every workspace file that uses it and lists the class in the system dropdown.

```{code} python
:caption: my_smib.py (in your workspace)
"""My SMIB: GFM_A_C_L on an infinite bus, the grid impedance embedded in the device."""

from typing import Any

import numpy as np

from codesimpy import System, system
from codesimpy.components import GFM_A_C_L, Infbus

@system(id="MySMIB_GFM", label="My SMIB · GFM_A_C_L", category="SMIB")
class MySMIB_GFM(System):
    """GFM_A_C_L on an infinite bus, the grid impedance embedded in the device."""
    ...
```

:::{list-table}
:header-rows: 1

* - Argument
  - Meaning
  - Default
* - `id`
  - unique name of the system among all systems
  - class name
* - `label`
  - name shown in the dropdown
  - `id`
* - `category`
  - group in the dropdown
  - none
* - `description`
  - one-line description
  - first line of the class docstring
:::

- The decorator is not inherited: a subclass of a decorated system needs its own `@system`.
- Keep the first line of the docstring a complete sentence; it becomes the description.
- Built-in systems inside the `codesimpy` package (like `SMIB__GFM_A_C_L`) are listed in `__all__` of `codesimpy/systems/__init__.py` instead, and use relative imports (`from ..components import …`).

The steps below show the body of `SMIB__GFM_A_C_L`; it is the same in your file.

(guide-s-params)=
### Step 2 — Default parameters

`defaultParameters` returns every parameter. VS Code passes the values a user edits as `overrides`, so derived values are computed after them. The grid impedance follows from the short-circuit ratio and $\rho=R/X$:

$$
L_g=\frac{1}{\mathrm{SCR}\,\sqrt{1+\rho^2}},\qquad R_g=\rho\,L_g\qquad\text{(pu)}
$$

```{code} python
:caption: defaultParameters
@staticmethod
def defaultParameters(overrides=None):
    p = {
        "Vbase": 1, "Sbase": 8100, "frequency_Hz": 50,
        "SCR": 2, "rho": 0.1,
        "Pref": 1, "Vref": 1.1, "wref": 1,
        "Lf": 0.07, "Rv": 0.05, "Lv": 0.01,
        "Imax": 1.5, "Kp": 0.05, "Kp_cc": 20, "Ki_cc": 5,
        "clearMode": "time", "clearAngle_rad": -np.pi / 3,
        "faultStart_s": 0.2, "faultComplete_s": 0.201,
        "minimumFaultDuration_s": 0.5, "recoveryStart_s": 1,
        "recoveryComplete_s": 1.001, "faultVoltage": 0.5,
        "validationTime_s": 2, "outputStep_s": 1e-4,
        "residualTolerance": 1e-4, "clearingState": "delta",
    }
    p.update(overrides or {})
    p["wBase"] = 2 * np.pi * p["frequency_Hz"]
    p["Lg"] = 1 / (p["SCR"] * np.sqrt(1 + p["rho"]**2))
    p["Rg"] = p["rho"] * p["Lg"]
    p["breakpoints_s"] = [p["faultStart_s"], p["faultComplete_s"]]
    p["breakpoints_s"] += (
        [p["recoveryStart_s"], p["recoveryComplete_s"]]
        if p["clearMode"] == "time"
        else [p["faultStart_s"] + p["minimumFaultDuration_s"]]
    )
    return p
```

- VS Code needs `validationTime_s` (end time), `outputStep_s`, `residualTolerance` and `breakpoints_s`.
- `breakpoints_s` are the times where a reference changes; the solver restarts there. With angle-based clearing the recovery time is not known in advance, so only the earliest clearing time is listed.

(guide-s-devices)=
### Step 3 — Create the devices

Create each device with a unique index (`"vsg"`, `"grid"`) and set its system-specific parameters through `device.p[name]["num"]` before the model is built.

```{code} python
:caption: __init__ — devices
def __init__(self, p: dict[str, Any]) -> None:
    vsg = GFM_A_C_L(p["Vbase"], p["Sbase"], "vsg")
    for name in ("Lf", "Rv", "Lv", "Rg", "Lg", "Imax", "Kp", "Kp_cc", "Ki_cc"):
        vsg.p[name]["num"] = p[name]

    source = Infbus(p["Vbase"], p["Sbase"], "grid")
```

(guide-s-references)=
### Step 4 — References

Every device reference gets a value, keyed by its symbol name `str(device.uref[k])`: a number or a function of time in seconds. The infinite-bus magnitude $A_{ref}(t)$ is the fault profile: a ramp down to $V_f$ at the fault and back up at the clearing time $t_c$ (`self.eventTime_s`):

$$
\begin{aligned}
A_{ref}(t) = 1 &- (1-V_f)\operatorname{clip}\!\left(\frac{t-t_{f}}{T_{fall}},0,1\right)\\
&+ (1-V_f)\operatorname{clip}\!\left(\frac{t-t_{c}}{T_{rise}},0,1\right)
\end{aligned}
$$

```{code} python
:caption: __init__ — references
# Grid voltage ramps down at the fault and back up at clearing
# (self.eventTime_s, which an angle-based clearing event sets later).
fallDuration = max(p["faultComplete_s"] - p["faultStart_s"], np.finfo(float).eps)
riseDuration = max(p["recoveryComplete_s"] - p["recoveryStart_s"], np.finfo(float).eps)
refs = {
    str(vsg.uref[0]): p["Pref"],
    str(vsg.uref[1]): p["Vref"],
    str(vsg.uref[2]): p["wref"],
    str(source.uref[0]): lambda t: float(
        1 - (1 - p["faultVoltage"]) * np.clip((t - p["faultStart_s"]) / fallDuration, 0, 1)
        + (1 - p["faultVoltage"]) * np.clip((t - self.eventTime_s) / riseDuration, 0, 1)
    ),
}
```

(guide-s-buses)=
### Step 5 — Place the devices on buses

`Vdev` and `Idev` have one entry per bus. The single bus has the infinite bus as voltage port and the converter as current port, so no `ZeroCurrent` is needed. Extra keys are handles for your own code.

```{code} python
:caption: __init__ — buses
devices = {
    "Vdev": [source], "Idev": [vsg],
    "vsg": vsg, "source": source,
}
```

(guide-s-network)=
### Step 6 — Network

`A` is the bus × branch incidence matrix ($+1$ where a branch leaves a bus, $-1$ where it enters); `R` and `L` have one value per branch. Each branch $k$ adds $L_k\,\dot{\bm i}_k=\bm v_{from}-\bm v_{to}-R_k\bm i_k$ in $\alpha\beta$. Here the only branch is inside the device, so `A` is $1\times0$.

```{code} python
:caption: __init__ — network
# There is one physical node and no explicit branch. Rg/Lg live in
# GFM_A_C_L, including the line-inductor transient, and reconstruct
# the converter-side PCC.
network = {
    "A": np.empty((1, 0)), "R": [], "L": [],
    "busNames": ["Infinite bus"],
}
```

(guide-s-build)=
### Step 7 — Build the model

`super().__init__` assembles the device and network equations and compiles them. Then set the solver options, `parameters` and `parameterNames` (shown in the VS Code parameter editor).

```{code} python
:caption: __init__ — build
base = {
    "Vbase": p["Vbase"], "Sbase": p["Sbase"],
    "frequency_Hz": p["frequency_Hz"],
}
super().__init__(type(self).__name__, network, devices, base, refs)
self.solverSettings = {
    "method": "RK45",
    "max_step": 5e-3 / (2 * np.pi * self.base["frequency_Hz"]),
    "rtol": 1e-3,
    "atol": 1e-5,
}
self.parameters = p
self.parameterNames = list(p)
```

(guide-s-events)=
### Step 8 — Fault clearing event (optional)

With `clearMode="angle"` the fault is cleared when $\delta$ reaches $\delta_{clear}$. `eventStartTime_s` arms the event; `eventHandler` returns a value whose zero crossing is the event, and the detected time is stored in `self.eventTime_s`, which the reference of step 4 uses as $t_c$:

$$
h(\bm x)=\sin\frac{\delta-\delta_{clear}}{2}
$$

```{code} python
:caption: __init__ — event, and eventHandler
self.clearingState = p["clearingState"]
self.eventName = "faultClearing"
self.eventTime_s = p["recoveryStart_s"]
if p["clearMode"] == "angle":
    self.eventTime_s = np.inf
    self.eventStartTime_s = (
        p["faultStart_s"] + p["minimumFaultDuration_s"]
    )
self.eventStateIndex = self.stateNames.index(
    f"{self.clearingState}{self.devices['vsg'].index}"
)


def eventHandler(self, _time_s: float, state: np.ndarray) -> float:
    return float(np.sin(
        (state[self.eventStateIndex] - self.parameters["clearAngle_rad"]) / 2
    ))
```

(guide-s-signals)=
### Step 9 — Output signals

`calculateSignals` returns the plotted signals. `busSignals` gives |V|, angle, P, Q and line currents; add your own. `state` has one row per time sample. Here the terminal voltage $\bm v_t$ is reconstructed from the states as in step 3 and the power is $P=\bm v_t\cdot\bm i_L$, $Q=v_{t,q}i_d-v_{t,d}i_q$.

```{code} python
:caption: calculateSignals (beginning)
def calculateSignals(self, time_s: Any, state: Any, net: dict[str, Any]):
    time_s, state = np.asarray(time_s), np.asarray(state)
    signals = self.busSignals(net, [0], 0)
    vsg = self.devices["vsg"]

    def states(*names):
        indices = [
            self.stateNames.index(f"{name}{vsg.index}") for name in names
        ]
        return state[:, indices]

    current = states("iLd", "iLq")
    reference = states("irefd", "irefq")
    theta = states("theta")[:, 0]
    delta = states("delta")[:, 0]
```

```{code} python
:caption: calculateSignals (end)
signals.update({
    "pccVoltage": np.hypot(vd, vq),
    "gridVoltage": np.abs(grid),
    "vd": vd,
    "vq": vq,
    "activePower": (
        vd * current[:, 0] + vq * current[:, 1]
    )[:, None],
    "reactivePower": (
        vq * current[:, 0] - vd * current[:, 1]
    )[:, None],
    "irefDQ": limitedReference,
    "filterCurrentMagnitude": np.hypot(
        current[:, 0], current[:, 1],
    ),
    "delta": (delta + np.pi) % (2 * np.pi) - np.pi,
    "dot_delta": np.zeros_like(delta),
})
if time_s.size > 1:
    signals["dot_delta"] = np.gradient(np.unwrap(delta), time_s)
return signals
```

::::{dropdown} Full file: codesimpy/systems/smib_gfm_a_c_l.py
```{code} python
:caption: codesimpy/systems/smib_gfm_a_c_l.py
"""SMIB__GFM_A_C_L: no-capacitor GFM connected directly to an infinite bus."""

from typing import Any

import numpy as np

from ..components import GFM_A_C_L, Infbus
from ..core import System


class SMIB__GFM_A_C_L(System):
    """SMIB · GFM_A_C_L: virtual admittance + current PI, circular current-reference limit, no terminal
    capacitor; the grid impedance is embedded in the device (current port)."""

    @staticmethod
    def defaultParameters(overrides=None):
        p = {
            "Vbase": 1, "Sbase": 8100, "frequency_Hz": 50,
            "SCR": 2, "rho": 0.1,
            "Pref": 1, "Vref": 1.1, "wref": 1,
            "Lf": 0.07, "Rv": 0.05, "Lv": 0.01,
            "Imax": 1.5, "Kp": 0.05, "Kp_cc": 20, "Ki_cc": 5,
            "clearMode": "time", "clearAngle_rad": -np.pi / 3,
            "faultStart_s": 0.2, "faultComplete_s": 0.201,
            "minimumFaultDuration_s": 0.5, "recoveryStart_s": 1,
            "recoveryComplete_s": 1.001, "faultVoltage": 0.5,
            "validationTime_s": 2, "outputStep_s": 1e-4,
            "residualTolerance": 1e-4, "clearingState": "delta",
        }
        p.update(overrides or {})
        p["wBase"] = 2 * np.pi * p["frequency_Hz"]
        p["Lg"] = 1 / (p["SCR"] * np.sqrt(1 + p["rho"]**2))
        p["Rg"] = p["rho"] * p["Lg"]
        p["breakpoints_s"] = [p["faultStart_s"], p["faultComplete_s"]]
        p["breakpoints_s"] += (
            [p["recoveryStart_s"], p["recoveryComplete_s"]]
            if p["clearMode"] == "time"
            else [p["faultStart_s"] + p["minimumFaultDuration_s"]]
        )
        return p

    def __init__(self, p: dict[str, Any]) -> None:
        vsg = GFM_A_C_L(p["Vbase"], p["Sbase"], "vsg")
        for name in ("Lf", "Rv", "Lv", "Rg", "Lg", "Imax", "Kp", "Kp_cc", "Ki_cc"):
            vsg.p[name]["num"] = p[name]

        source = Infbus(p["Vbase"], p["Sbase"], "grid")
        # Grid voltage ramps down at the fault and back up at clearing
        # (self.eventTime_s, which an angle-based clearing event sets later).
        fallDuration = max(p["faultComplete_s"] - p["faultStart_s"], np.finfo(float).eps)
        riseDuration = max(p["recoveryComplete_s"] - p["recoveryStart_s"], np.finfo(float).eps)
        refs = {
            str(vsg.uref[0]): p["Pref"],
            str(vsg.uref[1]): p["Vref"],
            str(vsg.uref[2]): p["wref"],
            str(source.uref[0]): lambda t: float(
                1 - (1 - p["faultVoltage"]) * np.clip((t - p["faultStart_s"]) / fallDuration, 0, 1)
                + (1 - p["faultVoltage"]) * np.clip((t - self.eventTime_s) / riseDuration, 0, 1)
            ),
        }
        devices = {
            "Vdev": [source], "Idev": [vsg],
            "vsg": vsg, "source": source,
        }
        # There is one physical node and no explicit branch. Rg/Lg live in
        # GFM_A_C_L, including the line-inductor transient, and reconstruct
        # the converter-side PCC.
        network = {
            "A": np.empty((1, 0)), "R": [], "L": [],
            "busNames": ["Infinite bus"],
        }
        base = {
            "Vbase": p["Vbase"], "Sbase": p["Sbase"],
            "frequency_Hz": p["frequency_Hz"],
        }
        super().__init__(type(self).__name__, network, devices, base, refs)
        self.solverSettings = {
            "method": "RK45",
            "max_step": 5e-3 / (2 * np.pi * self.base["frequency_Hz"]),
            "rtol": 1e-3,
            "atol": 1e-5,
        }
        self.parameters = p
        self.parameterNames = list(p)
        self.clearingState = p["clearingState"]
        self.eventName = "faultClearing"
        self.eventTime_s = p["recoveryStart_s"]
        if p["clearMode"] == "angle":
            self.eventTime_s = np.inf
            self.eventStartTime_s = (
                p["faultStart_s"] + p["minimumFaultDuration_s"]
            )
        self.eventStateIndex = self.stateNames.index(
            f"{self.clearingState}{self.devices['vsg'].index}"
        )

    def eventHandler(self, _time_s: float, state: np.ndarray) -> float:
        return float(np.sin(
            (state[self.eventStateIndex] - self.parameters["clearAngle_rad"]) / 2
        ))

    def calculateSignals(self, time_s: Any, state: Any, net: dict[str, Any]):
        time_s, state = np.asarray(time_s), np.asarray(state)
        signals = self.busSignals(net, [0], 0)
        vsg = self.devices["vsg"]

        def states(*names):
            indices = [
                self.stateNames.index(f"{name}{vsg.index}") for name in names
            ]
            return state[:, indices]

        current = states("iLd", "iLq")
        reference = states("irefd", "irefq")
        theta = states("theta")[:, 0]
        delta = states("delta")[:, 0]
        grid = net["busVoltage"][:, 0]
        gridD = grid.real * np.cos(theta) + grid.imag * np.sin(theta)
        gridQ = -grid.real * np.sin(theta) + grid.imag * np.cos(theta)
        wColumn = self.referenceNames.index(str(vsg.uref[2]))
        w = net["referenceTrace"][:, wColumn]
        gridSideD = (
            gridD + vsg.p["Rg"]["num"] * current[:, 0]
            - w * vsg.p["Lg"]["num"] * current[:, 1]
        )
        gridSideQ = (
            gridQ + vsg.p["Rg"]["num"] * current[:, 1]
            + w * vsg.p["Lg"]["num"] * current[:, 0]
        )

        referenceMagnitude = np.hypot(reference[:, 0], reference[:, 1])
        limitedReference = reference * np.minimum(
            1,
            vsg.p["Imax"]["num"]
            / np.maximum(referenceMagnitude, np.finfo(float).eps),
        )[:, None]
        controllerVoltage = (
            vsg.p["Kp_cc"]["num"] * (limitedReference - current)
            + vsg.p["Ki_cc"]["num"] * states("xccd", "xccq")
        )
        currentDerivative = (
            controllerVoltage
            - 0.5 * np.column_stack((gridSideD, gridSideQ))
        ) / (vsg.p["Lf"]["num"] + 0.5 * vsg.p["Lg"]["num"])
        vd = gridSideD + vsg.p["Lg"]["num"] * currentDerivative[:, 0]
        vq = gridSideQ + vsg.p["Lg"]["num"] * currentDerivative[:, 1]
        signals.update({
            "pccVoltage": np.hypot(vd, vq),
            "gridVoltage": np.abs(grid),
            "vd": vd,
            "vq": vq,
            "activePower": (
                vd * current[:, 0] + vq * current[:, 1]
            )[:, None],
            "reactivePower": (
                vq * current[:, 0] - vd * current[:, 1]
            )[:, None],
            "irefDQ": limitedReference,
            "filterCurrentMagnitude": np.hypot(
                current[:, 0], current[:, 1],
            ),
            "delta": (delta + np.pi) % (2 * np.pi) - np.pi,
            "dot_delta": np.zeros_like(delta),
        })
        if time_s.size > 1:
            signals["dot_delta"] = np.gradient(np.unwrap(delta), time_s)
        return signals
```
::::

(guide-run)=
## 5. Run it

### In VS Code

1. Restart the Python backend (power button in the topology toolbar); it reads the workspace files when it starts.
1. Select the system by its `label`, edit parameters, and run {kbd}`EMT`, {kbd}`RMS`, {kbd}`Small signal` or {kbd}`Interaction`. Files that fail to import are reported with their error.
1. Check the equations in Small signal → *Symbolic model*, one device at a time.

### From Python

```{code} python
:caption: session.py
import numpy as np

from codesimpy.systems import SMIB__GFM_A_C_L

p = SMIB__GFM_A_C_L.defaultParameters({"SCR": 2})
system = SMIB__GFM_A_C_L(p)
print(system.stateNames)

# Operating point: power flow -> initFromPowerFlow -> f = 0 for every non-theta state.
op = system.initialize(ResidualTolerance=p["residualTolerance"])
print("residual", op["initialization"]["residualInfNorm"])

# EMT simulation (time in seconds).
t = np.arange(0.0, p["validationTime_s"], p["outputStep_s"])
emt = system.simulate(t, InitialState=op["state"], Breakpoints_s=p["breakpoints_s"], saveSnapshot=False)

# Small-signal analysis and RMS simulation.
print("unstable poles:", system.smallSignal.analyze(op)["unstablePoleCount"])
rms = system.simulateRMS(t, InitialState=op["state"], Breakpoints_s=p["breakpoints_s"])
print(rms["differentialStateNames"])   # thetas + rmsDynamicIndices
```

(guide-troubleshooting)=
## 6. Troubleshooting

:::{list-table}
:header-rows: 1

* - Symptom
  - Likely cause and fix
* - `RuntimeError: initialization failed`
  - No power-flow solution for these set-points, or `initFromPowerFlow` starts on the wrong branch, or a state that is not in `thetaIndices` cannot reach $f=0$. Print `op["initialization"]["powerFlow"]` and compare `initialState` with the solution.
* - Power flow fails
  - A device returns other than two equations, there is no angle-fixing device, or a reference was used as a number instead of its symbol.
* - Wrong results, no error
  - Two devices share an index, or the state order differs between `f`, `initFromPowerFlow` and the index attributes.
* - `KeyError` while building
  - A device reference has no entry in the reference map.
* - `Jacobian contains unresolved symbolic derivatives`
  - A non-differentiable function (`Piecewise`, `Min`, `Max`, `sign`).
* - System missing in VS Code
  - No `@system` on the class, an import error in the file (shown in the error list), a duplicate `id`, or the backend was not restarted.
:::

For the equations of the existing devices and systems, see the [component reference](./components.md) and the [system reference](./systems.md).
