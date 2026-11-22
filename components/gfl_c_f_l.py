
"""Grid-following converter without outer loop."""

import numpy as np
import sympy as sp
from ..core import Device


class GFL_C_F_L(Device):
    """GFL · C: current references (no outer loop) + current PI, PLL · F: FRT/LVRT reactive-current priority · L: filter inductor only (a terminal capacitor, if any, is a separate Capacitor)."""
    thetaIndices = (4,)
    rmsDynamicIndices = (5,)  # xPLL

    def __init__(self, Vb, Sb, index):
        super().__init__(Vb, Sb, "Current", index, {
            "L": 0.07, "Kicc": 5.0, "Kpcc": 0.1,
            "Rg": 0.0, "Lg": 0.0,
            "KpPll": 0.01, "KiPll": 0.5,
            "wPllMin": 0.8, "wPllMax": 1.2,
            "KawPll": 5.0, "w0": 1.0,
            "K1": 1.5, "Imax": 1.5,
            "Id2": 1.2, "Vth": 0.9,
        }, (
            "id", "iq", "xPIccd", "xPIccq",
            "theta", "xPLL",
        ), ("Idref", "Iqref"))

    def getStateAndOutputEquationsSym(self):
        i = sp.Matrix(self.x[0:2])
        xcc = sp.Matrix(self.x[2:4])
        theta, xPLL = self.x[4:6]

        iref0 = sp.Matrix(self.uref)
        vabc = sp.Matrix(self.u)

        p = lambda name: self.p[name]["sym"]
        pos = lambda x: (x + sp.Abs(x)) / 2

        def clip(x, lower, upper):
            return lower + pos(x - lower) - pos(x - upper)

        J = sp.Matrix([[0, -1], [1, 0]])
        R = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta)],
            [sp.sin(theta),  sp.cos(theta)],
        ])

        # Reconstruct the converter terminal voltage from the infinite-bus
        # voltage and the internal, quasi-steady grid-impedance drop.
        vGrid = R.T * vabc
        v = vGrid + p("Rg") * i + p("w0") * p("Lg") * J * i
        vd, vq = v
        vmag = sp.sqrt(v.dot(v))

        # PLL
        wRaw = p("w0") + p("KpPll") * vq + p("KiPll") * xPLL
        w = clip(wRaw, p("wPllMin"), p("wPllMax"))
        dxPLL = vq + p("KawPll") * (w - wRaw)

        # Normal current priority
        Imax = p("Imax")
        idNormal = clip(iref0[0], 0, Imax)
        iqLimit = sp.sqrt(pos(Imax**2 - idNormal**2))
        iqNormal = clip(iref0[1], -iqLimit, iqLimit)
        iNormal = sp.Matrix([idNormal, iqNormal])

        # LVRT reactive-current priority
        iqLvrt = clip(
            -p("K1") * (p("Vth") - vmag), -Imax, Imax
        )
        idLimit = sp.sqrt(pos(Imax**2 - iqLvrt**2))
        idLvrt = clip(p("Id2"), 0, idLimit)
        iLvrt = sp.Matrix([idLvrt, iqLvrt])

        lvrtMode = (1 + sp.tanh(50 * (p("Vth") - vmag))) / 2

        iref = (1 - lvrtMode) * iNormal + lvrtMode * iLvrt

        # Inner current PI controller
        ei = iref - i
        L = p("L")

        # Converter voltage command
        vconv = (
            v + w * L * J * i
            + p("Kpcc") * ei
            + p("Kicc") * xcc
        )

        # Filter-inductor dynamics
        di = (vconv - v) / L - w * J * i

        # Current PI integrator
        dxcc = ei

        # Stationary-frame output current
        iout = R * i

        return (
            list(di) + list(dxcc)
            + [w, dxPLL],
            list(iout)
        )

    def getPowerFlowEquations(self, V, theta, P, Q):
        return [
            P - V * self.uref[0],
            Q + V * self.uref[1],
        ]

    def initFromPowerFlow(self, V, theta, P, Q):
        i = sp.Matrix([P / V, -Q / V])
        return (
            list(i) + [0.0, 0.0]
            + [theta, 0.0]
        )

    def calculateSignals(self, time_s, x, u, uref):
        """Terminal (PCC) voltage behind the internal Rg/Lg, its power, LVRT mode and PLL frequency."""
        p = lambda name: self.p[name]["num"]
        x, u = np.asarray(x), np.asarray(u)
        current, theta, xPLL = x[:, 0:2], x[:, 4], x[:, 5]
        cos, sin = np.cos(theta), np.sin(theta)
        vd = cos * u[:, 0] + sin * u[:, 1] + p("Rg") * current[:, 0] - p("w0") * p("Lg") * current[:, 1]
        vq = -sin * u[:, 0] + cos * u[:, 1] + p("Rg") * current[:, 1] + p("w0") * p("Lg") * current[:, 0]
        magnitude = np.hypot(vd, vq)
        return {
            "pccVoltage": magnitude,
            "activePower": vd * current[:, 0] + vq * current[:, 1],
            "reactivePower": vq * current[:, 0] - vd * current[:, 1],
            "lvrtMode": 0.5 * (1 + np.tanh(50 * (p("Vth") - magnitude))),
            "pllFrequency": np.clip(p("w0") + p("KpPll") * vq + p("KiPll") * xPLL, p("wPllMin"), p("wPllMax")),
        }
