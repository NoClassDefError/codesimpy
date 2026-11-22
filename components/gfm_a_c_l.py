"""GFM_A_C_L: Virtual-admittance GFM without a PCC capacitor."""

import numpy as np
import sympy as sp

from ..core import Device


class GFM_A_C_L(Device):
    """GFM · A: virtual admittance + current PI · C: circular limit on the current reference · L: no terminal capacitor (current port; the port voltage is the PCC voltage)."""
    thetaIndices = (7,)
    angleIndices = (6,)  # delta
    rmsDynamicIndices = (6,)  # delta

    def __init__(self, Vb: float, Sb: float, index: str) -> None:
        super().__init__(
            Vb, Sb, "Current", index,
            {
                "Lf": 0.07, "Rv": 0.1, "Lv": 0.01,
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
        pccVoltageStationary = sp.Matrix(self.u)

        p = lambda name: self.p[name]["sym"]
        J = sp.Matrix([[0, -1], [1, 0]])
        rotation = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta)],
            [sp.sin(theta), sp.cos(theta)],
        ])

        # The port voltage is the PCC voltage. A grid branch behind a PCC without a
        # capacitor is handled by the network (RLGrid.localDAEreduction).
        pccVoltage = rotation.T * pccVoltageStationary

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
        # Converter voltage = controller output + half of the PCC voltage (feedforward).
        dcurrent = (controllerVoltage - pccVoltage / 2) / p("Lf")

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
        pccVoltage = sp.Matrix([V, 0])
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
        pccVoltage = np.array([V, 0.0])
        internalVoltage = (
            pccVoltage + self.p["Rv"]["num"] * current
            + w * self.p["Lv"]["num"] * J @ current
        )
        delta = np.arctan2(internalVoltage[1], internalVoltage[0])
        xcc = 0.5 * pccVoltage / self.p["Ki_cc"]["num"]
        return np.r_[current, current, xcc, delta, theta]

    def calculateSignals(self, time_s, x, u, uref):
        """Terminal (PCC) voltage and its power: the PCC may be a bus without a voltage device."""
        x, u = np.asarray(x), np.asarray(u)
        current, theta = x[:, 2:4], x[:, 7]
        cos, sin = np.cos(theta), np.sin(theta)
        vd = cos * u[:, 0] + sin * u[:, 1]
        vq = -sin * u[:, 0] + cos * u[:, 1]
        return {
            "pccVoltage": np.hypot(vd, vq),
            "activePower": vd * current[:, 0] + vq * current[:, 1],
            "reactivePower": vq * current[:, 0] - vd * current[:, 1],
        }
