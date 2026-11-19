"""Eighth-order synchronous generator with AVR and governor.

Electrical parameters can be given as standard (datasheet) parameters and are
converted to the fundamental equivalent-circuit parameters the model uses,
following Kundur, *Power System Stability and Control*, Sec. 4.2.
"""

import numpy as np
import sympy as sp

from ..core import Device

# Kundur Example 4.1: 555 MVA, 0.9 pf, 60 Hz turbine generator, per unit on the
# machine base, time constants in seconds, classical definitions. The model has a
# single q-axis damper, so the q axis uses the salient-pole form (Eq. 4.42):
# X'q = 0.65 and T'q0 = 1.0 s from the example are not used.
KUNDUR_EXAMPLE_4_1 = {
    "Xd": 1.81, "Xq": 1.76, "Xl": 0.15, "Ra": 0.003,
    "Xdp": 0.3, "Xdpp": 0.23, "Xqpp": 0.25,
    "Td0p": 8.0, "Td0pp": 0.03, "Tq0pp": 0.07,
}
KUNDUR_FREQUENCY_HZ = 60.0

_FUNDAMENTAL_NAMES = ("Rs", "Xl", "Xad", "Xaq", "Xfd", "Xf1d", "X1d", "X1q", "Rfd", "R1d", "R1q")


class SG(Device):
    # RMS: every flux is fast. The damper currents vanish and Xad*ifd = Efd, so the
    # machine is the AVR-driven voltage Efd behind Xd/Xq, with the swing equation in w.
    thetaIndices = (6,)
    rmsDynamicIndices = (5, 7)  # vr, w

    def __init__(self, Vb, Sb, index, standard=None, frequency_Hz=KUNDUR_FREQUENCY_HZ):
        """``standard`` overrides entries of ``KUNDUR_EXAMPLE_4_1``; ``frequency_Hz``
        is the rated frequency used to turn time constants in seconds into the
        per-unit time (tau = wBase * t) the solver integrates in."""
        fundamental = self.standardToFundamental(**{**KUNDUR_EXAMPLE_4_1, **(standard or {})},
                                                 frequency_Hz=frequency_Hz)
        parameters = {
            **fundamental, "Rsnub": 1e4, "Csnub": 1e-4, "w0": 1.0,
            "Kq": 0.1, "Ka": 120.0, "Ta": 0.15, "Vf0": 0.0,
            "D": 1.0, "M": 2.0, "Kg": 100.0,
        }
        # Inverse-inductance entries are filled by _refreshInverseInductances below.
        parameters.update({f"Gd{row}{column}": 0.0 for row in range(1, 4) for column in range(1, 4)})
        parameters.update({f"Gq{row}{column}": 0.0 for row in range(1, 3) for column in range(1, 3)})
        super().__init__(Vb, Sb, "Current", index, parameters,
                         ("psi_d", "psi_kd", "psi_fd", "psi_q", "psi_kq1", "vr", "theta", "w"),
                         ("Pref", "Vdref", "Vqref", "wref"))
        self._refreshInverseInductances()

    @staticmethod
    def standardToFundamental(Xd, Xq, Xl, Xdp, Xdpp, Xqpp, Td0p, Td0pp, Tq0pp, Ra=0.0,
                              frequency_Hz=KUNDUR_FREQUENCY_HZ):
        """Convert standard parameters to fundamental ones (Kundur Sec. 4.2).

        Reactances are per unit; ``Td0p``, ``Td0pp``, ``Tq0pp`` are open-circuit
        time constants in seconds, read with the classical definitions (Eqs.
        4.23-4.33). The d axis has a field and one damper with equal mutual
        inductances (``Xf1d = 0``); the q axis has one damper (Eq. 4.42).
        Returns the per-unit values of the model's resistances and reactances.
        """
        if not Xd > Xdp > Xdpp > Xl:
            raise ValueError("need Xd > X'd > X''d > Xl")
        if not Xq > Xqpp > Xl:
            raise ValueError("need Xq > X''q > Xl")
        if min(Td0p, Td0pp, Tq0pp) <= 0:
            raise ValueError("time constants must be positive")
        w0 = 2 * np.pi * frequency_Hz
        Xad, Xaq = Xd - Xl, Xq - Xl
        transient, subtransient = Xdp - Xl, Xdpp - Xl     # Lad||Lfd and Lad||Lfd||L1d
        Xfd = Xad * transient / (Xad - transient)          # Eq. 4.29 solved for Lfd
        X1d = transient * subtransient / (transient - subtransient)  # Eq. 4.28 solved for L1d
        X1q = Xaq * (Xqpp - Xl) / (Xaq - (Xqpp - Xl))      # Eq. 4.42 solved for L1q
        return {
            "Rs": Ra, "Xl": Xl, "Xad": Xad, "Xaq": Xaq, "Xfd": Xfd, "Xf1d": 0.0,
            "X1d": X1d, "X1q": X1q,
            "Rfd": (Xad + Xfd) / (w0 * Td0p),              # T'd0 = T1
            "R1d": (X1d + transient) / (w0 * Td0pp),       # T''d0 = T3
            "R1q": (Xaq + X1q) / (w0 * Tq0pp),             # T''q0, Eq. 4.42
        }

    def standardParameters(self, frequency_Hz=KUNDUR_FREQUENCY_HZ):
        """Classical standard parameters implied by the current fundamental ones
        (Kundur Eqs. 4.41-4.42, which include the peripheral leakage ``Xf1d``)."""
        Rs, Xl, Xad, Xaq, Xfd, Xpl, X1d, X1q, Rfd, R1d, R1q = (
            self.p[name]["num"] for name in _FUNDAMENTAL_NAMES)
        w0 = 2 * np.pi * frequency_Hz
        Xfp = Xfd + Xpl
        subtransient = ((X1d * Xfd * Xad + X1d * Xpl * Xad + Xad * Xfd * Xpl)
                        / (Xad * Xfd + Xad * X1d + X1d * Xfd + X1d * Xpl + Xfd * Xpl))
        return {
            "Xd": Xl + Xad, "Xq": Xl + Xaq, "Xl": Xl, "Ra": Rs,
            "Xdp": Xl + Xad * Xfp / (Xad + Xfp), "Xdpp": Xl + subtransient,
            "Xqpp": Xl + Xaq * X1q / (Xaq + X1q),
            "Td0p": (Xad + Xfp) / Rfd / w0,
            "Td0pp": (X1d + Xfd * (Xad + Xpl) / (Xad + Xfp)) / R1d / w0,
            "Tq0pp": (Xaq + X1q) / R1q / w0,
        }

    def setStandardParameters(self, frequency_Hz=KUNDUR_FREQUENCY_HZ, H=None, **standard):
        """Replace the electrical parameters from standard ones given on this
        device's current base (see ``standardToFundamental``). Missing entries
        keep their current values. If the inertia constant ``H`` (s) is given,
        set ``M = 2 H wBase`` for the per-unit time the solver uses."""
        merged = {**self.standardParameters(frequency_Hz), **standard}
        for name, value in self.standardToFundamental(**merged, frequency_Hz=frequency_Hz).items():
            self.p[name]["num"] = value
        if H is not None:
            self.p["M"]["num"] = 2 * H * 2 * np.pi * frequency_Hz
        self._refreshInverseInductances()

    def _refreshInverseInductances(self):
        Xl, Xad, Xfd, Xf1d, X1d, X1q, Xaq = (
            self.p[name]["num"] for name in ("Xl", "Xad", "Xfd", "Xf1d", "X1d", "X1q", "Xaq"))
        Gd = np.linalg.inv(np.array([[Xl + Xad, Xad, Xad], [Xad, X1d + Xad, Xad + Xf1d],
                                     [Xad, Xad + Xf1d, Xfd + Xad]]))
        Gq = np.linalg.inv(np.array([[Xl + Xaq, Xaq], [Xaq, Xaq + X1q]]))
        for row in range(3):
            for column in range(3):
                self.p[f"Gd{row + 1}{column + 1}"]["num"] = Gd[row, column]
        for row in range(2):
            for column in range(2):
                self.p[f"Gq{row + 1}{column + 1}"]["num"] = Gq[row, column]
        self.parameterMap = self._parameter_map()

    def getStateAndOutputEquationsSym(self):
        psi_d, psi_kd, psi_fd, psi_q, psi_kq1, vr, theta, w = self.x
        Pref, Vdref, Vqref, wref = self.uref
        valpha, vbeta = self.u
        vd = valpha * sp.cos(theta) + vbeta * sp.sin(theta)
        vq = -valpha * sp.sin(theta) + vbeta * sp.cos(theta)
        Gd = sp.Matrix([[self.p[f"Gd{r}{c}"]["sym"] for c in range(1, 4)] for r in range(1, 4)])
        Gq = sp.Matrix([[self.p[f"Gq{r}{c}"]["sym"] for c in range(1, 3)] for r in range(1, 3)])
        idVector = Gd * sp.Matrix([psi_d, psi_kd, psi_fd])
        iqVector = Gq * sp.Matrix([psi_q, psi_kq1])
        id, iq = -idVector[0], -iqVector[0]
        P = vd * id + vq * iq
        Q = vq * id - vd * iq
        e = Vdref - sp.sqrt(vd**2 + vq**2) - self.p["Kq"]["sym"] * Q
        # Vf0 + vr is the exciter output Efd (1 pu gives 1 pu open-circuit voltage);
        # the field winding sees efd = Rfd / Xad * Efd (Kundur Ch. 3 per-unit system).
        vf = self.p["Rfd"]["sym"] / self.p["Xad"]["sym"] * (self.p["Vf0"]["sym"] + vr)
        dpsi_d = vd + self.p["Rs"]["sym"] * id + w * psi_q
        dpsi_kd = -self.p["R1d"]["sym"] * idVector[1]
        dpsi_fd = vf - self.p["Rfd"]["sym"] * idVector[2]
        dpsi_q = vq + self.p["Rs"]["sym"] * iq - w * psi_d
        dpsi_kq1 = -self.p["R1q"]["sym"] * iqVector[1]
        dvr = (-vr + self.p["Ka"]["sym"] * e) / self.p["Ta"]["sym"]
        dtheta = w
        dw = (-P + Pref - (self.p["D"]["sym"] + self.p["Kg"]["sym"]) * w
              + (self.p["D"]["sym"] + self.p["Kg"]["sym"]) * wref) / self.p["M"]["sym"]
        return [dpsi_d, dpsi_kd, dpsi_fd, dpsi_q, dpsi_kq1, dvr, dtheta, dw], [
            id * sp.cos(theta) - iq * sp.sin(theta),
            id * sp.sin(theta) + iq * sp.cos(theta),
        ]

    def getPowerFlowEquations(self, V, theta, P, Q):
        # Equilibrium of the dynamic model at the grid frequency w = w0:
        #   dw = 0  -> P = Pref + (D + Kg) * (wref - w0)
        #   dvr = 0 and dpsi_fd = 0 -> Xad * ifd = Vf0 + Ka * (Vdref - |v| - Kq * Q)
        # The AVR is proportional, so |v| does not settle at Vdref. Of the two
        # machine-angle candidates, the one with positive field current is taken.
        Pref, Vdref, _, wref = self.uref
        p = {name: entry["num"] for name, entry in self.p.items()}
        w, Rs = p["w0"], p["Rs"]
        Xd, Xq = p["Xl"] + p["Xad"], p["Xl"] + p["Xaq"]
        activeCurrent, reactiveCurrent = P / V, Q / V
        cosineCoefficient = V + Rs * activeCurrent + w * Xq * reactiveCurrent
        sineCoefficient = Rs * reactiveCurrent - w * Xq * activeCurrent
        delta = sp.atan2(-cosineCoefficient, sineCoefficient)
        vq = V * sp.sin(delta)
        id = activeCurrent * sp.cos(delta) + reactiveCurrent * sp.sin(delta)
        iq = activeCurrent * sp.sin(delta) - reactiveCurrent * sp.cos(delta)
        psiD = (vq + Rs * iq) / w
        fieldCurrent = sp.Abs((psiD + Xd * id) / p["Xad"])
        return [
            P - Pref - (p["D"] + p["Kg"]) * (wref - p["w0"]),
            p["Xad"] * fieldCurrent - p["Vf0"] - p["Ka"] * (Vdref - V - p["Kq"] * Q),
        ]

    def initFromPowerFlow(self, V, theta, P, Q):
        V = abs(V)
        p = {name: entry["num"] for name, entry in self.p.items()}
        w, Rs = p["w0"], p["Rs"]
        Xd, Xq = p["Xl"] + p["Xad"], p["Xl"] + p["Xaq"]
        activeCurrent, reactiveCurrent = P / V, Q / V
        cosineCoefficient = V + Rs * activeCurrent + w * Xq * reactiveCurrent
        sineCoefficient = Rs * reactiveCurrent - w * Xq * activeCurrent
        delta = np.arctan2(-cosineCoefficient, sineCoefficient)
        id = activeCurrent * np.cos(delta) + reactiveCurrent * np.sin(delta)
        iq = activeCurrent * np.sin(delta) - reactiveCurrent * np.cos(delta)
        psiD = (V * np.sin(delta) + Rs * iq) / w
        fieldCurrent = (psiD + Xd * id) / p["Xad"]
        if fieldCurrent < 0:
            # The angle delta + pi negates every dq quantity; take the positive-field one.
            delta, id, iq, psiD, fieldCurrent = delta + np.pi, -id, -iq, -psiD, -fieldCurrent
        psiQ = -Xq * iq
        psiKd = -p["Xad"] * id + (p["Xad"] + p["Xf1d"]) * fieldCurrent
        psiFd = -p["Xad"] * id + (p["Xfd"] + p["Xad"]) * fieldCurrent
        psiKq = -p["Xaq"] * iq
        regulatorState = p["Xad"] * fieldCurrent - p["Vf0"]
        return [psiD, psiKd, psiFd, psiQ, psiKq, regulatorState, theta - delta, w]

    def convertToAnotherBase(self, Vbnew, Sbnew):
        kZ = (Vbnew**2 / Sbnew) / (self.Vb**2 / self.Sb)
        for name in ("Rs", "Rfd", "R1d", "R1q", "Rsnub", "Xl", "Xad", "Xaq", "Xf1d", "Xfd", "X1d", "X1q"):
            self.p[name]["num"] *= kZ
        self._refreshInverseInductances()
        self.Vb, self.Sb = Vbnew, Sbnew
        self.Ib = Sbnew / (np.sqrt(3 / 2) * Vbnew)
