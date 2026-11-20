"""SMIB__SG: Synchronous generator connected to an infinite bus."""

import numpy as np

from ..components import Capacitor, Infbus, SG, ZeroCurrent
from ..core import System


class SMIB__SG(System):
    """SMIB · SG: synchronous generator with a terminal capacitor on an infinite bus."""
    @staticmethod
    def defaultParameters(overrides=None):
        p = {
            "Vbase": 1, "Sbase": 8100, "frequency_Hz": 50, "SCR": 2,
            "rho": 0.25, "Ka": 0.5, "Vf0": 1.0, "Vdref": 1.2812,
            "validationTime_s": 0.4, "faultStart_s": 0.10,
            "faultComplete_s": 0.101, "recoveryStart_s": 0.20,
            "recoveryComplete_s": 0.201, "faultVoltage": 0.2,
            "outputStep_s": 2e-3, "residualTolerance": 1e-7,
        }
        p.update(overrides or {})
        p["wBase"] = 2 * np.pi * p["frequency_Hz"]
        p["Lg"] = 1 / (p["SCR"] * np.sqrt(1 + p["rho"]**2))
        p["Rg"] = p["rho"] * p["Lg"]
        p["breakpoints_s"] = [p[name] for name in (
            "faultStart_s", "faultComplete_s", "recoveryStart_s", "recoveryComplete_s",
        )]
        p.setdefault("gridVoltage", lambda t: float(np.interp(
            t, [0, *p["breakpoints_s"], p["validationTime_s"]],
            [1, 1, p["faultVoltage"], p["faultVoltage"], 1, 1], right=1,
        )))
        return p

    def __init__(self, p):
        sg = SG(p["Vbase"], p["Sbase"], "1", frequency_Hz=p["frequency_Hz"])
        sg.p["Ka"]["num"], sg.p["Vf0"]["num"] = p["Ka"], p["Vf0"]
        capacitor = Capacitor(p["Vbase"], p["Sbase"], "2")
        source = Infbus(p["Vbase"], p["Sbase"], "4")
        devices = {"Vdev": [capacitor, source],
                   "Idev": [sg, ZeroCurrent(p["Vbase"], p["Sbase"], "3")],
                   "sg": sg, "capacitor": capacitor, "source": source}
        refs = {str(sg.uref[0]): 0.0, str(sg.uref[1]): p["Vdref"], str(sg.uref[2]): 0.0,
                str(sg.uref[3]): 1.0, str(source.uref[0]): p["gridVoltage"]}
        network = {"A": np.array([[1.0], [-1.0]]), "R": [p["Rg"]], "L": [p["Lg"]],
                   "busNames": ["SG PCC", "Infinite bus"]}
        super().__init__(type(self).__name__, network, devices,
                         {"Vbase": p["Vbase"], "Sbase": p["Sbase"], "frequency_Hz": p["frequency_Hz"]}, refs)
        self.solverSettings.update({"max_step": 0.1 / (2 * np.pi * p["frequency_Hz"]), "rtol": 1e-4, "atol": 1e-6})
        self.parameters, self.parameterNames = p, list(p)

    def calculateSignals(self, time_s, state, networkSignals):
        signals = self.busSignals(networkSignals, np.arange(self.rlGrid.n), self.rlGrid.n - 1)
        idx = [self.stateNames.index(str(symbol)) for symbol in self.devices["sg"].x]
        signals["lineCurrentMagnitude"] = np.abs(networkSignals["branchCurrent"])
        signals["sgSpeed"] = np.asarray(state)[:, idx[7]]
        return signals
