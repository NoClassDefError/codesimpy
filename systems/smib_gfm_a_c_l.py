"""SMIB__GFM_A_C_L: no-capacitor GFM connected directly to an infinite bus."""

from typing import Any

import numpy as np

from ..components import GFM_A_C_L, Infbus
from ..core import System


class SMIB__GFM_A_C_L(System):
    """SMIB · GFM_A_C_L: virtual admittance + current PI, circular current-reference limit, no terminal
    capacitor (current port). The grid impedance Rg/Lg is a branch from the PCC to the infinite bus; the PCC has
    no voltage device and is reduced by RLGrid.localDAEreduction."""

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
        for name in ("Lf", "Rv", "Lv", "Imax", "Kp", "Kp_cc", "Ki_cc"):
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
            "Vdev": [None, source], "Idev": [vsg, None],
            "vsg": vsg, "source": source,
        }
        # PCC (current device only) -- Rg/Lg -- infinite bus.
        network = {
            "A": np.array([[1.0], [-1.0]]), "R": [p["Rg"]], "L": [p["Lg"]],
            "busNames": ["PCC", "Infinite bus"],
        }
        base = {
            "Vbase": p["Vbase"], "Sbase": p["Sbase"],
            "frequency_Hz": p["frequency_Hz"],
        }
        super().__init__(type(self).__name__, network, devices, base, refs)
        self.solverSettings = {
            "method": "RK45",
            "max_step": 0.1 / (2 * np.pi * self.base["frequency_Hz"]),
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
