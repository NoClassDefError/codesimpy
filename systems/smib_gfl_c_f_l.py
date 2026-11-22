"""SMIB__GFL_C_F_L: GFL low-voltage ride-through benchmark."""

import numpy as np

from ..components import GFL_C_F_L, Infbus
from ..core import System


class SMIB__GFL_C_F_L(System):
    """SMIB · GFL_C_F_L: GFL with current references (no outer loop), PLL and FRT/LVRT current priority, without
    terminal capacitor (the grid impedance is embedded in the device). LVRT and post-fault voltage studies."""
    @staticmethod
    def defaultParameters(overrides=None):
        p = {
            "SCR": 2, "rho": 0.25, "Pref": 1, "Qref": -0.1,
            "K1": 1, "Id2": 1.0,
            "frequency_Hz": 50, "validationTime_s": 1,
            "faultStart_s": 0.3, "faultComplete_s": 0.301,
            "recoveryStart_s": 0.6, "recoveryComplete_s": 0.601,
            "faultVoltage": 0.5,
            "outputStep_s": 1e-4, "residualTolerance": 1e-3,
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
        gfl = GFL_C_F_L(1, 8100, "1")
        for name, value in (("KpPll", 0.2), ("KiPll", 0.2)):
            gfl.p[name]["num"] = value
        for name in ("Rg", "Lg", "K1", "Id2"):
            gfl.p[name]["num"] = p[name]
        source = Infbus(1, 8100, "2")
        # Same convention as the CIGRE systems: Iqref = -Qref, so a positive Qref injects reactive power.
        refs = {str(gfl.uref[0]): p["Pref"], str(gfl.uref[1]): -p["Qref"], str(source.uref[0]): p["gridVoltage"]}
        devices = {"Vdev": [source], "Idev": [gfl], "gfl": gfl, "source": source}
        network = {
            "A": np.empty((1, 0)), "R": [], "L": [],
            "busNames": ["PCC / Infinite bus"],
        }
        super().__init__(type(self).__name__, network, devices,
            {"Vbase": 1, "Sbase": 8100, "frequency_Hz": 50}, refs,
        )
        self.solverSettings = {
            "method": "DOP853",
            "max_step": 0.1 / (2 * np.pi * self.base["frequency_Hz"]),
            "rtol": 1e-3,
            "atol": 1e-5,
        }
        self.parameters, self.parameterNames = p, list(p)
