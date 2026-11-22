"""Common symbolic interface for voltage and current devices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

import numpy as np
import sympy as sp


class Device(ABC):
    """Base class for all converter, source, load, and passive devices.

    Concrete device classes only need to define their symbolic equations and
    initialization/power-flow relations. Symbolic expressions remain in memory;
    MATLAB's generated ``.m``/``.mat`` cache has deliberately not been ported.
    """

    def __init__(
        self,
        Vb: float,
        Sb: float,
        type: str,
        index: str | int,
        paramConfig: Mapping[str, Any] | Iterable[tuple[str, Any]],
        stateNames: Iterable[str],
        referenceNames: Iterable[str],
    ) -> None:
        if type not in {"Voltage", "Current"}:
            raise ValueError("type must be 'Voltage' or 'Current'")

        self.Vb = float(Vb)
        self.Sb = float(Sb)
        self.Ib = self.Sb / (np.sqrt(3 / 2) * self.Vb)
        self.type = type
        self.index = f"_{index}"
        self.Inputscale = 1.0
        self.Outputscale = 1.0

        if type == "Voltage":
            self.y = self._symbols(("va", "vb"))
            self.u = self._symbols(("ia", "ib"))
        else:
            self.y = self._symbols(("ia", "ib"))
            self.u = self._symbols(("va", "vb"))

        items = paramConfig.items() if isinstance(paramConfig, Mapping) else paramConfig
        self.p: dict[str, dict[str, Any]] = {}
        for name, value in items:
            array = np.asarray(value)
            if array.ndim == 0:
                symbol: Any = sp.Symbol(f"{name}{self.index}", real=True)
            else:
                symbol = np.asarray(
                    sp.symbols(f"{name}{self.index}_0:{array.size}", real=True),
                    dtype=object,
                ).reshape(array.shape, order="F")
            self.p[str(name)] = {"num": value, "sym": symbol}

        self.x = self._symbols(stateNames)
        self.uref = self._symbols(referenceNames)
        self.n = len(self.x)
        self.f, self.g = self.getStateAndOutputEquationsSym()
        self.f = sp.Matrix(self.f) if len(self.f) else sp.zeros(0, 1)
        self.g = sp.Matrix(self.g) if len(self.g) else sp.zeros(0, 1)
        self.f_core = None
        self.g_core = None
        self.parameterMap = self._parameter_map()

    def _symbols(self, names: Iterable[str]) -> tuple[sp.Symbol, ...]:
        return tuple(sp.Symbol(f"{name}{self.index}", real=True) for name in names)

    def _parameter_map(self) -> dict[str, Any]:
        symbols: list[sp.Expr] = []
        values: list[float] = []
        for parameter in self.p.values():
            symbols.extend(np.asarray(parameter["sym"], dtype=object).reshape(-1, order="F"))
            values.extend(np.asarray(parameter["num"], dtype=float).reshape(-1, order="F"))
        return {
            "parameterSymbols": tuple(symbols),
            "parameterValues": np.asarray(values, dtype=float),
        }

    def getStateMatrix(self) -> sp.Matrix:
        return self.f.jacobian(self.x)

    def convertToAnotherBase(self, Vbnew: float, Sbnew: float) -> None:
        voltageOutputScale = self.Vb / Vbnew
        newCurrentBase = Sbnew / (np.sqrt(3 / 2) * Vbnew)
        currentOutputScale = self.Ib / newCurrentBase
        if self.type == "Voltage":
            self.Inputscale = 1 / currentOutputScale
            self.Outputscale = voltageOutputScale
        else:
            self.Inputscale = 1 / voltageOutputScale
            self.Outputscale = currentOutputScale

    # A device has four model layers:
    #   EMT         getStateAndOutputEquationsSym  dx/dtau = f(x, u), output g(x, u)
    #   power flow  getPowerFlowEquations          static algebraic model
    #   init        getFInit                       the f rows of every state except theta
    #   RMS         getRMSEquationsSym             f with dx_f = 0: dx_s = f_rms, 0 = g_rms

    @abstractmethod
    def getStateAndOutputEquationsSym(self) -> tuple[Any, Any]:
        """EMT model: state derivative ``f`` (per-unit time) and port output ``g``.
        States are in the device's own dq frame, whose angle ``theta`` is a state;
        the port quantities ``u`` and ``g`` are in the stationary frame."""

    @abstractmethod
    def getPowerFlowEquations(self) -> Any:
        """Static algebraic model used by the power flow."""

    @abstractmethod
    def initFromPowerFlow(self, P: float, Q: float, V: float, theta: float) -> Any:
        """Return an initial state estimate for a power-flow solution."""

    #: Local indices of the frame-angle states ``theta`` (no steady-state row).
    thetaIndices: tuple[int, ...] = ()
    #: Local indices of the other angle states (the model is 2*pi-periodic in them,
    #: e.g. a GFM ``delta``). They and ``thetaIndices`` are wrapped to [-pi, pi).
    angleIndices: tuple[int, ...] = ()
    #: Local indices of the slow states that stay dynamic in the RMS model
    #: besides ``theta``.
    rmsDynamicIndices: tuple[int, ...] = ()

    def calculateSignals(self, time_s: Any, x: Any, u: Any, uref: Any) -> dict[str, Any]:
        """Output signals of this device that are neither states nor network signals.

        Per sample: ``x`` (samples x states), ``u`` (samples x 2, the port input in
        alpha-beta on the device base: the terminal voltage of a current-port device,
        the current of a voltage-port device), ``uref`` (samples x references).
        Returns ``{name: array (samples,)}`` on the device base. Default: none.
        """
        return {}

    def getFInit(self) -> sp.Matrix:
        """The EMT rows ``f_i`` of every state except ``theta`` (steady state: ``0 = f_i``)."""
        return sp.Matrix([self.f[i] for i in range(self.n) if i not in self.thetaIndices])

    # ------------------------------------------------------------------ RMS
    def getRMSEquationsSym(self) -> tuple[list[sp.Expr], list[sp.Expr]]:
        """RMS model: the EMT model with ``dx_f = 0``, a DAE in the DQ frame.

        Returns ``(f_rms, g_rms)``, both in state order:

        * ``dx_s/dtau = f_rms`` for the slow states (``thetaIndices`` and
          ``rmsDynamicIndices``). ``theta`` is measured relative to the DQ frame
          rotating at 1 pu, so its row is ``f_theta - 1``.
        * ``0 = g_rms`` for the fast states (every other state).
        """
        slow = set(self.thetaIndices) | set(self.rmsDynamicIndices)
        f_rms = [self.f[i] - (1 if i in self.thetaIndices else 0) for i in range(self.n) if i in slow]
        g_rms = [self.f[i] for i in range(self.n) if i not in slow]
        return f_rms, g_rms
