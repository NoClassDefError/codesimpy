"""Symbolic assembly and numerical solution of an RL multi-infeed grid."""

from __future__ import annotations

import re
import warnings
from collections import OrderedDict
from functools import cached_property
from typing import Any, Iterable

import numpy as np
import sympy as sp
from scipy.linalg import eig
from scipy.optimize import least_squares, root

from .symbolic import cseJacobian, expandCse, lambdifyCse, splitColumns

# Symbolic models keyed by RLGrid._structureKey(); oldest entries are evicted first.
_MODEL_CACHE: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
_MODEL_CACHE_SIZE = 8


class RLGrid:
    """Interconnect voltage/current devices in stationary alpha-beta axes.

    Unlike the MATLAB implementation this class compiles SymPy expressions to
    NumPy callables in memory. There are no generated source files, cache hashes,
    path manipulation, or toolbox-specific derivative cleanup passes.
    """

    def __init__(
        self, A: Any, L: Any, R: Any, Vb: float, Sb: float,
        Vdev: Iterable[Any], Idev: Iterable[Any], systemName: str = "default",
        busNames: Iterable[Any] | None = None,
    ) -> None:
        """``Vdev[i]`` is the voltage-port device of node i or ``None``; ``Idev[i]`` is a
        current-port device, a list of them, or ``None``.

        Nodes with a voltage device are retained. Every other node belongs to a
        *cluster* of connected nodes without a voltage device; each cluster is
        removed by :meth:`localDAEreduction` (KCL of the cluster combined with
        its branch and current-device equations). Branches between two retained
        nodes form the network ``A``/``L``/``R``; the power flow uses the whole
        network (``pfA``/``pfL``/``pfR``: retained nodes, then ``floatingNodes``).
        """
        from ..components import ZeroCurrent

        # 1. Inputs. Every Idev entry becomes a list of current-port devices.
        A = np.asarray(A, dtype=float)
        L = np.asarray(L, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float).reshape(-1)
        Vdev = list(Vdev)
        Idev = [[] if item is None else list(item) if isinstance(item, (list, tuple)) else [item]
                for item in Idev]
        nodeCount, branchCount = A.shape
        if len(Vdev) != nodeCount or len(Idev) != nodeCount:
            raise ValueError("Vdev and Idev need one entry per node")
        names = [str(name) for name in busNames] if busNames is not None else [str(i + 1) for i in range(nodeCount)]
        safeNames = [re.sub(r"\W", "_", name) for name in names]  # for symbol names

        # 2. Node and branch classes. A node with a voltage device is *retained*. A branch
        #    between two retained nodes is a *network branch*: its current is a state of the
        #    EMT model, so it needs an inductance. Every other branch touches a cluster.
        isVoltage = np.asarray([device is not None for device in Vdev], dtype=bool)
        voltageNodes = [int(i) for i in np.flatnonzero(isVoltage)]
        ends = [np.flatnonzero(A[:, column]) for column in range(branchCount)]  # nodes of each branch
        for column in range(branchCount):
            if R[column] == 0 and L[column] == 0:
                raise ValueError(f"branch {column + 1} has zero impedance (R = L = 0)")
        networkColumns = [column for column in range(branchCount) if isVoltage[ends[column]].all()]
        for column in networkColumns:
            if L[column] <= 0:
                raise ValueError(
                    f"branch {column + 1} between buses with voltage devices "
                    f"({', '.join(names[i] for i in ends[column])}) has L = 0; "
                    "a branch between two voltage ports needs an inductance")

        # 3. Clusters: connected groups of nodes without a voltage device (found by a graph
        #    search that does not cross retained nodes). Each cluster is later reduced by
        #    localDAEreduction; here its structure is fixed:
        #    * columns: every branch that touches the cluster;
        #    * removed: inductive branches whose current is eliminated, one per KCL combination
        #      that does not contain the cluster voltages. Those combinations are the left null
        #      space of the Laplacian of the resistive (L = 0) branches; a branch is taken when it
        #      raises the rank of the combinations restricted to the chosen branches;
        #    * kept: the other inductive branches, whose currents stay states.
        self.clusters: list[dict[str, Any]] = []
        seen: set[int] = set()
        for start in range(nodeCount):
            if isVoltage[start] or start in seen:
                continue
            nodes, frontier = [], [start]
            seen.add(start)
            while frontier:
                node = frontier.pop()
                nodes.append(node)
                for column in np.flatnonzero(A[node]):
                    for other in ends[column]:
                        if not isVoltage[other] and int(other) not in seen:
                            seen.add(int(other))
                            frontier.append(int(other))
            nodes.sort()
            columns = sorted({int(column) for node in nodes for column in np.flatnonzero(A[node])})
            devices = [(device, node) for node in nodes for device in Idev[node]]
            if not devices and not any(isVoltage[ends[column]].any() for column in columns):
                continue  # passive island without a source: no current can flow, drop it
            inductive = [column for column in columns if L[column] > 0]
            resistive = [column for column in columns if L[column] == 0]
            incidenceR = A[np.ix_(nodes, resistive)]
            laplacian = (incidenceR @ np.diag(1 / R[resistive]) @ incidenceR.T if resistive
                         else np.zeros((len(nodes), len(nodes))))
            left, singular, _ = np.linalg.svd(laplacian)
            voltageFreeCombinations = left[:, singular <= 1e-9 * max(1.0, singular.max(initial=0.0))].T
            coupling = voltageFreeCombinations @ A[np.ix_(nodes, inductive)]
            removed: list[int] = []
            for position, column in enumerate(inductive):
                chosen = [inductive.index(item) for item in removed] + [position]
                if np.linalg.matrix_rank(coupling[:, chosen], tol=1e-9) == len(chosen):
                    removed.append(column)
            # Current symbols of the cluster branches, named after their first cluster node.
            symbols = {}
            for column in columns:
                owner = safeNames[next(node for node in ends[column] if not isVoltage[node])]
                symbols[column] = (sp.Symbol(f"iga_{owner}_{column + 1}", real=True),
                                   sp.Symbol(f"igb_{owner}_{column + 1}", real=True))
            self.clusters.append({
                "nodes": nodes, "columns": columns, "devices": devices, "removed": removed,
                "kept": [column for column in inductive if column not in removed],
                "symbols": symbols,
                "Lsym": {column: sp.Symbol(f"Lc{column + 1}", real=True) for column in columns},
                "Rsym": {column: sp.Symbol(f"Rc{column + 1}", real=True) for column in columns},
                "label": ", ".join(names[node] for node in nodes),
            })
        self._fullIncidence = A
        self.floatingNodes = [node for cluster in self.clusters for node in cluster["nodes"]]
        self.floatingColumns = [column for cluster in self.clusters for column in cluster["columns"]]

        # 4. Networks. EMT: retained nodes and network branches (A, L, R). Power flow: the
        #    whole network, retained nodes first, then floatingNodes (pfA, pfL, pfR).
        self.systemName = str(systemName or "default")
        self.Vb, self.Sb = float(Vb), float(Sb)
        self.A = A[np.ix_(voltageNodes, networkColumns)]
        self.L, self.R = L[networkColumns], R[networkColumns]
        self.pfA = A[np.ix_(voltageNodes + self.floatingNodes, networkColumns + self.floatingColumns)]
        self.pfL, self.pfR = L[networkColumns + self.floatingColumns], R[networkColumns + self.floatingColumns]
        self.n, self.b = self.A.shape
        self._pfNode = {node: position for position, node in enumerate(voltageNodes + self.floatingNodes)}

        # 5. Devices. Each retained node has one voltage device and one current device in the
        #    regular slots (ZeroCurrent if it has none). Every other current device is
        #    *attached* as (device, power-flow node): extra devices of a retained node, then the
        #    devices of the clusters. All devices are converted to the system base.
        self.Vdev = [Vdev[node] for node in voltageNodes]
        self.Idev = []
        self.attached: list[tuple[Any, int]] = []
        for position, node in enumerate(voltageNodes):
            first, *extra = Idev[node] or [ZeroCurrent(Vb, Sb, f"zero{node + 1}")]
            self.Idev.append(first)
            self.attached.extend((device, position) for device in extra)
        for cluster in self.clusters:
            self.attached.extend(
                (device, self.n + self.floatingNodes.index(node)) for device, node in cluster["devices"])
        for device in self.Vdev + self.Idev + [device for device, _ in self.attached]:
            if device.Vb != self.Vb or device.Sb != self.Sb:
                device.convertToAnotherBase(self.Vb, self.Sb)

        # 6. Branch parameters as symbols (values in p): network branches Lg/Rg,
        #    cluster branches Lc/Rc.
        self.p: dict[str, Any] = {}
        self.Lsym = tuple(sp.Symbol(f"Lg{i + 1}", real=True) for i in range(self.b))
        self.Rsym = tuple(sp.Symbol(f"Rg{i + 1}", real=True) for i in range(self.b))
        for i in range(self.b):
            self.p[str(self.Lsym[i])] = {"num": self.L[i], "sym": self.Lsym[i]}
            self.p[str(self.Rsym[i])] = {"num": self.R[i], "sym": self.Rsym[i]}
        for cluster in self.clusters:
            for column in cluster["columns"]:
                self.p[str(cluster["Lsym"][column])] = {"num": L[column], "sym": cluster["Lsym"][column]}
                self.p[str(cluster["Rsym"][column])] = {"num": R[column], "sym": cluster["Rsym"][column]}

        # 7. State x and reference u order: branch currents (network branches, then kept
        #    cluster branches; branchStateCount of them), then per retained node its voltage
        #    and current device, then the attached devices.
        self.ibsym = tuple(
            symbol for i in range(self.b)
            for symbol in (sp.Symbol(f"iga{i + 1}", real=True), sp.Symbol(f"igb{i + 1}", real=True))
        )
        self.x = list(self.ibsym) + [
            symbol for cluster in self.clusters for column in cluster["kept"] for symbol in cluster["symbols"][column]
        ]
        self.branchStateCount = len(self.x)
        self.u: list[sp.Symbol] = []
        for voltageDevice, currentDevice in zip(self.Vdev, self.Idev):
            self.u.extend(voltageDevice.uref)
            self.u.extend(currentDevice.uref)
            self.x.extend(voltageDevice.x)
            self.x.extend(currentDevice.x)
        for device, _ in self.attached:
            self.u.extend(device.uref)
            self.x.extend(device.x)
        # Frame angles (Device.thetaIndices): the model is 2*pi-periodic in them.
        self.thetaStateIndices = np.asarray([
            self.x.index(device.x[i])
            for device in self.Vdev + self.Idev + [device for device, _ in self.attached]
            for i in device.thetaIndices
        ], dtype=int)
        # Every angle state (frame angles and Device.angleIndices), wrapped to [-pi, pi).
        self.angleStateIndices = np.asarray([
            self.x.index(device.x[i])
            for device in self.Vdev + self.Idev + [device for device, _ in self.attached]
            for i in sorted(set(device.thetaIndices) | set(device.angleIndices))
        ], dtype=int)
        self.y: list[sp.Symbol] = []
        self.vnsym: tuple[sp.Symbol, ...] = ()

        # 8. Symbolic model (built once per structure, see _structureKey).
        self.refreshParameters(rebuild=True)
        self._buildSymbolicModel()

    def getStateAndOutputEquationsSym(self) -> tuple[sp.Matrix, sp.Matrix]:
        """EMT model ``dx/dtau = f``, outputs ``g = [ib, vn, deviceCurrent, v_floating, i_floating]``.

        All vectors are alpha-beta pairs on the system base; time is per unit.

        * ``ib``: network branch currents (from the ``+1`` to the ``-1`` node of ``A``);
        * ``vn``: retained-node voltages (outputs of the voltage devices);
        * ``deviceCurrent``: current injected at each retained node by its current
          devices (regular and extra) and by the cluster branches that end on it;
        * ``v_floating``: voltages of ``floatingNodes``; they are symbols defined by
          ``self._intermediates`` (see :meth:`localDAEreduction`);
        * ``i_floating``: currents of ``floatingColumns`` (in the direction of ``A``).

        ``f`` follows the state order of ``self.x``.
        """
        # 1. Network branch currents and the current they draw from each retained node.
        Abar = (
            sp.kronecker_product(sp.Matrix(self.A), sp.eye(2))
            if self.b else sp.zeros(2 * self.n, 0)
        )
        ib = sp.Matrix(self.ibsym) if self.b else sp.zeros(0, 1)
        networkCurrent = Abar * ib

        # 2. Retained nodes: the voltage device gives the node voltage vn (its output depends
        #    on states only), the regular current device its current from vn.
        vn = sp.zeros(2 * self.n, 1)
        deviceCurrent = sp.zeros(2 * self.n, 1)
        for i, (Vdev, Idev) in enumerate(zip(self.Vdev, self.Idev)):
            indices = [2 * i, 2 * i + 1]
            voltageSubs = dict(zip(Vdev.u, [Vdev.Inputscale * networkCurrent[j] for j in indices]))
            voltage = Vdev.Outputscale * Vdev.g.subs(voltageSubs, simultaneous=True)
            vn[indices[0]], vn[indices[1]] = voltage[0], voltage[1]
            currentSubs = dict(zip(Idev.u, [Idev.Inputscale * vn[j] for j in indices]))
            current = Idev.Outputscale * Idev.g.subs(currentSubs, simultaneous=True)
            deviceCurrent[indices[0]], deviceCurrent[indices[1]] = current[0], current[1]

        # 3. Extra current devices on retained nodes: same voltage vn, their currents add.
        attachedRows: dict[int, sp.Matrix] = {}
        for index, (device, node) in enumerate(self.attached):
            if node < self.n:
                currentSubs = dict(zip(device.u, [device.Inputscale * vn[j] for j in (2 * node, 2 * node + 1)]))
                current = device.Outputscale * device.g.subs(currentSubs, simultaneous=True)
                deviceCurrent[2 * node] += current[0]
                deviceCurrent[2 * node + 1] += current[1]
                attachedRows[index] = device.f.subs(currentSubs, simultaneous=True)

        # 4. Clusters (nodes without a voltage device): write each one as the DAE
        #        E dz/dt = F(z, v),   0 = KCL(z, v)
        #    z: its device states and inductive branch currents, v: its node voltages, and
        #    reduce it to an ODE with localDAEreduction.
        floatingVoltage = sp.zeros(2 * len(self.floatingNodes), 1)
        floatingCurrent = sp.zeros(2 * len(self.floatingColumns), 1)
        clusterRows: dict[sp.Symbol, sp.Expr] = {}  # dz/dt of the kept cluster states
        self._intermediates: list[tuple[sp.Symbol, sp.Expr]] = []  # definitions of the v symbols
        self._reductionWarnings = []
        fullA = self._fullIncidence
        for cluster in self.clusters:
            nodes, columns = cluster["nodes"], cluster["columns"]
            # Voltage of every node the cluster touches: unknown symbols on its own nodes,
            # vn on retained nodes, zero for a missing branch end.
            voltageSymbols = [sp.Dummy(f"v{axis}_{node}", real=True) for node in nodes for axis in "ab"]
            voltageAt = {node: sp.Matrix(voltageSymbols[2 * k:2 * k + 2]) for k, node in enumerate(nodes)}
            voltageAt[None] = sp.zeros(2, 1)
            for column in columns:
                for node in np.flatnonzero(fullA[:, column]):
                    position = self._pfNode[int(node)]
                    if position < self.n:
                        voltageAt[int(node)] = vn[2 * position:2 * position + 2, :]

            clusterStates, E, clusterF = [], [], []
            injection = {node: sp.zeros(2, 1) for node in nodes}  # device current into each node
            # 4a. Current devices: E = 1, F = f(x, v of their node). Their output must not
            #     depend on the port voltage, otherwise KCL would be implicit in v.
            for device, node in cluster["devices"]:
                if set(device.u) & sp.Matrix(device.g).free_symbols:
                    raise ValueError(
                        f"bus {cluster['label']}: {type(device).__name__} output depends on its port "
                        "voltage, so the bus cannot be reduced; add a C or a voltage device")
                portSubs = dict(zip(device.u, list(device.Inputscale * voltageAt[node])))
                injection[node] += device.Outputscale * device.g
                clusterStates.extend(device.x)
                E.extend([1] * device.n)
                clusterF.extend(device.f.subs(portSubs, simultaneous=True))
            # 4b. Branches: inductive ones are states, L di/dt = (v+ - v-) - R i; resistive
            #     ones (L = 0) are algebraic, i = (v+ - v-) / R.
            branchCurrent = {}
            for column in columns:
                plus = np.flatnonzero(fullA[:, column] > 0)
                minus = np.flatnonzero(fullA[:, column] < 0)
                drop = (voltageAt[int(plus[0]) if plus.size else None]
                        - voltageAt[int(minus[0]) if minus.size else None])
                Lc, Rc = cluster["Lsym"][column], cluster["Rsym"][column]
                if column in cluster["removed"] or column in cluster["kept"]:
                    current = sp.Matrix(cluster["symbols"][column])
                    clusterStates.extend(current)
                    E.extend([Lc, Lc])
                    clusterF.extend(drop - Rc * current)
                else:
                    current = drop / Rc
                branchCurrent[column] = current
            # 4c. KCL at every cluster node: branch currents leaving = device currents injected.
            kcl = sp.Matrix.vstack(*[
                sum((int(fullA[node, column]) * branchCurrent[column] for column in columns
                     if fullA[node, column]), sp.zeros(2, 1)) - injection[node]
                for node in nodes
            ])
            # 4d. Reduce: eliminated branch currents become functions of the kept states, the
            #     node voltages stay symbols defined in self._intermediates.
            reduction = self.localDAEreduction(
                E, sp.Matrix(clusterF), kcl, clusterStates, voltageSymbols,
                eliminate=[symbol for column in cluster["removed"] for symbol in cluster["symbols"][column]],
                label=f"bus {cluster['label']}")
            stateMap = reduction["stateMap"]
            clusterRows.update(reduction["rows"])
            self._intermediates.extend(reduction["definitions"])
            # 4e. Outputs, and the cluster branch currents as injections at the retained nodes.
            for k, node in enumerate(nodes):
                offset = 2 * self.floatingNodes.index(node)
                floatingVoltage[offset:offset + 2, :] = sp.Matrix(voltageSymbols[2 * k:2 * k + 2])
            for column in columns:
                current = branchCurrent[column].xreplace(stateMap)
                offset = 2 * self.floatingColumns.index(column)
                floatingCurrent[offset:offset + 2, :] = current
                for node in np.flatnonzero(fullA[:, column]):
                    position = self._pfNode[int(node)]
                    if position < self.n:
                        deviceCurrent[2 * position:2 * position + 2, :] -= fullA[node, column] * current
            removedNames = [str(symbol) for symbol in stateMap if stateMap[symbol] != symbol]
            # Every reduced bus is reported: it has no voltage device, so its voltage is algebraic.
            self._reductionWarnings.append(
                f"bus {cluster['label']} has no C or voltage-port device and was reduced by local DAE->ODE "
                f"({len(columns)} branch(es), {len(cluster['devices'])} current device(s)); "
                f"removed state(s): {', '.join(removedNames) or 'none'}; "
                f"driven by the reduced voltage: {', '.join(map(str, reduction['driven'])) or 'none'}")

        # 5. State equations in the order of self.x: network branches L di/dt = A^T vn - R i,
        #    kept cluster branches, then per retained node its voltage device (input: the
        #    current it supplies, network minus devices) and current device (input: vn), then
        #    the attached devices.
        if self.b:
            Ldiag = sp.diag(*[value for item in self.Lsym for value in (item, item)])
            Rdiag = sp.diag(*[value for item in self.Rsym for value in (item, item)])
            branchDerivative = Ldiag.inv() * (Abar.T * vn - Rdiag * ib)
        else:
            branchDerivative = sp.zeros(0, 1)
        stateParts: list[sp.Matrix] = [branchDerivative, sp.Matrix(
            [clusterRows[symbol] for symbol in self.x[2 * self.b:self.branchStateCount]])]
        for i, (Vdev, Idev) in enumerate(zip(self.Vdev, self.Idev)):
            indices = [2 * i, 2 * i + 1]
            voltageInput = [networkCurrent[j] - deviceCurrent[j] for j in indices]
            voltageSubs = dict(zip(Vdev.u, [Vdev.Inputscale * item for item in voltageInput]))
            currentSubs = dict(zip(Idev.u, [Idev.Inputscale * vn[j] for j in indices]))
            stateParts.extend([
                Vdev.f.subs(voltageSubs, simultaneous=True),
                Idev.f.subs(currentSubs, simultaneous=True),
            ])
        for index, (device, node) in enumerate(self.attached):
            stateParts.append(attachedRows[index] if node < self.n
                              else sp.Matrix([clusterRows[symbol] for symbol in device.x]))
        stateParts = [part for part in stateParts if part.rows]
        return (sp.Matrix.vstack(*stateParts) if stateParts else sp.zeros(0, 1),
                sp.Matrix.vstack(ib, vn, deviceCurrent, floatingVoltage, floatingCurrent))

    @staticmethod
    def localDAEreduction(
        E: Iterable[Any], f: sp.Matrix, g: sp.Matrix, x: Iterable[sp.Symbol], y: Iterable[sp.Symbol],
        *, eliminate: Iterable[sp.Symbol] = (), label: str = "DAE",
    ) -> dict[str, Any]:
        """Reduce the semi-explicit DAE ``E dx/dt = f(x, y)``, ``0 = g(x, y)`` to an ODE.

        * ``E``: diagonal of the descriptor matrix, one nonzero entry per state.
        * ``f``, ``g``: column matrices; ``x``: the states; ``y``: the algebraic
          unknowns, as many as rows of ``g``.
        * ``eliminate``: the states that the y-free constraints are solved for, one
          per constraint.

        Rows of ``g`` that contain ``y`` are used as they are (index 1); the
        y-free combinations ``w g`` (``w`` in the left null space of ``dg/dy``)
        are differentiated (index 2): ``w (dg/dx) E^-1 f = 0``. These equations
        give ``y``; the y-free constraints give the eliminated states. The
        reduction is exact: it raises ``ValueError`` when the equations for
        ``y`` are not linear in ``y`` or a constraint has no state to eliminate
        (the DAE is then not reducible to an ODE in closed form; no DAE is solved).

        Returns a dict:

        * ``states``: the kept states (``x`` order); ``rows``: ``dx/dt`` of each kept
          state, in terms of the kept states and the symbols ``y``;
        * ``definitions``: ordered ``(symbol, expression)`` pairs that define ``y``
          (and the auxiliary symbols it uses) from the kept states. ``y`` is not
          substituted into ``rows``: its expression can be large, so compile and
          differentiate through the definitions (``cseJacobian(intermediates=...)``);
        * ``stateMap``: every state of ``x`` -> its expression in the kept states
          (a kept state maps to itself; no ``y``);
        * ``driven``: states whose ``f`` row depends on ``y``.

        Raises ``ValueError`` (prefixed with ``label``) if the reduction fails.
        """
        # 0. Inputs.
        E, x, y, eliminate = list(E), list(x), list(y), list(eliminate)
        f, g = sp.Matrix(f), sp.Matrix(g)
        if len(E) != len(x) or f.rows != len(x) or g.rows != len(y):
            raise ValueError(f"{label}: inconsistent DAE dimensions")
        if any(sp.sympify(item) == 0 for item in E):
            raise ValueError(f"{label}: the descriptor matrix E is singular")
        randomPoint = np.random.default_rng(1)  # structural rank tests at random values

        # 1. Split the constraints by index. Index 1: the rows of g that contain y (a set of
        #    independent rows of dg/dy); they determine y directly. Index 2: the combinations
        #    w g with w dg/dy = 0; they are free of y and constrain the states only.
        dgdy = g.jacobian(y)
        _, index1Rows = dgdy.T.rref(simplify=True)
        index2Combinations = [vector.T for vector in dgdy.T.nullspace(simplify=True)]
        constraints = [sp.expand((w * g)[0]) for w in index2Combinations]  # 0 = w g(x)
        if any(set(y) & sp.sympify(item).free_symbols for item in constraints):
            raise ValueError(f"{label}: a constraint combination still depends on the algebraic unknowns")

        # 2. Equations for y: the index-1 rows, and the time derivative of each index-2
        #    constraint, d(w g)/dt = w (dg/dx) E^-1 f(x, y) = 0, which contains y through f.
        dgdx = g.jacobian(x) if x else sp.zeros(g.rows, 0)
        dxdt = sp.Matrix(len(x), 1, [f[i] / E[i] for i in range(len(x))])
        yEquations = sp.Matrix([g[i] for i in index1Rows] + [(w * dgdx * dxdt)[0] for w in index2Combinations])

        # 3. Closed form only if they are linear in y: M y + b = 0. Nonlinear -> error (no DAE
        #    is solved). M must be nonsingular (checked at random values of its symbols).
        M = yEquations.jacobian(y)
        if set(y) & M.free_symbols:
            nonlinear = [str(x[i]) for i in range(len(x)) if set(y) & f[i:i + 1, :].jacobian(y).free_symbols]
            raise ValueError(
                f"{label}: the DAE cannot be reduced to an ODE, its equations are nonlinear in the "
                f"algebraic unknowns (rows of {', '.join(nonlinear) or 'the constraints'}); "
                "for a bus: add a C or a voltage device")
        b = yEquations.xreplace({symbol: 0 for symbol in y})
        symbols = sorted(M.free_symbols, key=str)
        numericM = sp.lambdify(symbols, M, "numpy", cse=True)
        with np.errstate(all="ignore"):
            samples = [np.asarray(numericM(*randomPoint.uniform(0.5, 1.5, len(symbols))), dtype=float)
                       for _ in range(5)]
        conditions = [np.linalg.cond(sample) for sample in samples if np.isfinite(sample).all()]
        if conditions and min(conditions) >= 1e10:
            raise ValueError(f"{label}: the reduced equations are singular, the algebraic unknowns are not "
                             "determined (for a bus: add a C or a voltage device)")

        # 4. Eliminated states: each index-2 constraint 0 = w g(x) is linear in the
        #    eliminated states and is solved for them (one state per constraint).
        stateMap: dict[sp.Symbol, sp.Expr] = {}
        if len(eliminate) != len(constraints):
            raise ValueError(
                f"{label}: the DAE cannot be reduced to an ODE, {len(constraints)} constraint(s) but "
                f"{len(eliminate)} state(s) to eliminate (for a bus: a current device needs a branch "
                "to a voltage source, or add a C)")
        if eliminate:
            coefficients = sp.Matrix(constraints).jacobian(eliminate)
            if set(eliminate) & coefficients.free_symbols:
                raise ValueError(f"{label}: the constraints are not linear in the eliminated states")
            # Pick constraint rows whose coefficients are independent (rank at random values).
            rows: list[int] = []
            for row in range(coefficients.rows):
                selected = coefficients[rows + [row], :]
                symbols = sorted(selected.free_symbols, key=str)
                values = dict(zip(symbols, randomPoint.uniform(0.5, 1.5, len(symbols))))
                if np.linalg.matrix_rank(np.asarray(selected.xreplace(values).evalf(), dtype=float), tol=1e-9) == len(rows) + 1:
                    rows.append(row)
                if len(rows) == len(eliminate):
                    break
            if len(rows) < len(eliminate):
                raise ValueError(f"{label}: the states {', '.join(map(str, eliminate))} "
                                 "cannot be solved from the constraints")
            # Linear: constraint = coefficients * eliminate + rest, so eliminate -= coefficients^-1 constraint.
            solution = sp.Matrix(eliminate) - coefficients[rows, :].LUsolve(sp.Matrix([constraints[row] for row in rows]))
            stateMap = dict(zip(eliminate, solution))

        # 5. Solve M y = -b on placeholder symbols for the nonzero entries of M and b. The
        #    entries can be large (they contain device equations), so they are not substituted
        #    into y: definitions lists placeholders, then y, each defined from earlier symbols.
        definitions: list[tuple[sp.Symbol, sp.Expr]] = []
        placeholderM, placeholderB = sp.zeros(len(y), len(y)), sp.zeros(len(y), 1)
        for i in range(len(y)):
            if b[i] != 0:
                placeholderB[i] = sp.Dummy(f"e{i}", real=True)
                definitions.append((placeholderB[i], b[i].xreplace(stateMap)))
            for j in range(len(y)):
                if M[i, j] != 0:
                    placeholderM[i, j] = sp.Dummy(f"J{i}{j}", real=True)
                    definitions.append((placeholderM[i, j], M[i, j].xreplace(stateMap)))
        if (set(y) | set(stateMap)) & set().union(*(item.free_symbols for _, item in definitions)):
            raise ValueError(f"{label}: reduction failed, the algebraic solution still contains "
                             "an algebraic unknown or an eliminated state")
        step = placeholderM.LUsolve(placeholderB, iszerofunc=lambda item: item == 0)
        definitions += [(symbol, -step[i]) for i, symbol in enumerate(y)]

        # 6. The ODE: dx/dt of every kept state, with the eliminated states substituted;
        #    y stays a symbol (defined by definitions).
        kept = [symbol for symbol in x if symbol not in stateMap]
        reducedRows = {}
        for i, symbol in enumerate(x):
            if symbol in stateMap:
                continue
            row = dxdt[i].xreplace(stateMap)
            if set(stateMap) & row.free_symbols or row.has(sp.Derivative):
                raise ValueError(f"{label}: reduction failed, the row of {symbol} still contains "
                                 "an eliminated state")
            reducedRows[symbol] = row
        stateMap.update({symbol: symbol for symbol in kept})
        driven = [x[i] for i in range(len(x)) if set(y) & sp.sympify(f[i]).free_symbols]
        return {
            "states": kept, "rows": reducedRows, "definitions": definitions, "stateMap": stateMap,
            "driven": driven,
        }

    def _structureKey(self) -> tuple:
        """Fingerprint everything that enters the symbolic model, but no numeric values.

        Parameter values (``p[...]["num"]``, ``L``, ``R``) are passed to the compiled
        functions at call time, so grids that differ only in those values share one
        cached model. Anything written into the equations as a number (per-unit scales,
        values substituted inside a device) is part of ``f``/``g``/``getFInit`` or the
        scales below, so changing it gives a different key.
        """
        def matrix(value: Any) -> sp.ImmutableMatrix:
            return sp.ImmutableMatrix(sp.Matrix(value) if len(value) else sp.zeros(0, 1))

        devices = tuple(
            (
                f"{type(device).__module__}.{type(device).__qualname__}",
                device.index, device.type,
                float(device.Inputscale), float(device.Outputscale),
                tuple(device.x), tuple(device.u), tuple(device.uref),
                matrix(device.f), matrix(device.g), matrix(device.getFInit()),
            )
            for device in self.Vdev + self.Idev + [device for device, _ in self.attached]
        )
        clusters = tuple(
            (tuple(cluster["nodes"]), tuple(cluster["columns"]), tuple(cluster["removed"]),
             tuple(cluster["kept"]), tuple(node for _, node in cluster["devices"]))
            for cluster in self.clusters
        )
        return (
            self.A.shape, self.A.tobytes(), self.pfA.shape, self.pfA.tobytes(), devices,
            tuple(node for _, node in self.attached), clusters,
            tuple(self.x), tuple(self.u),
            tuple(self.parameterMap["parameterSymbols"]),
        )

    def _buildSymbolicModel(self) -> None:
        """Attach the symbolic model, reusing a cached build when the structure matches."""
        key = self._structureKey()
        entry = _MODEL_CACHE.get(key)
        if entry is None:
            entry = self._buildSymbolicEntry()
            entry["warnings"] = tuple(self._reductionWarnings)
            entry["intermediates"] = tuple(self._intermediates)
            _MODEL_CACHE[key] = entry
            while len(_MODEL_CACHE) > _MODEL_CACHE_SIZE:
                _MODEL_CACHE.popitem(last=False)
        else:
            _MODEL_CACHE.move_to_end(key)
        for message in entry["warnings"]:
            warnings.warn(f"{self.systemName}: {message}", stacklevel=3)
        self._modelEntry = entry
        self._intermediates = list(entry["intermediates"])
        for name in ("A_cse", "A_sym", "A_core"):
            self.__dict__.pop(name, None)

        self.f, self.g = entry["f"], entry["g"]
        self.f_init_sym = entry["f_init_sym"]
        self._f_init_residual = entry["f_init_residual"]
        self.fInitStateIndices = entry["fInitStateIndices"]
        # Wrap per grid: the wrappers read this grid's parameter values.
        self.f_core = self._wrap(entry["fFunction"])
        self.g_core = self._wrap(entry["gFunction"])
        self.f_init_core = self._wrap(entry["fInitFunction"])
        self.f_numeric = self.f_core
        self.f_init = self.f_init_core

        def g_numeric(state: Any, *reference: Any):
            value = self.g_core(state, *reference)
            branchStop = 2 * self.b
            voltageStop = branchStop + 2 * self.n
            return value[:branchStop], value[branchStop:voltageStop], value[voltageStop:]
        self.g_numeric = g_numeric

    def _buildSymbolicEntry(self) -> dict[str, Any]:
        """Derive and compile the symbolic model (slow; result is shared via the cache)."""
        f, g = self.getStateAndOutputEquationsSym()
        branchStates = self.branchStateCount
        if branchStates:
            Jbar = sp.kronecker_product(sp.eye(branchStates // 2), sp.Matrix([[0, -1], [1, 0]]))
            branchInit = f[:branchStates, :] - Jbar * sp.Matrix(self.x[:branchStates])
        else:
            branchInit = sp.zeros(0, 1)
        f_init_sym = sp.MutableDenseMatrix(f)
        f_init_sym[:branchStates, :] = branchInit

        residualParts: list[sp.Matrix] = [branchInit]
        residualStateIndices = list(range(branchStates))
        stateOffset = branchStates
        for device, substitutions in self._devicePortSubstitutions(g):
            init = device.getFInit()
            init = sp.Matrix(init) if len(init) else sp.zeros(0, 1)
            residualStateIndices.extend(
                stateOffset + i for i in range(device.n) if i not in device.thetaIndices
            )
            stateOffset += device.n
            residualParts.append(init.subs(substitutions, simultaneous=True))
        f_init_residual = sp.Matrix.vstack(*residualParts)
        fInitStateIndices = np.asarray(residualStateIndices, dtype=int)
        if fInitStateIndices.size != f_init_residual.rows:
            raise ValueError("initialization residual/state mapping is inconsistent")
        fInitStateIndices.setflags(write=False)
        # Cached objects are shared between grids, so store them immutable.
        # The state Jacobian is not built here: it dominates construction time
        # and is only needed for linear analyses (see A_sym / A_core).
        return {
            "f": sp.ImmutableMatrix(f), "g": sp.ImmutableMatrix(g),
            "f_init_sym": sp.ImmutableMatrix(f_init_sym),
            "f_init_residual": sp.ImmutableMatrix(f_init_residual),
            "fInitStateIndices": fInitStateIndices,
            "fFunction": self._lambdify(f),
            "gFunction": self._lambdify(g),
            "fInitFunction": self._lambdify(f_init_residual),
        }

    @cached_property
    def A_cse(self) -> tuple[list[tuple[sp.Symbol, sp.Expr]], sp.ImmutableMatrix]:
        """State Jacobian in CSE form ``(intermediates, J)``, shared via the model cache.

        See :func:`codesimpy.core.symbolic.cseJacobian`: far faster and smaller
        than differentiating the expanded model.
        """
        if "A_cse" not in self._modelEntry:
            self._modelEntry["A_cse"] = cseJacobian(self.f_init_sym, self.x, intermediates=self._intermediates)
        return self._modelEntry["A_cse"]

    @cached_property
    def A_sym(self) -> sp.Matrix:
        """Fully expanded symbolic state Jacobian (back-substituted :attr:`A_cse`).

        Kept for API compatibility; numerical analyses use :attr:`A_core`.
        """
        if "A_sym" not in self._modelEntry:
            self._modelEntry["A_sym"] = expandCse(*self.A_cse)
        return self._modelEntry["A_sym"]

    @cached_property
    def A_core(self):
        """Numeric state Jacobian, compiled on first use and shared via the model cache."""
        if "AFunction" not in self._modelEntry:
            self._modelEntry["AFunction"] = self._lambdifyCse(*self.A_cse)
        return self._wrap(self._modelEntry["AFunction"], matrix=True)

    # ------------------------------------------------------------------ RMS
    def _devicePortSubstitutions(self, g: sp.MatrixBase):
        """Yield ``(device, substitutions)`` in state order: each device's port
        input ``u`` written with the grid output ``g`` (node voltages, device
        currents) and the branch currents."""
        Abar = (
            sp.kronecker_product(sp.Matrix(self.A), sp.eye(2))
            if self.b else sp.zeros(2 * self.n, 0)
        )
        networkCurrent = Abar * (sp.Matrix(self.ibsym) if self.b else sp.zeros(0, 1))
        voltageStart = 2 * self.b
        currentStart = voltageStart + 2 * self.n
        nodeVoltage = g[voltageStart:currentStart, :]
        deviceCurrent = g[currentStart:currentStart + 2 * self.n, :]
        floatingVoltage = g[currentStart + 2 * self.n:currentStart + 2 * self.n + 2 * len(self.floatingNodes), :]
        for i, (Vdev, Idev) in enumerate(zip(self.Vdev, self.Idev)):
            indices = [2 * i, 2 * i + 1]
            voltageInput = [networkCurrent[j] - deviceCurrent[j] for j in indices]
            yield Vdev, dict(zip(Vdev.u, [Vdev.Inputscale * item for item in voltageInput]))
            yield Idev, dict(zip(Idev.u, [Idev.Inputscale * nodeVoltage[j] for j in indices]))
        for device, node in self.attached:
            voltage = (nodeVoltage[2 * node:2 * node + 2, :] if node < self.n
                       else floatingVoltage[2 * (node - self.n):2 * (node - self.n) + 2, :])
            yield device, dict(zip(device.u, [device.Inputscale * item for item in voltage]))

    def getRMSEquationsSym(self) -> tuple[sp.Matrix, sp.Matrix, list[int]]:
        """RMS model of the grid in the DQ frame rotating at 1 pu, as a DAE.

        ``dx_s/dtau = f_rms`` for the slow states (each device's ``f_rms``) and
        ``0 = g_rms`` for every other state: the branch rows
        ``L^-1 (A^T v - R i) - J i`` and each device's ``g_rms``, with the same
        port substitutions as the EMT model. Returns ``(f_rms, g_rms, slow)``,
        ``slow`` being the global indices of the slow states.
        """
        f_rms: list[sp.Expr] = []
        g_rms: list[sp.Expr] = list(self.f_init_sym[:self.branchStateCount, :])
        slow: list[int] = []
        offset = self.branchStateCount
        for device, substitutions in self._devicePortSubstitutions(self.g):
            fDevice, gDevice = device.getRMSEquationsSym()
            f_rms.extend(sp.Matrix(fDevice).subs(substitutions, simultaneous=True))
            g_rms.extend(sp.Matrix(gDevice).subs(substitutions, simultaneous=True))
            slow.extend(offset + i for i in sorted(set(device.thetaIndices) | set(device.rmsDynamicIndices)))
            offset += device.n
        return sp.Matrix(f_rms), sp.Matrix(g_rms), slow

    @property
    def rms(self) -> dict[str, Any]:
        """Compiled RMS model, built on first use and shared via the model cache.

        ``F`` has one row per state: ``f_rms`` at the slow (differential) states
        and ``g_rms`` at the others (algebraic, ``0 = F_a``). Keys:
        ``differentialIndices``, ``algebraicIndices``, ``F`` (``F(x, u)``), ``A``
        (``dF/dx``), ``B`` (``dF/du``), ``F_sym`` and the CSE-form Jacobians
        ``A_cse`` / ``B_cse`` (see :func:`cseJacobian`).
        """
        if "rms" not in self._modelEntry:
            f_rms, g_rms, slow = self.getRMSEquationsSym()
            fast = [i for i in range(len(self.x)) if i not in slow]
            F = sp.zeros(len(self.x), 1)
            for rows, indices in ((f_rms, slow), (g_rms, fast)):
                for row, index in zip(rows, indices):
                    F[index] = row
            # One CSE pass for [dF/dx, dF/du], split afterwards.
            Ax, Bu = splitColumns(*cseJacobian(F, list(self.x) + list(self.u), intermediates=self._intermediates), len(self.x))
            self._modelEntry["rms"] = {
                "differential": tuple(slow), "F_sym": sp.ImmutableMatrix(F),
                "A_cse": Ax, "B_cse": Bu,
                "FFunction": self._lambdify(F),
                "AFunction": self._lambdifyCse(*Ax),
                "BFunction": self._lambdifyCse(*Bu),
            }
        entry = self._modelEntry["rms"]
        differential = entry["differential"]
        return {
            "differentialIndices": np.asarray(differential, dtype=int),
            "algebraicIndices": np.asarray(
                [i for i in range(len(self.x)) if i not in differential], dtype=int),
            "F_sym": entry["F_sym"],
            "A_cse": entry["A_cse"], "B_cse": entry["B_cse"],
            "F": self._wrap(entry["FFunction"]),
            "A": self._wrap(entry["AFunction"], matrix=True),
            "B": self._wrap(entry["BFunction"], matrix=True),
        }

    def _lambdify(self, expression: sp.Matrix):
        """Compile an expression; arguments are states, references, then parameters."""
        arguments = list(self.x) + list(self.u) + list(self.parameterMap["parameterSymbols"])
        if not self._intermediates:
            return sp.lambdify(arguments, expression, modules="numpy", cse=True, docstring_limit=0)
        # Reduced bus voltages (localDAEreduction) are computed first from their definitions.
        # One definition at a time: a definition may use the symbols defined before it.
        head, names = [], sp.numbered_symbols("_d", real=True)
        for symbol, definition in self._intermediates:
            replacements, reducedDefinition = sp.cse([definition], symbols=names)
            head += replacements + [(symbol, reducedDefinition[0])]
        replacements, reduced = sp.cse(sp.Matrix(expression), symbols=sp.numbered_symbols("_c", real=True))
        return lambdifyCse(arguments, head + replacements, reduced[0])

    def _lambdifyCse(self, replacements, expression: sp.Matrix):
        """Compile a CSE-form expression (see :func:`lambdifyCse`); same arguments as :meth:`_lambdify`."""
        arguments = list(self.x) + list(self.u) + list(self.parameterMap["parameterSymbols"])
        return lambdifyCse(arguments, replacements, expression)

    def _compile(self, expression: sp.Matrix, matrix: bool = False):
        return self._wrap(self._lambdify(expression), matrix=matrix)

    def _wrap(self, function: Any, matrix: bool = False):
        """Bind a compiled function to this grid's current parameter values."""
        def numeric(state: Any, *reference: Any):
            stateValues = np.asarray(state, dtype=float).reshape(-1)
            if len(reference) == 1 and np.asarray(reference[0]).size == len(self.u):
                referenceValues = np.asarray(reference[0], dtype=float).reshape(-1)
            else:
                referenceValues = np.asarray(reference, dtype=float).reshape(-1)
            values = np.r_[stateValues, referenceValues, self.parameterMap["parameterValues"]]
            result = np.asarray(function(*values), dtype=float)
            return result if matrix else result.reshape(-1)
        return numeric

    def refreshParameters(self, rebuild: bool = False) -> None:
        symbols: list[Any] = list(self.Lsym) + list(self.Rsym)
        values: list[float] = list(self.L) + list(self.R)
        for cluster in self.clusters:
            for column in cluster["columns"]:
                for key in ("Lsym", "Rsym"):
                    symbols.append(cluster[key][column])
                    values.append(self.p[str(cluster[key][column])]["num"])
        for devices in (self.Vdev, self.Idev, [device for device, _ in self.attached]):
            for device in devices:
                for parameter in device.p.values():
                    symbols.extend(np.asarray(parameter["sym"], dtype=object).reshape(-1, order="F"))
                    values.extend(np.asarray(parameter["num"], dtype=float).reshape(-1, order="F"))
        valuesArray = np.asarray(values, dtype=float)
        self.parameterMap = {"parameterSymbols": tuple(symbols), "parameterValues": valuesArray}

    def getStateMatrix(self, x0: Any, uref: Any) -> np.ndarray:
        self.refreshParameters()
        return np.asarray(self.A_core(x0, uref), dtype=float)

    def analyzeSmallSignal(self, x0: Any, uref: Any, wBase: float) -> dict[str, Any]:
        A = self.getStateMatrix(x0, uref)
        poles, leftVectors, rightVectors = eig(A, left=True, right=True)
        for i in range(len(poles)):
            pairProduct = np.vdot(leftVectors[:, i], rightVectors[:, i])
            if abs(pairProduct) > 1e-10:
                leftVectors[:, i] /= np.conj(pairProduct)
        participation = np.abs(rightVectors * np.conj(leftVectors))
        participation /= np.maximum(participation.sum(axis=0, keepdims=True), np.finfo(float).eps)
        order = np.argsort(poles.real)[::-1]
        poles, participation = poles[order], participation[:, order]
        rightVectors, leftVectors = rightVectors[:, order], leftVectors[:, order]
        stateNames = [str(symbol) for symbol in self.x]
        dominant = participation.argmax(axis=0)
        tolerance = 1e-7
        modeTable = []
        for i, pole in enumerate(poles):
            stability = "Unstable" if pole.real > tolerance else "Marginal" if abs(pole.real) <= tolerance else "Stable"
            modeTable.append({
                "Mode": i + 1, "Eigenvalue": pole,
                "Frequency_Hz": abs(pole.imag) * wBase / (2 * np.pi),
                "DampingRatio": -pole.real / max(abs(pole), np.finfo(float).eps),
                "Stability": stability, "DominantState": stateNames[dominant[i]],
                "DominantParticipation": participation[dominant[i], i],
                "RealPart": pole.real, "ImaginaryPart": pole.imag,
            })
        return {
            "A": A, "poles": poles, "participation": participation,
            "stateNames": stateNames, "modeTable": modeTable,
            "unstablePoleCount": int(np.sum(poles.real > tolerance)),
            "poleTolerance": tolerance, "rightEigenvectors": rightVectors,
            "leftEigenvectors": leftVectors, "leadingModeTable": modeTable[:12],
        }

    def solvePowerFlow(self, vPref: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Power flow on the whole network: retained nodes, then ``floatingNodes``.

        The cluster reduction only concerns the EMT model; here every current device
        sits on its own node. ``V``/``theta`` cover all power-flow nodes,
        ``busVoltage`` the retained nodes and ``floatingVoltage`` the others;
        ``Pdevice``/``Qdevice`` follow ``Vdev + Idev + attached``.
        """
        vPref = np.asarray(vPref, dtype=float).reshape(-1)
        devices = self.Vdev + self.Idev + [device for device, _ in self.attached]
        deviceNode = np.r_[np.arange(self.n), np.arange(self.n), [node for _, node in self.attached]].astype(int)
        nodeCount = self.pfA.shape[0]
        Z = np.r_[self.R + 1j * self.L, [
            self.p[str(cluster["Rsym"][column])]["num"] + 1j * self.p[str(cluster["Lsym"][column])]["num"]
            for cluster in self.clusters for column in cluster["columns"]
        ]]
        Y = self.pfA @ np.diag(1 / Z) @ self.pfA.T
        P, Q, V, theta = sp.symbols("P Q V theta", real=True)
        deviceFunctions = []
        for device in devices:
            powerScale = device.Outputscale / device.Inputscale
            localVoltage = V / device.Outputscale if device.type == "Voltage" else device.Inputscale * V
            equation = sp.Matrix(device.getPowerFlowEquations(
                localVoltage, theta, P / powerScale, Q / powerScale
            ))
            equation = equation.subs({symbol: vPref[self.u.index(symbol)] for symbol in device.uref})
            deviceFunctions.append(sp.lambdify((P, Q, V, theta), equation, "numpy"))
        nd = len(devices)

        def residual(z: np.ndarray) -> np.ndarray:
            magnitude, angle = z[:nodeCount], z[nodeCount:2 * nodeCount]
            active = z[2 * nodeCount:2 * nodeCount + nd]
            reactive = z[2 * nodeCount + nd:]
            voltage = magnitude * np.exp(1j * angle)
            networkPower = voltage * np.conj(Y @ voltage)
            activeNode = np.bincount(deviceNode, weights=active, minlength=nodeCount)
            reactiveNode = np.bincount(deviceNode, weights=reactive, minlength=nodeCount)
            parts = [networkPower.real - activeNode, networkPower.imag - reactiveNode]
            for d, function in enumerate(deviceFunctions):
                parts.append(np.asarray(
                    function(active[d], reactive[d], magnitude[deviceNode[d]], angle[deviceNode[d]]),
                    dtype=float,
                ).reshape(-1))
            return np.concatenate(parts)

        z0 = np.r_[np.ones(nodeCount), np.zeros(nodeCount + 2 * nd)]
        solution = root(residual, z0, method="hybr", options={"xtol": 1e-10})
        z = solution.x
        magnitude, angle = z[:nodeCount], z[nodeCount:2 * nodeCount]
        active = z[2 * nodeCount:2 * nodeCount + nd]
        reactive = z[2 * nodeCount + nd:]
        voltage = magnitude * np.exp(1j * angle)
        info = {
            "exitflag": 1 if solution.success else -1, "success": bool(solution.success),
            "output": {"message": solution.message, "nfev": solution.nfev},
            "residualInfNorm": float(np.linalg.norm(residual(z), np.inf)),
            "busVoltage": voltage[:self.n], "floatingVoltage": voltage[self.n:],
            "branchCurrent": (self.A.T @ voltage[:self.n]) / Z[:self.b],
            "floatingBranchCurrent": (self.pfA[:, self.b:].T @ voltage) / Z[self.b:],
            "V": magnitude, "theta": angle, "Pdevice": active, "Qdevice": reactive,
            "deviceNode": deviceNode,
        }
        return z, info

    def getOperatingPoints(self, vPref: Any) -> tuple[np.ndarray, dict[str, Any]]:
        _, powerFlowInfo = self.solvePowerFlow(vPref)
        V, theta = powerFlowInfo["V"], powerFlowInfo["theta"]
        active, reactive = powerFlowInfo["Pdevice"], powerFlowInfo["Qdevice"]
        clusterCurrent = dict(zip(self.floatingColumns, powerFlowInfo["floatingBranchCurrent"]))
        branchCurrent = np.r_[powerFlowInfo["branchCurrent"], [
            clusterCurrent[column] for cluster in self.clusters for column in cluster["kept"]]]
        xInit = np.zeros(len(self.x))
        xInit[:self.branchStateCount] = np.column_stack((branchCurrent.real, branchCurrent.imag)).reshape(-1)
        nextState = self.branchStateCount
        # Devices in state order with their index in Pdevice/Qdevice and their node.
        ordered = [item for i, (Vdev, Idev) in enumerate(zip(self.Vdev, self.Idev))
                   for item in ((Vdev, i, i), (Idev, self.n + i, i))]
        ordered += [(device, 2 * self.n + k, node) for k, (device, node) in enumerate(self.attached)]
        for device, index, node in ordered:
            powerScale = device.Outputscale / device.Inputscale
            localV = V[node] / device.Outputscale if device.type == "Voltage" else device.Inputscale * V[node]
            deviceState = np.asarray(device.initFromPowerFlow(
                localV, theta[node], active[index] / powerScale,
                reactive[index] / powerScale,
            ), dtype=float).reshape(-1)
            xInit[nextState:nextState + len(device.x)] = deviceState
            nextState += len(device.x)
        self.refreshParameters()
        solution = least_squares(
            lambda state: self.f_init(state, vPref), xInit,
            method="lm" if self._f_init_residual.rows >= len(xInit) else "trf",
            max_nfev=50_000, ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        info = {
            "powerFlow": powerFlowInfo, "initialState": xInit,
            "exitflag": 1 if solution.success else -1,
            "output": {"message": solution.message, "nfev": solution.nfev},
            "residualInfNorm": float(np.linalg.norm(solution.fun, np.inf)),
        }
        return solution.x, info
