"""Systems described by a SPICE netlist instead of a hand-written ``System`` subclass.

The format follows SPICE3 / ngspice / HSPICE wherever a standard card exists:

* line 1 is the title; ``*`` comment lines, ``;`` and `` $ `` inline comments,
  ``+`` continuation; cards, names, nodes and parameters are case-insensitive;
  ground is node ``0`` (alias ``gnd``); ``.end`` ends the netlist.
* numbers take the suffixes ``f p n u m k meg g t`` (``mil``); trailing unit
  letters are ignored (``10uF``, ``4.8ohm``); ``.param`` values and ``{...}`` or
  ``'...'`` expressions use ``sqrt exp log sin cos tan atan atan2 abs min max
  pow hypot`` and ``pi``.
* ``Rname n1 n2 value`` (ohm), ``Lname n1 n2 value`` (H), ``Cname n 0 value``
  (F); ``Xname n+ 0 Model par=val ...`` instantiates a device; ``PWL(t1 v1 ...)``
  is a reference waveform; ``.include file`` inserts netlist text;
  ``.tran tstep tstop [tstart [tmax]]``; ``.options reltol= abstol= method=``.
* ``.meas tran name WHEN expr=value RISE=1|FALL=1|CROSS=1 [FROM=t]`` is the
  state-triggered event; ``.probe tran name=par('expr')`` adds output signals.
  Expressions can read ``V(n)``, ``VM(n)``, ``VP(n)`` (rad, relative to the
  reference bus), ``I(Rname|Lname)``, ``IM(...)``, device states
  ``@xname[state]`` and ``time``.

CodeSim extensions:

* ``.options sbase= vbase= freq= residualtol= units=si|pu``: system base in VA,
  line-line V and Hz, initialization tolerance, and the unit of R/L/C values
  (``pu``: per unit on the system base, L and C as reactance/susceptance).
* ``.hdl file.py`` imports a Python file so its ``@device`` classes can be used.
* An ``X`` model is a built-in component name (``codesimpy.components``), a
  ``@device`` class name or a dotted ``module.Class`` path.
  Instance parameters: ``Sb``, ``Vb`` (default: system base), any parameter of
  the device and any of its references; references may be ``PWL``.
* ``{name}`` of the ``.meas`` card is the detected event time (``inf`` until the
  event fires) and may be used as a PWL time; unreached knots are ignored, so
  the waveform holds its value until the event.
* ``P(n)``, ``Q(n)``: bus power injection; ``wrap(x)``: angle in (-pi, pi].
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .device import Device
from .rl_grid import RLGrid
from .system import System

_SUFFIXES = {"t": 1e12, "g": 1e9, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
# A number followed by letters (an engineering suffix and/or a unit), not part of a name.
_NUMBER = re.compile(r"(?<![\w.])((?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)([a-df-z][a-z]*|e(?![+-]?\d)[a-z]*)")
_REFERENCE = re.compile(r"@(\w+)\[\s*(\w+)\s*\]|\b(vm|vp|v|im|i|p|q)\(\s*([\w.+-]+)\s*\)")
_FUNCTIONS: dict[str, Any] = {
    "__builtins__": {},
    "pi": np.pi, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan": np.arctan, "atan2": np.arctan2,
    "abs": np.abs, "min": np.minimum, "max": np.maximum, "pow": np.power, "hypot": np.hypot,
    "real": np.real, "imag": np.imag, "angle": np.angle,
    "wrap": lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi,
}
# Names the time-domain expressions (.meas, .probe) are evaluated with.
_RUNTIME_NAMES = {"S", "V", "IB", "SB", "IDX", "NODE", "BR", "REF", "time"}
_METHODS = {
    "rk45": "RK45", "rk23": "RK23", "dop853": "DOP853", "radau": "Radau",
    "bdf": "BDF", "gear": "BDF", "lsoda": "LSODA",
}
_NUMERIC_OPTIONS = ("sbase", "vbase", "freq", "reltol", "abstol", "residualtol")


class NetlistSystem(System):
    """A system whose network, devices, references, event and signals come from a netlist.

    Subclass it with the netlist as a class keyword (the file path is relative to
    the module that defines the class)::

        class MySMIB(NetlistSystem, netlistFile="smib.cir"):
            pass

    or ``netlist="..."`` for netlist text. ``.cir`` files in a workspace become
    such classes automatically (see :func:`codesimpy.registry.discover`).
    """

    def __init_subclass__(cls, netlist: str | None = None, netlistFile: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if netlist is None and netlistFile is None:
            return  # a subclass of a netlist class keeps its parent's netlist
        module = sys.modules.get(cls.__module__)
        directory = Path(module.__file__).parent if getattr(module, "__file__", None) else Path.cwd()
        if netlistFile is not None:
            path = Path(netlistFile)
            path = (path if path.is_absolute() else directory / path).resolve()
            filename, directory = str(path), path.parent
            netlist = path.read_text(encoding="utf-8")
        else:
            filename = f"<{cls.__name__} netlist>"
        lines = netlist.splitlines()
        start = next((i for i, line in enumerate(lines) if line.strip()), len(lines))
        title = lines[start].strip() if start < len(lines) else ""
        if title.startswith("*"):
            title = title.lstrip("*").strip()

        # 1. Lines -> cards: comments, .include expansion, '+' continuation, .end.
        # Each line keeps (file, line number, text, chain of files including it).
        pending = [(filename, number, line, (filename,))
                   for number, line in enumerate(lines[start + 1:], start + 2)]
        cards: list[list[Any]] = []
        position = 0
        while position < len(pending):
            file, number, line, chain = pending[position]
            position += 1
            line = re.sub(r";.*$|\s\$\s.*$", "", line).strip()
            if not line or line.startswith("*"):
                continue
            if line.startswith("+"):
                if not cards:
                    raise SyntaxError("continuation line '+' without a card", (file, number, 0, None))
                cards[-1][2] += " " + line[1:]
                continue
            keyword = line.split()[0].lower()
            if keyword == ".end":
                break
            if keyword in (".include", ".inc"):
                target = line.split(None, 1)[1].strip().strip("'\"") if len(line.split()) > 1 else ""
                if not target:
                    raise SyntaxError(".include needs a file name", (file, number, 0, None))
                includePath = Path(target)
                base = Path(file).parent if Path(file).is_file() else directory
                includePath = (includePath if includePath.is_absolute() else base / includePath).resolve()
                if str(includePath) in chain:
                    raise SyntaxError(f".include cycle: {includePath}", (file, number, 0, None))
                try:
                    included = includePath.read_text(encoding="utf-8").splitlines()
                except OSError as error:
                    raise SyntaxError(f"cannot read .include file: {error}", (file, number, 0, None)) from None
                pending[position:position] = [
                    (str(includePath), includedNumber, includedLine, (*chain, str(includePath)))
                    for includedNumber, includedLine in enumerate(included, 1)
                ]
                continue
            cards.append([file, number, line])

        # 2. Cards -> structure. Every value is a source string, compiled in step 3.
        sources: list[tuple[str, str, int]] = []
        options: dict[str, Any] = {}
        params: list[tuple[str, int]] = []
        tran: list[int] = []
        hdl: list[Path] = []
        elements: list[dict[str, Any]] = []
        meas: dict[str, Any] | None = None
        probes: list[tuple[str, int]] = []
        for file, number, text in cards:
            tokens, current, depth, quoted = [], "", 0, False
            for char in text + " ":
                if char == "'":
                    quoted = not quoted
                elif not quoted and char in "({[":
                    depth += 1
                elif not quoted and char in ")}]":
                    depth -= 1
                if quoted or depth or char == "'" or char not in " \t,=":
                    current += char
                    continue
                if current:
                    tokens.append(current)
                    current = ""
                if char == "=":
                    tokens.append("=")
            if quoted or depth:
                raise SyntaxError("unbalanced quote or bracket", (file, number, 0, None))
            positional, pairs, index = [], {}, 0
            while index < len(tokens):
                if index + 2 < len(tokens) and tokens[index + 1] == "=":
                    pairs[tokens[index].lower()] = (tokens[index], tokens[index + 2])
                    index += 3
                elif tokens[index] == "=":
                    raise SyntaxError("'=' without a name or a value", (file, number, 0, None))
                else:
                    positional.append(tokens[index])
                    index += 1
            card = positional[0].lower() if positional else ""
            if card in (".options", ".option", ".opt"):
                for key, (raw, value) in pairs.items():
                    if key in ("method", "units"):
                        options[key] = value.strip("'\"").lower()
                        if key == "method" and options[key] not in _METHODS:
                            raise SyntaxError(f"unknown method {value!r}; use one of {', '.join(_METHODS)}", (file, number, 0, None))
                        if key == "units" and options[key] not in ("si", "pu"):
                            raise SyntaxError("units must be 'si' or 'pu'", (file, number, 0, None))
                    elif key in _NUMERIC_OPTIONS:
                        sources.append((value, file, number))
                        options[key] = len(sources) - 1
                    else:
                        raise SyntaxError(
                            f"unknown option {raw!r}; known: {', '.join((*_NUMERIC_OPTIONS, 'method', 'units'))}", (file, number, 0, None))
            elif card in (".param", ".params"):
                if len(positional) > 1:
                    raise SyntaxError(".param expects name=value pairs", (file, number, 0, None))
                for key, (raw, value) in pairs.items():
                    sources.append((value, file, number))
                    params.append((raw, len(sources) - 1))
            elif card == ".title":
                title = text.split(None, 1)[1].strip() if len(text.split()) > 1 else ""
            elif card == ".hdl":
                if len(positional) != 2:
                    raise SyntaxError(".hdl needs one file name", (file, number, 0, None))
                hdlPath = Path(positional[1].strip("'\""))
                base = Path(file).parent if Path(file).is_file() else directory
                hdl.append((hdlPath if hdlPath.is_absolute() else base / hdlPath).resolve())
            elif card == ".tran":
                values = [token for token in positional[1:] if token.lower() != "uic"]
                if not 2 <= len(values) <= 4 or pairs:
                    raise SyntaxError(".tran expects: tstep tstop [tstart [tmax]]", (file, number, 0, None))
                for value in values:
                    sources.append((value, file, number))
                    tran.append(len(sources) - 1)
            elif card in (".meas", ".measure"):
                if meas is not None:
                    raise SyntaxError("only one .meas ... WHEN event is supported", (file, number, 0, None))
                equals = tokens.index("=") if "=" in tokens else -1
                if len(tokens) < 7 or tokens[1].lower() != "tran" or tokens[3].lower() != "when" or equals < 5:
                    raise SyntaxError(".meas expects: .meas tran name WHEN expr=value [RISE=1|FALL=1|CROSS=1] [FROM=t]", (file, number, 0, None))
                name = tokens[2]
                sources.append((f"({' '.join(tokens[4:equals])}) - ({tokens[equals + 1]})", file, number))
                meas = {"name": name, "code": len(sources) - 1, "direction": 1.0, "from": None,
                        "file": file, "line": number}
                rest, index = tokens[equals + 2:], 0
                while index < len(rest):
                    if index + 2 >= len(rest) or rest[index + 1] != "=":
                        raise SyntaxError(f"unexpected {rest[index]!r} in .meas", (file, number, 0, None))
                    key, value = rest[index].lower(), rest[index + 2]
                    index += 3
                    if key in ("rise", "fall", "cross"):
                        if value != "1":
                            raise SyntaxError(f"only {key.upper()}=1 (the first crossing) is supported", (file, number, 0, None))
                        meas["direction"] = {"rise": 1.0, "fall": -1.0, "cross": 0.0}[key]
                    elif key == "from":
                        sources.append((value, file, number))
                        meas["from"] = len(sources) - 1
                    else:
                        raise SyntaxError(f"unknown .meas keyword {key.upper()}", (file, number, 0, None))
            elif card in (".probe", ".save"):
                if positional[1:2] and positional[1].lower() == "tran":
                    positional = positional[:1] + positional[2:]
                outputs = [(raw, value) for raw, value in pairs.values()]
                outputs += [(token, token) for token in positional[1:]]
                if not outputs:
                    raise SyntaxError(f"{card} needs at least one output", (file, number, 0, None))
                for raw, value in outputs:
                    if value.lower().startswith("par(") and value.endswith(")"):
                        value = value[4:-1]
                    sources.append((value, file, number))
                    probes.append((raw, len(sources) - 1))
            elif card.startswith("."):
                raise SyntaxError(f"unsupported card {positional[0]}", (file, number, 0, None))
            elif card[:1] in ("r", "l", "c"):
                if len(positional) != 4 or pairs:
                    raise SyntaxError(f"{positional[0]} expects: name node1 node2 value", (file, number, 0, None))
                sources.append((positional[3], file, number))
                elements.append({"kind": card[0], "name": positional[0], "nodes": positional[1:3],
                                 "value": len(sources) - 1, "file": file, "line": number})
            elif card[:1] == "x":
                if positional[4:5] and positional[4].lower() == "params:":
                    positional = positional[:4] + positional[5:]
                if len(positional) != 4:
                    raise SyntaxError(f"{positional[0]} expects: name node+ node- Model par=value ...", (file, number, 0, None))
                instance: dict[str, Any] = {}
                for key, (raw, value) in pairs.items():
                    if value.lower().startswith("pwl(") and value.endswith(")"):
                        knots = re.findall(r"\{[^{}]*\}|'[^']*'|[^\s,{}']+", value[4:-1])
                        if len(knots) < 2 or len(knots) % 2:
                            raise SyntaxError(f"PWL of {raw} needs time-value pairs", (file, number, 0, None))
                        for knot in knots:
                            sources.append((knot, file, number))
                        indices = range(len(sources) - len(knots), len(sources))
                        instance[key] = (raw, list(zip(indices[0::2], indices[1::2])))
                    else:
                        sources.append((value, file, number))
                        instance[key] = (raw, len(sources) - 1)
                elements.append({"kind": "x", "name": positional[0], "nodes": positional[1:3],
                                 "model": positional[3], "params": instance, "file": file, "line": number})
            else:
                raise SyntaxError(f"unsupported element {positional[0]} (use R, L, C or X)", (file, number, 0, None))

        names = [element["name"].lower() for element in elements]
        for element in elements:
            if names.count(element["name"].lower()) > 1:
                raise SyntaxError(f"duplicate element name {element['name']}", (element["file"], element["line"], 0, None))
        if meas is not None and meas["name"].lower() in {name.lower() for name, _ in params}:
            raise SyntaxError(f".meas name {meas['name']} is also a .param", (meas["file"], meas["line"], 0, None))

        # 3. Compile every source: strip {...} / '...', protect output references,
        # scale suffixed numbers, then turn the references into array lookups.
        codes = []
        for text, file, number in sources:
            text = text.strip()
            if (text[:1], text[-1:]) in (("{", "}"), ("'", "'")):
                text = text[1:-1]
            text = text.lower().replace("^", "**")
            pieces, references, position = [], [], 0
            for match in [*_REFERENCE.finditer(text), None]:
                end = match.start() if match else len(text)
                pieces.append(_NUMBER.sub(lambda number: (
                    repr(float(number[1]) * (1e6 if number[2].startswith("meg") else 25.4e-6))
                    if number[2].startswith(("meg", "mil"))
                    else repr(float(number[1]) * _SUFFIXES[number[2][0]]) if number[2][0] in _SUFFIXES
                    else number[1]
                ), text[position:end]))
                if match is None:
                    break
                position = match.end()
                if match[1]:
                    key = f"{match[1]}.{match[2]}"
                    references.append(("state", key))
                    pieces.append(f"S[..., IDX[{key!r}]]")
                    continue
                function, argument = match[3], match[4]
                if function in ("i", "im"):
                    references.append(("branch", argument))
                    lookup = f"IB[..., BR[{argument!r}]]"
                    pieces.append(f"real({lookup})" if function == "i" else f"abs({lookup})")
                    continue
                argument = "0" if argument == "gnd" else argument
                references.append(("node", argument))
                lookup = f"V[..., NODE[{argument!r}]]"
                pieces.append({
                    "v": f"real({lookup})", "vm": f"abs({lookup})",
                    "vp": f"angle({lookup} / V[..., REF])",
                    "p": f"real(SB[..., NODE[{argument!r}]])", "q": f"imag(SB[..., NODE[{argument!r}]])",
                }[function])
            try:
                code = compile("".join(pieces), file, "eval")
            except SyntaxError as error:
                raise SyntaxError(f"invalid expression {text!r}: {error.msg}", (file, number, 0, None)) from None
            codes.append((code, references, file, number))

        # .hdl: import user device files so their @device models register.
        from .. import registry

        for path in hdl:
            loaded = any(
                getattr(module, "__file__", None) and Path(module.__file__).resolve() == path
                for module in list(sys.modules.values())
            )
            if not loaded:
                registry.import_user_file(path, path.parent)

        cls._filename, cls._title, cls._options, cls._params = filename, title, options, params
        cls._tran, cls._elements, cls._meas, cls._probes, cls._codes = tran, elements, meas, probes, codes

    @classmethod
    def defaultParameters(cls, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """``.param`` values (``overrides`` replace them before dependent values are
        evaluated), then the derived run settings the frontend needs."""
        if not hasattr(cls, "_codes"):
            raise TypeError(f"{cls.__name__} has no netlist: class X(NetlistSystem, netlistFile=...)")
        overrides = dict(overrides or {})
        env, p = dict(_FUNCTIONS), {}
        eventName = cls._meas["name"].lower() if cls._meas else None
        code, file, number = None, cls._filename, 0
        try:
            for name, index in cls._params:
                code, _, file, number = cls._codes[index]
                p[name] = overrides.pop(name) if name in overrides else float(np.real(eval(code, env)))
                env[name.lower()] = p[name]
            option = {}
            for key in _NUMERIC_OPTIONS:
                if key in cls._options:
                    code, _, file, number = cls._codes[cls._options[key]]
                    option[key] = float(eval(code, env))
            times = []
            for index in cls._tran:
                code, _, file, number = cls._codes[index]
                times.append(float(eval(code, env)))
            breakpoints = []
            for element in cls._elements:
                for raw, value in element.get("params", {}).values():
                    if isinstance(value, list):
                        for timeIndex, _ in value:
                            code, _, file, number = cls._codes[timeIndex]
                            if eventName not in code.co_names:
                                breakpoints.append(float(eval(code, env)))
            if cls._meas and cls._meas["from"] is not None:
                code, _, file, number = cls._codes[cls._meas["from"]]
                breakpoints.append(float(eval(code, env)))
        except NameError as error:
            raise SyntaxError(f"{error}", (file, number, 0, None)) from None
        if times[2:3] and times[2] != 0:
            raise SyntaxError(".tran tstart must be 0", (file, number, 0, None))
        p["frequency_Hz"] = option.get("freq", 50.0)
        p["wBase"] = 2 * np.pi * p["frequency_Hz"]
        p["validationTime_s"] = times[1] if times else 1.0
        p["outputStep_s"] = times[0] if times and times[0] > 0 else 1e-4
        if times[3:4]:
            p["maxStep_s"] = times[3]
        p["residualTolerance"] = option.get("residualtol", 1e-4)
        p.update(overrides)
        p["breakpoints_s"] = sorted({
            time for time in breakpoints if 0 < time < p["validationTime_s"] and np.isfinite(time)
        })
        return p

    def __init__(self, p: dict[str, Any] | None = None) -> None:
        cls = type(self)
        p = cls.defaultParameters() if p is None else p
        env = {**_FUNCTIONS, **{str(name).lower(): value for name, value in p.items()}}
        eventName = cls._meas["name"].lower() if cls._meas else None
        for code, _, file, number in cls._codes:
            unknown = set(code.co_names) - env.keys() - _RUNTIME_NAMES - {eventName}
            if unknown:
                raise SyntaxError(f"unknown name(s): {', '.join(sorted(unknown))}", (file, number, 0, None))
        option = {key: float(eval(cls._codes[cls._options[key]][0], env))
                  for key in _NUMERIC_OPTIONS if key in cls._options}
        sbase, vbase = option.get("sbase", 1.0), option.get("vbase", 1.0)
        wBase = 2 * np.pi * p["frequency_Hz"]
        perUnit = cls._options.get("units", "si") == "pu"
        impedanceBase = vbase**2 / sbase

        from .. import components, registry

        # Buses in order of first appearance; ground is a bus only when a branch reaches it.
        busNames, busIndex = [], {}
        for element in cls._elements:
            nodes = [node.lower() for node in element["nodes"]]
            nodes = ["0" if node == "gnd" else node for node in nodes]
            element["keys"] = nodes
            if element["kind"] in ("c", "x"):
                if element["kind"] == "x" and nodes[1] != "0":
                    raise SyntaxError(f"{element['name']}: node- must be 0 (devices connect a bus to ground)", (element["file"], element["line"], 0, None))
                if nodes.count("0") != 1:
                    raise SyntaxError(f"{element['name']}: needs exactly one node at ground 0", (element["file"], element["line"], 0, None))
                nodes = [node for node in nodes if node != "0"]
            elif nodes[0] == nodes[1]:
                raise SyntaxError(f"{element['name']}: both ends on node {element['nodes'][0]}", (element["file"], element["line"], 0, None))
            for node, raw in zip(nodes, element["nodes"]):
                if node not in busIndex:
                    busIndex[node] = len(busNames)
                    # The frontend draws a bus named "ground" as a ground symbol per element.
                    busNames.append("ground" if node == "0" else raw)
        count = len(busNames)

        # Branches: one R and one L per node pair, merged into one RL branch.
        branches: dict[tuple[str, str], dict[str, Any]] = {}
        capacitance, capacitorLine = np.zeros(count), [None] * count
        voltageDevices, currentDevices = [None] * count, [None] * count
        deviceLine = [None] * count
        references: dict[str, Any] = {}
        devices: dict[str, Any] = {}
        for element in cls._elements:
            kind, name, file, number = element["kind"], element["name"], element["file"], element["line"]
            if kind in ("r", "l", "c"):
                value = float(eval(cls._codes[element["value"]][0], env))
                if value < 0 or (kind != "r" and value == 0):
                    raise SyntaxError(f"{name}: value must be positive", (file, number, 0, None))
            if kind in ("r", "l"):
                key = tuple(sorted(element["keys"]))
                branch = branches.setdefault(key, {"nodes": element["keys"], "R": None, "L": None,
                                                   "names": [], "file": file, "line": number})
                if branch[kind.upper()] is not None:
                    raise SyntaxError(f"{name}: a second {kind.upper()} between the same nodes; "
                                       "only one R and one L per node pair", (file, number, 0, None))
                if not perUnit:
                    value = value / impedanceBase if kind == "r" else wBase * value / impedanceBase
                branch[kind.upper()] = value
                branch["names"].append(name.lower())
            elif kind == "c":
                node = busIndex[next(node for node in element["keys"] if node != "0")]
                capacitance[node] += value if perUnit else wBase * value * impedanceBase
                capacitorLine[node] = capacitorLine[node] or (file, number)
            else:
                model = element["model"]
                if "." in model:
                    moduleName, _, className = model.rpartition(".")
                    try:
                        deviceClass = getattr(importlib.import_module(moduleName), className, None)
                    except ImportError as error:
                        raise SyntaxError(f"{name}: cannot import {moduleName}: {error}", (file, number, 0, None)) from None
                else:
                    # User @device models first, then the built-in components by class name.
                    builtins = {className.lower(): getattr(components, className) for className in components.__all__}
                    deviceClass = registry.DEVICES.get(model.lower(), builtins.get(model.lower()))
                if not (isinstance(deviceClass, type) and issubclass(deviceClass, Device)):
                    known = ", ".join(sorted({*components.__all__, *(
                        getattr(device, "__codesim_device__", device.__name__) for device in registry.DEVICES.values())}))
                    raise SyntaxError(f"{name}: unknown device model {model!r}; known models: {known}", (file, number, 0, None))
                instance = element["params"]
                deviceBase = {}
                for key, default in (("vb", vbase), ("sb", sbase)):
                    deviceBase[key] = default
                    if key in instance:
                        if isinstance(instance[key][1], list):
                            raise SyntaxError(f"{name}: {instance[key][0]} cannot be a PWL", (file, number, 0, None))
                        deviceBase[key] = float(eval(cls._codes[instance[key][1]][0], env))
                index = re.sub(r"\W", "_", name.lower())
                device = deviceClass(deviceBase["vb"], deviceBase["sb"], index)
                parameterNames = {parameter.lower(): parameter for parameter in device.p}
                referenceNames = {str(symbol)[:-len(device.index)].lower(): str(symbol) for symbol in device.uref}
                for key, (raw, value) in instance.items():
                    if key in ("vb", "sb"):
                        continue
                    if key in referenceNames:
                        if isinstance(value, list):
                            knots = [(cls._codes[a][0], cls._codes[b][0]) for a, b in value]
                            if any(eventName in code.co_names for knot in knots for code in knot):
                                # Knots at the event time move once the event fires; inf knots are not reached yet.
                                references[referenceNames[key]] = lambda t, knots=knots: float(np.interp(t, *zip(*[
                                    (x, y) for x, y in (
                                        (eval(a, env, {eventName: self.eventTime_s}),
                                         eval(b, env, {eventName: self.eventTime_s})) for a, b in knots
                                    ) if np.isfinite(x)
                                ])))
                            else:
                                xs = np.asarray([float(eval(a, env)) for a, _ in knots])
                                ys = np.asarray([float(eval(b, env)) for _, b in knots])
                                if np.any(np.diff(xs) < 0):
                                    raise SyntaxError(f"{name}: PWL times of {raw} must not decrease", (file, number, 0, None))
                                references[referenceNames[key]] = lambda t, xs=xs, ys=ys: float(np.interp(t, xs, ys))
                        else:
                            references[referenceNames[key]] = float(eval(cls._codes[value][0], env))
                    elif key in parameterNames:
                        if isinstance(value, list):
                            raise SyntaxError(f"{name}: PWL is only allowed for references "
                                               f"({', '.join(sorted(referenceNames))})", (file, number, 0, None))
                        device.p[parameterNames[key]]["num"] = float(eval(cls._codes[value][0], env))
                    else:
                        raise SyntaxError(
                            f"{name}: {model} has no parameter or reference {raw!r}; parameters: "
                            f"{', '.join(device.p)}; references: {', '.join(sorted(referenceNames))}", (file, number, 0, None))
                missing = [symbol for symbol in referenceNames.values() if symbol not in references]
                if missing:
                    raise SyntaxError(
                        f"{name}: missing reference(s) "
                        f"{', '.join(symbol[:-len(device.index)] for symbol in missing)}", (file, number, 0, None))
                node = busIndex[element["keys"][0]]
                ports = voltageDevices if device.type == "Voltage" else currentDevices
                if ports[node] is not None:
                    raise SyntaxError(f"{name}: bus {busNames[node]} already has a {device.type.lower()}-port "
                                       "device; a bus takes one voltage and one current device", (file, number, 0, None))
                ports[node] = device
                deviceLine[node] = deviceLine[node] or (file, number)
                devices[name.lower()] = device

        for node in range(count):
            safeName = re.sub(r"\W", "_", busNames[node])
            if capacitance[node]:
                if voltageDevices[node] is not None:
                    raise SyntaxError(
                        f"bus {busNames[node]} has a C and a voltage-port device; "
                        "fold the capacitance into that device's C parameter", (*capacitorLine[node], 0, None))
                voltageDevices[node] = components.Capacitor(vbase, sbase, f"{safeName}_c")
                voltageDevices[node].p["C"]["num"] = capacitance[node]
                voltageDevices[node].p["Gd"]["num"], voltageDevices[node].p["w"]["num"] = 0.0, 1.0
            if node == busIndex.get("0"):
                voltageDevices[node] = components.Infbus(vbase, sbase, "ground")
                voltageDevices[node].p["kA"]["num"] = 100.0
                references[str(voltageDevices[node].uref[0])] = 0.0
            if currentDevices[node] is not None and voltageDevices[node] is None:
                raise SyntaxError(
                    f"bus {busNames[node]} has only a current-port device; it needs a voltage port "
                    "(a C to ground or a voltage-port device)", (*deviceLine[node], 0, None))
            if voltageDevices[node] is not None and currentDevices[node] is None:
                currentDevices[node] = components.ZeroCurrent(
                    vbase, sbase, f"{safeName}_zero")

        branchList = list(branches.values())
        incidence = np.zeros((count, len(branchList)))
        for column, branch in enumerate(branchList):
            incidence[busIndex[branch["nodes"][0]], column] = 1
            incidence[busIndex[branch["nodes"][1]], column] = -1
        branchR = np.asarray([branch["R"] or 0.0 for branch in branchList])
        branchL = np.asarray([branch["L"] or 0.0 for branch in branchList])
        passive = [node for node in range(count) if voltageDevices[node] is None]
        _, reducedL, _ = RLGrid.kronReduction(incidence, branchL, branchR, passive)
        if np.any(reducedL <= 0):
            branch = next(branch for branch in branchList if branch["L"] is None)
            raise SyntaxError(
                f"{branch['names'][0]}: a branch without inductance remains after merging series "
                "elements; add an L in series", (branch["file"], branch["line"], 0, None))

        devices.update({"Vdev": voltageDevices, "Idev": currentDevices})
        network = {"A": incidence, "R": branchR, "L": branchL, "busNames": busNames}
        super().__init__(cls.__name__, network, devices,
                         {"Vbase": vbase, "Sbase": sbase, "frequency_Hz": p["frequency_Hz"]}, references)
        self.solverSettings = {
            "method": _METHODS[cls._options.get("method", "dop853")],
            "max_step": p.get("maxStep_s", 0.1 / wBase),
            "rtol": option.get("reltol", 1e-3),
            "atol": option.get("abstol", 1e-5),
        }
        self.parameters, self.parameterNames = p, list(p)

        # Lookups for the time-domain expressions: state columns, retained buses, branches.
        retained = {original: position for position, original in enumerate(self.retainedNodeIndices)}
        stateIndex = {
            f"{name}.{str(symbol)[:-len(device.index)].lower()}": self.stateNames.index(str(symbol))
            for name, device in devices.items() if name not in ("Vdev", "Idev") for symbol in device.x
        }
        nodeIndex = {node: retained[index] for node, index in busIndex.items() if index in retained}
        survivors = [branch for branch in branchList
                     if busIndex[branch["nodes"][0]] in retained and busIndex[branch["nodes"][1]] in retained]
        branchIndex = {name: rank for rank, branch in enumerate(survivors) for name in branch["names"]}
        maps = {"state": stateIndex, "node": nodeIndex, "branch": branchIndex}
        for _, used, file, number in cls._codes:
            for kind, key in used:
                if key not in maps[kind]:
                    reason = {
                        "state": "no such device state (use @xname[state])",
                        "node": "no such bus, or the node has no device and is eliminated",
                        "branch": "no such R/L, or its branch is merged away with a passive node",
                    }[kind]
                    raise SyntaxError(f"{key}: {reason}", (file, number, 0, None))
        reference = next((position for position, original in enumerate(self.retainedNodeIndices)
                          if isinstance(voltageDevices[original], components.Infbus)
                          and original != busIndex.get("0")), 0)
        self.netlistEnvironment = {**env, "IDX": stateIndex, "NODE": nodeIndex, "BR": branchIndex,
                                   "REF": reference}
        if cls._meas:
            self.eventName = cls._meas["name"]
            self.eventDirection = cls._meas["direction"]
            self.eventStartTime_s = (0.0 if cls._meas["from"] is None
                                     else float(eval(cls._codes[cls._meas["from"]][0], env)))

    def eventHandler(self, time_s: float, state: np.ndarray) -> float:
        """The ``.meas ... WHEN`` expression minus its value; the event fires at its zero crossing."""
        if not type(self)._meas:
            return super().eventHandler(time_s, state)
        code = type(self)._codes[type(self)._meas["code"]][0]
        local = {"S": np.asarray(state), "time": time_s}
        if {"V", "IB", "SB"} & set(code.co_names):
            branch, voltage, current = self.rlGrid.g_numeric(state, self.getReference(time_s))
            local["V"] = np.asarray(voltage).reshape(-1, 2) @ np.array([1, 1j])
            local["IB"] = np.asarray(branch).reshape(-1, 2) @ np.array([1, 1j])
            local["SB"] = local["V"] * np.conj(local["IB"] @ self.rlGrid.A.T)
        return float(np.real(eval(code, self.netlistEnvironment, local)))

    def calculateSignals(self, time_s: Any, state: Any, networkSignals: dict[str, Any]):
        """Bus voltages, angles, powers and line currents, plus the ``.probe`` outputs under ``probes``."""
        state, time_s = np.asarray(state), np.asarray(time_s)
        signals = self.busSignals(networkSignals, np.arange(self.rlGrid.n), self.netlistEnvironment["REF"])
        local = {"S": state, "V": networkSignals["busVoltage"], "IB": networkSignals["branchCurrent"],
                 "SB": networkSignals["busPower"], "time": time_s}
        signals["probes"] = {
            name: np.real(eval(type(self)._codes[index][0], self.netlistEnvironment, local)) * np.ones(len(state))
            for name, index in type(self)._probes
        }
        return signals
