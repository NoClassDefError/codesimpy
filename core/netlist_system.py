"""Systems described by a SPICE netlist instead of a hand-written ``System`` subclass.

The format follows SPICE3 / ngspice / HSPICE / LTspice wherever a standard card exists:

* line 1 is the title; ``*`` comment lines, ``;`` and `` $ `` inline comments,
  ``+`` continuation; cards, names, nodes and parameters are case-insensitive;
  ground is node ``0`` (alias ``gnd``); ``.end`` ends the netlist (optional).
* numbers take the suffixes ``f p n u m k meg g t`` (``mil``), also as a decimal
  point (``6k34`` = 6340); trailing unit letters are ignored (``10uF``, ``4.8ohm``).
* expressions (``.param`` values, ``{...}`` or ``'...'``): ``+ - * / %``, ``**``
  and ``^`` (both power), comparisons ``== != < > <= >=``, logic ``& | !``
  (``&& ||``), ``c ? a : b``; true is a value above 0.5 and yields 1. Functions:
  ``sqrt exp ln log log10 sin cos tan asin acos atan atan2 sinh cosh tanh abs
  min max pow pwr pwrs hypot cbrt limit if u uramp sgn floor ceil int round mod
  buf inv xor table wrap``; constants ``pi boltz echarge planck kelvin gmin``.
* ``.param`` values may be written in any order; ``.func name(a, b) {expr}``
  defines a function of its arguments and the ``.param`` values.
* ``Rname n1 n2 value`` (ohm), ``Lname n1 n2 value`` (H), ``Cname n 0 value``
  (F); ``Lname n1 n2 value R=r`` (HSPICE) or ``Rname n1 n2 value L=l`` is one
  series RL branch per card (parallel branches allowed); plain R and L cards on
  the same node pair merge into one series branch; ``L = 0`` is allowed except
  between two buses with voltage devices; ``Xname n+ 0 Model par=val ...`` instantiates a device; ``PWL(t1 v1 ...)``
  is a reference waveform; ``.include file`` inserts netlist text;
  ``.tran tstep tstop [tstart [tmax]]``; ``.options reltol= abstol= method=``.
* ``.model name Model(par=val ...)`` (or ``ako:other``) names a device with
  default parameters; ``.subckt name pins... [params: a=1]`` ... ``.ends``
  defines a subcircuit of R, L, C, X, ``.param`` and ``.model`` cards, used by
  ``Xname nodes... name a=2``; its elements and nodes are named ``xname:inner``
  (``.global`` nodes and ``0`` are shared). ``.lib file [section]`` reads the
  ``.model .subckt .param .func`` cards of a library (``.lib section`` ...
  ``.endl`` blocks).
* ``.meas tran name WHEN expr=value RISE=1|FALL=1|CROSS=1 [FROM=t]`` is the
  state-triggered event; after a run, ``.meas`` also computes the LTspice forms
  ``FIND|DERIV expr AT=t|WHEN expr=value [TD=t] [RISE|FALL|CROSS=n|LAST]``,
  ``AVG|MAX|MIN|PP|RMS|INTEG expr [FROM=t] [TO=t]``, ``TRIG ... TARG ...`` and
  ``PARAM expr`` (``result["measurements"]``). ``.probe tran name=par('expr')``
  adds output signals.
  Expressions can read ``V(n)``, ``V(n1,n2)``, ``VM(n)``, ``VP(n)`` (rad, relative to the
  reference bus), ``I(Rname|Lname)``, ``IM(...)``, device states
  ``@xname[state]`` and ``time``.
* Analysis and display cards CodeSim does not run (``.ac .dc .noise .op .step``
  ...) and unknown options are skipped with a warning; other elements
  (D, Q, M, K, E, V, ...) are errors.

CodeSim extensions:

* ``.options sbase= vbase= freq= residualtol= units=si|pu``: system base in VA,
  line-line V and Hz, initialization tolerance, and the unit of R/L/C values
  (``pu``: per unit on the system base, L and C as reactance/susceptance).
* ``.hdl file.py`` imports a Python file so its ``@device`` classes can be used.
* An ``X`` model is a ``.model`` or ``.subckt`` name, a built-in component name
  (``codesimpy.components``), a ``@device`` class name or a dotted ``module.Class`` path.
  Instance parameters: ``Sb``, ``Vb`` (default: system base), any parameter of
  the device and any of its references; references may be ``PWL``.
* ``{name}`` of the ``.meas`` event is the detected event time (``inf`` until the
  event fires) and may be used as a PWL time; unreached knots are ignored, so
  the waveform holds its value until the event.
* ``P(n)``, ``Q(n)``: bus power injection; ``wrap(x)``: angle in (-pi, pi].
* A bus without a C or a voltage-port device (with or without current-port
  devices) is removed by ``RLGrid.localDAEreduction``; a bus may hold several
  current-port devices. ``V(n)``, ``P(n)`` and ``I(...)`` work on such buses.
"""

from __future__ import annotations

import importlib
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .device import Device
from .system import System

_SUFFIXES = {"t": 1e12, "g": 1e9, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
# Expression tokens: a number with its suffix / unit tail (and LTspice's 6k34), a name, an operator.
_TOKEN = re.compile(
    r"(?P<number>(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(?P<tail>[a-z]+\d*)?"
    r"|(?P<name>[a-z_]\w*)"
    r"|(?P<op>\*\*|==|!=|<=|>=|&&|\|\||[-+*/%^<>!~&|?:(),{}])"
)
_REFERENCE = re.compile(
    r"@([\w:]+)\[\s*(\w+)\s*\]|\b(vm|vp|v|im|i|p|q)\(\s*([\w.:+-]+)\s*(?:,\s*([\w.:+-]+)\s*)?\)")

# A value is true above 0.5 (LTspice); logic results are 1.0 / 0.0.

_FUNCTIONS: dict[str, Any] = {
    "__builtins__": {},
    "pi": np.pi, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan": np.arctan, "atan2": np.arctan2,
    "asin": np.arcsin, "acos": np.arccos, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "abs": np.abs, "min": np.minimum, "max": np.maximum, "pow": np.power, "hypot": np.hypot,
    "cbrt": np.cbrt, "pwr": lambda x, y: np.abs(x) ** y, "pwrs": lambda x, y: np.sign(x) * np.abs(x) ** y,
    "limit": lambda x, a, b: np.clip(x, np.minimum(a, b), np.maximum(a, b)),
    "u": lambda x: 1.0 * (np.asarray(x) > 0), "uramp": lambda x: np.maximum(x, 0),
    "sgn": np.sign, "floor": np.floor, "ceil": np.ceil, "int": np.trunc, "round": np.round,
    "mod": np.fmod, "buf": lambda x: 1.0 * (np.asarray(x) > 0.5), "inv": lambda x: 1.0 * ~(np.asarray(x) > 0.5),
    "xor": lambda a, b: 1.0 * ((np.asarray(a) > 0.5) ^ (np.asarray(b) > 0.5)),
    "table": lambda x, *pairs: np.interp(x, pairs[0::2], pairs[1::2]),
    "real": np.real, "imag": np.imag, "angle": np.angle,
    "wrap": lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi,
    "boltz": 1.3806503e-23, "echarge": 1.602176462e-19, "planck": 6.62620e-34,
    "kelvin": -273.15, "gmin": 1e-12,
    # Targets of the operators translated by _toPython.
    "_if": lambda condition, a, b: np.where((np.asarray(condition) > 0.5), a, b)[()],
    "_and": lambda a, b: 1.0 * ((np.asarray(a) > 0.5) & (np.asarray(b) > 0.5)),
    "_or": lambda a, b: 1.0 * ((np.asarray(a) > 0.5) | (np.asarray(b) > 0.5)),
    "_not": lambda a: 1.0 * ~(np.asarray(a) > 0.5),
    "_mod": np.fmod,
}
# Names the time-domain expressions (.meas, .probe) are evaluated with.
_RUNTIME_NAMES = {"S", "V", "IB", "SB", "IDX", "NODE", "BR", "REF", "time"}
_METHODS = {
    "rk45": "RK45", "rk23": "RK23", "dop853": "DOP853", "radau": "Radau",
    "bdf": "BDF", "gear": "BDF", "lsoda": "LSODA",
}
_NUMERIC_OPTIONS = ("sbase", "vbase", "freq", "reltol", "abstol", "residualtol")
# LTspice cards CodeSim does not run: skipped with a warning.
_IGNORED_CARDS = {
    ".ac", ".dc", ".noise", ".tf", ".four", ".op", ".step", ".temp", ".wave", ".backanno",
    ".net", ".fra", ".ic", ".nodeset", ".savestate", ".loadstate", ".savebias", ".text",
}
_TRAN_MODIFIERS = {"uic", "steady", "nodiscard", "startup", "step", "convreport"}
_MEASURE_OPERATIONS = ("find", "deriv", "avg", "max", "min", "pp", "rms", "integ", "param", "trig", "when")


def _toPython(text: str) -> str:
    """Translate a lowercase SPICE expression to Python source.

    Precedence, lowest first: ``c ? a : b``; ``& |`` (``&& ||``); comparisons;
    ``+ -``; ``* / %``; unary ``- + ! ~``; ``**`` and ``^`` (power, right-associative).
    Numbers keep their SPICE suffixes; operators become the ``_if _and _or _not _mod``
    functions of ``_FUNCTIONS``. Raises ``SyntaxError``.
    """
    tokens, position = [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = _TOKEN.match(text, position)
        if match is None:
            raise SyntaxError(f"unexpected {text[position:position + 12]!r}")
        position = match.end()
        if match["number"] is not None:
            tail = match["tail"] or ""
            letters = tail.rstrip("0123456789")
            scale = (1e6 if letters.startswith("meg") else 25.4e-6 if letters.startswith("mil")
                     else _SUFFIXES.get(letters[:1], 1.0))
            number = match["number"]
            if tail[len(letters):] and scale != 1.0 and number.isdigit():
                number = f"{number}.{tail[len(letters):]}"  # 6k34 = 6.34k
            tokens.append(("value", repr(float(number) * scale)))
        elif match["name"] is not None:
            tokens.append(("name", match["name"]))
        else:
            tokens.append(("op", {"{": "(", "}": ")", "&&": "&", "||": "|"}.get(match["op"], match["op"])))
    index = 0

    def peek() -> str | None:
        return tokens[index][1] if index < len(tokens) and tokens[index][0] == "op" else None

    def expect(operator: str) -> None:
        nonlocal index
        if peek() != operator:
            where = f"before {tokens[index][1]!r}" if index < len(tokens) else "at the end"
            raise SyntaxError(f"expected {operator!r} {where}")
        index += 1

    def ternary() -> str:
        nonlocal index
        condition = logic()
        if peek() != "?":
            return condition
        index += 1
        whenTrue = ternary()
        expect(":")
        return f"_if({condition}, {whenTrue}, {ternary()})"

    def logic() -> str:
        nonlocal index
        left = compare()
        while peek() in ("&", "|"):
            function = "_and" if peek() == "&" else "_or"
            index += 1
            left = f"{function}({left}, {compare()})"
        return left

    def compare() -> str:
        nonlocal index
        left = additive()
        while peek() in ("==", "!=", "<", ">", "<=", ">="):
            operator = peek()
            index += 1
            left = f"(1.0*(({left}) {operator} ({additive()})))"
        return left

    def additive() -> str:
        nonlocal index
        left = term()
        while peek() in ("+", "-"):
            operator = peek()
            index += 1
            left = f"({left} {operator} {term()})"
        return left

    def term() -> str:
        nonlocal index
        left = unary()
        while peek() in ("*", "/", "%"):
            operator = peek()
            index += 1
            left = f"_mod({left}, {unary()})" if operator == "%" else f"({left} {operator} {unary()})"
        return left

    def unary() -> str:
        nonlocal index
        operator = peek()
        if operator in ("-", "+"):
            index += 1
            return f"({operator}{unary()})"
        if operator in ("!", "~"):
            index += 1
            return f"_not({unary()})"
        return power()

    def power() -> str:
        nonlocal index
        base = primary()
        if peek() in ("**", "^"):
            index += 1
            return f"({base} ** {unary()})"
        return base

    def primary() -> str:
        nonlocal index
        if index >= len(tokens):
            raise SyntaxError("unexpected end of expression")
        kind, value = tokens[index]
        index += 1
        if kind == "value":
            return value
        if kind == "name":
            if peek() != "(":
                return value
            index += 1
            arguments = []
            if peek() != ")":
                arguments.append(ternary())
                while peek() == ",":
                    index += 1
                    arguments.append(ternary())
            expect(")")
            return f"{'_if' if value == 'if' else value}({', '.join(arguments)})"
        if value == "(":
            inner = ternary()
            expect(")")
            return f"({inner})"
        raise SyntaxError(f"unexpected {value!r}")

    result = ternary()
    if index < len(tokens):
        raise SyntaxError(f"unexpected {tokens[index][1]!r}")
    return result


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
        notes: list[str] = []  # skipped cards and options, warned when the system is built

        # 1. Lines -> cards: comments, .include / .lib expansion, '+' continuation, .end.
        # Each line keeps (file, line number, text, chain of files including it, library only).
        pending = [(filename, number, line, (filename,), False)
                   for number, line in enumerate(lines[start + 1:], start + 2)]
        cards: list[list[Any]] = []
        position = 0
        while position < len(pending):
            file, number, line, chain, libOnly = pending[position]
            position += 1
            line = re.sub(r";.*$|\s\$\s.*$", "", line).strip()
            if not line or line.startswith("*"):
                continue
            if line.startswith("+"):
                if not cards:
                    raise SyntaxError("continuation line '+' without a card", (file, number, 0, None))
                cards[-1][2] += " " + line[1:]
                continue
            words = line.split()
            keyword = words[0].lower()
            if keyword == ".end":
                if file == filename:
                    break
                # .end of an included file ends that file only.
                while position < len(pending) and pending[position][3][:len(chain)] == chain:
                    position += 1
                continue
            if keyword == ".endl":
                continue
            if keyword == ".machine":
                notes.append(f"{Path(file).name}:{number}: .machine is not supported by CodeSim; block skipped")
                while position < len(pending) and pending[position][2].strip().lower()[:11] != ".endmachine":
                    position += 1
                position += 1
                continue
            if keyword in (".include", ".inc", ".lib"):
                arguments = [word.strip("'\"") for word in words[1:]]
                if keyword == ".lib" and len(arguments) == 1 and any(
                        entry[0] == file and entry[2].strip().lower().startswith(".endl")
                        for entry in pending[position:]):
                    # A '.lib section' ... '.endl' block, read only through '.lib file section'.
                    while position < len(pending):
                        position += 1
                        if pending[position - 1][0] == file and pending[position - 1][2].strip().lower().startswith(".endl"):
                            break
                    continue
                target = (arguments[0] if keyword == ".lib" and arguments
                          else line.split(None, 1)[1].strip().strip("'\"") if len(words) > 1 else "")
                if not target:
                    raise SyntaxError(f"{keyword} needs a file name", (file, number, 0, None))
                includePath = Path(target)
                base = Path(file).parent if Path(file).is_file() else directory
                includePath = (includePath if includePath.is_absolute() else base / includePath).resolve()
                if str(includePath) in chain:
                    raise SyntaxError(f"{keyword} cycle: {includePath}", (file, number, 0, None))
                try:
                    included = list(enumerate(includePath.read_text(encoding="utf-8").splitlines(), 1))
                except OSError as error:
                    raise SyntaxError(f"cannot read {keyword} file: {error}", (file, number, 0, None)) from None
                if keyword == ".lib" and len(arguments) > 1:
                    section = arguments[1].lower()
                    begin = next((k for k, (_, text) in enumerate(included)
                                  if [word.lower() for word in text.split()[:2]] == [".lib", section]), None)
                    if begin is None:
                        raise SyntaxError(f".lib section {arguments[1]} not found in {includePath}", (file, number, 0, None))
                    end = next((k for k in range(begin + 1, len(included))
                                if included[k][1].strip().lower().startswith(".endl")), len(included))
                    included = included[begin + 1:end]
                pending[position:position] = [
                    (str(includePath), includedNumber, includedLine, (*chain, str(includePath)),
                     libOnly or keyword == ".lib")
                    for includedNumber, includedLine in included
                ]
                continue
            cards.append([file, number, line, libOnly])

        # 2. Cards -> structure. Every value is a source string, compiled in step 3.
        # Cards are a worklist: X instances of a .subckt are replaced by the expanded
        # body cards, which are read like any other card.
        sources: list[tuple[str, str, int]] = []
        options: dict[str, Any] = {}
        params: list[tuple[str, int]] = []
        tran: list[int] = []
        hdl: list[Path] = []
        elements: list[dict[str, Any]] = []
        meas: dict[str, Any] | None = None
        measures: list[dict[str, Any]] = []
        probes: list[tuple[str, int]] = []
        funcs: list[tuple[str, list[str], int]] = []
        models: dict[str, dict[str, Any]] = {}
        subckts: dict[str, dict[str, Any]] = {}
        globalNodes: set[str] = set()
        current: dict[str, Any] | None = None  # .subckt being read
        position = 0
        while True:
            while position < len(cards):
                file, number, text, libOnly = cards[position]
                position += 1
                location = f"{Path(file).name}:{number}"
                tokens, word, depth, quoted = [], "", 0, False
                for char in text + " ":
                    if char == "'":
                        quoted = not quoted
                    elif not quoted and char in "({[":
                        depth += 1
                    elif not quoted and char in ")}]":
                        depth -= 1
                    if quoted or depth or char == "'" or char not in " \t,=":
                        word += char
                        continue
                    if word:
                        tokens.append(word)
                        word = ""
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
                letter = card.rsplit(":", 1)[-1][:1]  # element type of a (sub)circuit element name
                if current is not None:
                    if card == ".subckt":
                        raise SyntaxError("nested .subckt definitions are not supported", (file, number, 0, None))
                    if card == ".ends":
                        if positional[1:] and positional[1].lower() != current["name"]:
                            raise SyntaxError(f".ends {positional[1]} closes .subckt {current['name']}", (file, number, 0, None))
                        subckts[current["name"]] = current
                        current = None
                    else:
                        current["body"].append([file, number, text, libOnly])
                    continue
                if card == ".subckt":
                    if len(positional) < 2:
                        raise SyntaxError(".subckt expects: name pin ... [params: name=value ...]", (file, number, 0, None))
                    if positional[1].lower() in subckts:
                        raise SyntaxError(f"duplicate .subckt {positional[1]}", (file, number, 0, None))
                    current = {"name": positional[1].lower(), "file": file, "line": number, "body": [],
                               "pins": [pin.lower() for pin in positional[2:] if pin.lower() != "params:"],
                               "defaults": {key: value for key, (_, value) in pairs.items()}}
                    continue
                if card == ".ends":
                    raise SyntaxError(".ends without .subckt", (file, number, 0, None))
                if libOnly and card not in (".model", ".param", ".params", ".func", ".global"):
                    notes.append(f"{location}: {positional[0] if positional else text} is outside a .subckt in a "
                                 ".lib file; only .model, .subckt, .param and .func are read")
                    continue
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
                            notes.append(f"{location}: option {raw} is not used by CodeSim (known: "
                                         f"{', '.join((*_NUMERIC_OPTIONS, 'method', 'units'))}); ignored")
                    for flag in positional[1:]:
                        if flag.lower() != "reject_number_tails":
                            notes.append(f"{location}: option {flag} is not used by CodeSim; ignored")
                elif card in (".param", ".params"):
                    if len(positional) > 1:
                        raise SyntaxError(".param expects name=value pairs", (file, number, 0, None))
                    for key, (raw, value) in pairs.items():
                        sources.append((value, file, number))
                        params.append((raw, len(sources) - 1))
                elif card == ".func":
                    if len(positional) == 3 and not pairs:
                        head, body = positional[1], positional[2]
                    elif len(positional) == 1 and len(pairs) == 1:
                        head, body = next(iter(pairs.values()))
                    else:
                        head, body = "", ""
                    signature = re.fullmatch(r"(\w+)\s*\(([\w\s,]*)\)", head)
                    if not signature or not body:
                        raise SyntaxError(".func expects: name(arg, ...) {expression}", (file, number, 0, None))
                    sources.append((body, file, number))
                    funcs.append((signature[1].lower(),
                                  [argument.strip().lower() for argument in signature[2].split(",") if argument.strip()],
                                  len(sources) - 1))
                elif card == ".global":
                    globalNodes.update(node.lower() for node in positional[1:])
                elif card == ".title":
                    title = text.split(None, 1)[1].strip() if len(text.split()) > 1 else ""
                elif card == ".hdl":
                    if len(positional) != 2:
                        raise SyntaxError(".hdl needs one file name", (file, number, 0, None))
                    hdlPath = Path(positional[1].strip("'\""))
                    base = Path(file).parent if Path(file).is_file() else directory
                    hdl.append((hdlPath if hdlPath.is_absolute() else base / hdlPath).resolve())
                elif card == ".tran":
                    values = []
                    for token in positional[1:]:
                        if token.lower() not in _TRAN_MODIFIERS:
                            values.append(token)
                        elif token.lower() != "uic":
                            notes.append(f"{location}: .tran {token} is not supported by CodeSim; ignored")
                    if pairs:
                        notes.append(f"{location}: .tran {', '.join(raw for raw, _ in pairs.values())} "
                                     "not supported by CodeSim; ignored")
                    if not 2 <= len(values) <= 4:
                        raise SyntaxError(".tran expects: tstep tstop [tstart [tmax]]", (file, number, 0, None))
                    for value in values:
                        sources.append((value, file, number))
                        tran.append(len(sources) - 1)
                elif card in (".meas", ".measure"):
                    words = tokens[1:]
                    analysis = "tran"
                    if words and words[0].lower() in ("tran", "ac", "dc", "noise", "op"):
                        analysis, words = words[0].lower(), words[1:]
                    if analysis != "tran":
                        notes.append(f"{location}: .meas {analysis} is not supported by CodeSim; card skipped")
                        continue
                    usage = (".meas expects: [tran] name WHEN expr=value (event), or name FIND|DERIV expr "
                             "AT=t|WHEN expr=value, AVG|MAX|MIN|PP|RMS|INTEG expr [FROM=t] [TO=t], "
                             "TRIG expr=value ... TARG expr=value ..., PARAM expr")
                    if len(words) < 2 or words[1].lower() not in _MEASURE_OPERATIONS:
                        raise SyntaxError(usage, (file, number, 0, None))
                    name, operation = words[0], words[1].lower()
                    # Clauses: the leading expression, then AT / WHEN / TRIG / TARG, each with KEY=value pairs.
                    sections = [["", []]]
                    for token in words[1:] if operation in ("when", "trig") else words[2:]:
                        if token.lower() in ("at", "when", "trig", "targ"):
                            sections.append([token.lower(), []])
                        else:
                            sections[-1][1].append(token)
                    clauses = {}
                    for kind, items in sections:
                        if kind in clauses:
                            raise SyntaxError(f"{kind.upper()} given twice", (file, number, 0, None))
                        head, keys, index = [], {}, 0
                        while index < len(items):
                            key = items[index].lower()
                            if key in ("val", "td", "from", "to", "rise", "fall", "cross") and index + 2 < len(items) \
                                    and items[index + 1] == "=":
                                keys[key] = items[index + 2]
                                index += 3
                            elif key == "last":
                                keys["last"] = "last"
                                index += 1
                            else:
                                head.append(items[index])
                                index += 1
                        clauses[kind] = (head, keys)
                    spec: dict[str, Any] = {"name": name, "op": operation, "file": file, "line": number}
                    # Crossings: expr=value (or expr VAL=value), [TD=t | FROM=t], RISE|FALL|CROSS=n|LAST.
                    for kind in ("when", "trig", "targ"):
                        if kind not in clauses:
                            continue
                        head, keys = clauses[kind]
                        if "=" in head:
                            split = head.index("=")
                            lhs, rhs = " ".join(head[:split]), " ".join(head[split + 1:])
                        else:
                            lhs, rhs = " ".join(head), keys.get("val", "")
                        if not lhs or not rhs:
                            raise SyntaxError(f"{kind.upper()} expects expr=value or expr VAL=value", (file, number, 0, None))
                        sources.append((f"({lhs}) - ({rhs})", file, number))
                        crossing = {"code": len(sources) - 1, "edge": "cross", "count": "1", "td": None}
                        for key in ("rise", "fall", "cross"):
                            if key in keys:
                                crossing["edge"], crossing["count"] = key, keys[key].lower()
                        if "last" in keys:
                            crossing["count"] = "last"
                        if crossing["count"] != "last" and (not crossing["count"].isdigit() or crossing["count"] == "0"):
                            raise SyntaxError(f"{crossing['edge'].upper()} must be a positive count or LAST", (file, number, 0, None))
                        if keys.get("td", keys.get("from")) is not None:
                            sources.append((keys.get("td", keys.get("from")), file, number))
                            crossing["td"] = len(sources) - 1
                        spec[kind] = crossing
                    if operation == "when":
                        # The state-triggered event: the first crossing after FROM (its time is also reported).
                        if meas is not None:
                            raise SyntaxError("only one .meas ... WHEN event is supported", (file, number, 0, None))
                        crossing = spec["when"]
                        if crossing["count"] != "1":
                            raise SyntaxError(f"only {crossing['edge'].upper()}=1 (the first crossing) is supported "
                                              "for the WHEN event; use FIND time WHEN ... for later crossings",
                                              (file, number, 0, None))
                        meas = {"name": name, "code": crossing["code"], "from": crossing["td"], "file": file,
                                "line": number, "direction": {"rise": 1.0, "fall": -1.0, "cross": 0.0}[crossing["edge"]]}
                        continue
                    head, keys = clauses[""]
                    if operation == "trig":
                        if "targ" not in spec:
                            raise SyntaxError("TRIG needs a TARG clause", (file, number, 0, None))
                    else:
                        if not head:
                            raise SyntaxError(f"{operation.upper()} needs an expression", (file, number, 0, None))
                        sources.append((" ".join(head), file, number))
                        spec["expr"] = len(sources) - 1
                    if operation in ("find", "deriv"):
                        if "at" in clauses:
                            value = [item for item in clauses["at"][0] if item != "="]
                            if len(value) != 1:
                                raise SyntaxError("AT expects AT=time", (file, number, 0, None))
                            sources.append((value[0], file, number))
                            spec["at"] = len(sources) - 1
                        elif "when" not in spec:
                            raise SyntaxError(f"{operation.upper()} needs AT=time or WHEN expr=value", (file, number, 0, None))
                    for key in ("from", "to"):
                        if key in keys:
                            sources.append((keys[key], file, number))
                            spec[key] = len(sources) - 1
                    measures.append(spec)
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
                elif card in _IGNORED_CARDS:
                    notes.append(f"{location}: {positional[0]} is not supported by CodeSim; card skipped")
                elif card.startswith(".") and card != ".model":
                    raise SyntaxError(f"unsupported card {positional[0]}", (file, number, 0, None))
                elif letter in ("r", "l", "c"):
                    # HSPICE-style combined branch: L... R=r, or R... L=l.
                    series = {"l": "r", "r": "l"}.get(letter)
                    if len(positional) != 4 or set(pairs) - {series}:
                        usage = f" [{series.upper()}=value]" if series else ""
                        raise SyntaxError(f"{positional[0]} expects: name node1 node2 value{usage}", (file, number, 0, None))
                    sources.append((positional[3], file, number))
                    element = {"kind": letter, "name": positional[0], "nodes": positional[1:3],
                               "value": len(sources) - 1, "file": file, "line": number}
                    if series in pairs:
                        sources.append((pairs[series][1], file, number))
                        element["series"] = len(sources) - 1
                    elements.append(element)
                elif card == ".model" or letter == "x":
                    positional = [token for token in positional if token.lower() != "params:"]
                    if card == ".model":
                        # A parenthesised parameter list is read again as plain name=value pairs.
                        grouped = next((token for token in positional[2:] if "(" in token), None)
                        if grouped is not None:
                            head, _, inner = grouped.partition("(")
                            cards.insert(position, [file, number, text.replace(grouped, f"{head} {inner[:-1]}", 1), libOnly])
                            continue
                        ako = len(positional) > 2 and positional[2].lower().startswith("ako:")
                        if len(positional) not in ((3, 4) if ako else (3,)):
                            raise SyntaxError(".model expects: name Model(par=value ...) or "
                                              "name ako:other [Model](par=value ...)", (file, number, 0, None))
                    elif len(positional) < 3:
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
                    if card == ".model":
                        if positional[1].lower() in models:
                            raise SyntaxError(f"duplicate .model {positional[1]}", (file, number, 0, None))
                        models[positional[1].lower()] = {
                            "base": positional[2][4:] if ako else positional[2], "params": instance,
                            "file": file, "line": number}
                    else:
                        elements.append({"kind": "x", "name": positional[0], "nodes": positional[1:-1],
                                         "model": positional[-1], "params": instance, "file": file, "line": number,
                                         "texts": {key: value for key, (_, value) in pairs.items()}})
                else:
                    raise SyntaxError(f"unsupported element {positional[0]} (use R, L, C or X; CodeSim has no "
                                      f"{letter.upper()} elements)", (file, number, 0, None))
            if current is not None:
                raise SyntaxError(f".subckt {current['name']} has no .ends", (current["file"], current["line"], 0, None))

            # Expand the X instances of a .subckt: element names and local nodes get the instance
            # prefix, pins become the instance nodes, local parameters are substituted.
            instances = [element for element in elements
                         if element["kind"] == "x" and element["model"].lower() in subckts]
            if not instances:
                break
            expanded = {id(element) for element in instances}
            elements = [element for element in elements if id(element) not in expanded]
            for element in instances:
                definition = subckts[element["model"].lower()]
                where = (element["file"], element["line"], 0, None)
                prefix = element["name"].lower() + ":"
                if prefix.count(":") > 20:
                    raise SyntaxError(f"{element['name']}: .subckt recursion deeper than 20 levels", where)
                if len(element["nodes"]) != len(definition["pins"]):
                    raise SyntaxError(
                        f"{element['name']}: .subckt {definition['name']} has {len(definition['pins'])} pin(s) "
                        f"({' '.join(definition['pins'])}), got {len(element['nodes'])} node(s)", where)
                pins = dict(zip(definition["pins"], (node.lower() for node in element["nodes"])))
                # Local parameters: params: defaults, the body's .param values, the instance's (parent scope).
                values = dict(definition["defaults"])
                for _, _, bodyText, _ in definition["body"]:
                    if bodyText.split()[0].lower() in (".param", ".params"):
                        values.update((key.lower(), value) for key, value in re.findall(
                            r"(\w+)\s*=\s*(\{[^{}]*\}|'[^']*'|[^\s=]+)", bodyText.split(None, 1)[1] if len(bodyText.split()) > 1 else ""))
                fixed = element["texts"]
                unknown = set(fixed) - set(values)
                if unknown:
                    raise SyntaxError(f"{element['name']}: .subckt {definition['name']} has no parameter(s) "
                                      f"{', '.join(sorted(unknown))}; parameters: {', '.join(values) or 'none'}", where)
                names = "|".join(re.escape(name) for name in sorted(values, key=len, reverse=True))
                pattern = re.compile(rf"\{{\s*({names})\s*\}}|(?<![\w@:.])({names})(?![\w:(])(?!\s*=(?!=))",
                                     re.IGNORECASE) if names else None
                bare: dict[str, str] = {}
                # A PWL value replaces the name as it is; any other value is parenthesised.
                replace = lambda match: (
                    bare[key] if bare[(key := (match[1] or match[2]).lower())].lower().startswith("pwl(")
                    else f"{{({bare[key]})}}" if match[1] else f"({bare[key]})")
                for _ in range(21):
                    resolved = {**values, **fixed}
                    bare = {key: value[1:-1] if (value[:1], value[-1:]) in (("{", "}"), ("'", "'")) else value
                            for key, value in resolved.items()}
                    open_ = [key for key in values if key not in fixed and pattern.search(values[key])] if pattern else []
                    if not open_:
                        break
                    for key in open_:
                        values[key] = pattern.sub(replace, values[key])
                else:
                    raise SyntaxError(f"{element['name']}: circular .subckt parameters: {', '.join(open_)}", where)
                for bodyFile, bodyNumber, bodyText, _ in definition["body"]:
                    words = bodyText.split()
                    first = words[0].lower()
                    if first in (".param", ".params"):
                        continue
                    if first == ".model":
                        cards.append([bodyFile, bodyNumber, pattern.sub(replace, bodyText) if pattern else bodyText, False])
                        continue
                    if first[:1] in ("r", "l", "c"):
                        count = 3
                    elif first[:1] == "x":
                        count = 1
                        while (count < len(words) and "=" not in words[count] and words[count].lower() != "params:"
                               and not (count + 1 < len(words) and words[count + 1].startswith("="))):
                            count += 1
                    else:
                        raise SyntaxError(f"{words[0]}: a .subckt may hold R, L, C, X, .param and .model cards",
                                          (bodyFile, bodyNumber, 0, None))
                    if len(words) < count or (first[:1] == "x" and count < 3):
                        raise SyntaxError(f"{words[0]}: missing node(s)", (bodyFile, bodyNumber, 0, None))
                    parts = bodyText.split(None, count)
                    rest = parts[count] if len(parts) > count else ""
                    mapped = [node if node in ("0", "gnd") or node in globalNodes or node.startswith("$g_")
                              else pins.get(node, prefix + node)
                              for node in (word.lower() for word in (words[1:count - 1] if first[:1] == "x" else words[1:3]))]
                    head = [prefix + first, *mapped, *([words[count - 1]] if first[:1] == "x" else [])]
                    cards.append([bodyFile, bodyNumber,
                                  " ".join([*head, pattern.sub(replace, rest) if pattern else rest]), False])

        # Device models: follow .model names (and ako: chains) to the device class name;
        # the model's parameters are defaults of the instance.
        for element in elements:
            if element["kind"] != "x":
                continue
            if len(element["nodes"]) != 2:
                raise SyntaxError(f"{element['name']} expects: name node+ node- Model par=value ... "
                                  f"(or nodes of a .subckt; no .subckt {element['model']})",
                                  (element["file"], element["line"], 0, None))
            merged, model, chain = {}, element["model"], []
            while model.lower() in models:
                if model.lower() in chain:
                    raise SyntaxError(f"{element['name']}: .model cycle {' -> '.join(chain)}",
                                      (element["file"], element["line"], 0, None))
                chain.append(model.lower())
                merged = {**models[model.lower()]["params"], **merged}
                model = models[model.lower()]["base"]
            element["params"] = {**merged, **element["params"]}
            element["model"] = model

        names = [element["name"].lower() for element in elements]
        for element in elements:
            if names.count(element["name"].lower()) > 1:
                raise SyntaxError(f"duplicate element name {element['name']}", (element["file"], element["line"], 0, None))
        measureNames = [spec["name"].lower() for spec in measures] + ([meas["name"].lower()] if meas else [])
        for spec in measures + ([meas] if meas else []):
            if measureNames.count(spec["name"].lower()) > 1:
                raise SyntaxError(f"duplicate .meas name {spec['name']}", (spec["file"], spec["line"], 0, None))
            if spec["name"].lower() in {name.lower() for name, _ in params}:
                raise SyntaxError(f".meas name {spec['name']} is also a .param", (spec["file"], spec["line"], 0, None))

        # 3. Compile every source: strip {...} / '...', replace the output references by
        # placeholders, translate the SPICE expression, then put the array lookups in.
        lambdaArgs = {index: args for _, args, index in funcs}
        codes = []
        for sourceIndex, (text, file, number) in enumerate(sources):
            text = text.strip()
            if (text[:1], text[-1:]) in (("{", "}"), ("'", "'")):
                text = text[1:-1]
            text = text.lower()
            pieces, references, lookups, position = [], [], [], 0
            for match in _REFERENCE.finditer(text):
                pieces += [text[position:match.start()], f"_r{len(lookups)}_"]
                position = match.end()
                if match[1]:
                    key = f"{match[1]}.{match[2]}"
                    references.append(("state", key))
                    lookups.append(f"S[..., IDX[{key!r}]]")
                    continue
                function, argument, second = match[3], match[4], match[5]
                if function in ("i", "im"):
                    if second:
                        raise SyntaxError(f"{function.upper()}() takes one R/L name", (file, number, 0, None))
                    references.append(("branch", argument))
                    lookup = f"IB[..., BR[{argument!r}]]"
                    lookups.append(f"real({lookup})" if function == "i" else f"abs({lookup})")
                    continue
                if second and function not in ("v", "vm"):
                    raise SyntaxError(f"{function.upper()}() takes one node", (file, number, 0, None))
                argument = "0" if argument == "gnd" else argument
                references.append(("node", argument))
                lookup = f"V[..., NODE[{argument!r}]]"
                if second:
                    second = "0" if second == "gnd" else second
                    references.append(("node", second))
                    lookup = f"({lookup} - V[..., NODE[{second!r}]])"
                lookups.append({
                    "v": f"real({lookup})", "vm": f"abs({lookup})",
                    "vp": f"angle({lookup} / V[..., REF])",
                    "p": f"real(SB[..., NODE[{argument!r}]])", "q": f"imag(SB[..., NODE[{argument!r}]])",
                }[function])
            pieces.append(text[position:])
            if sourceIndex in lambdaArgs and references:
                raise SyntaxError(".func bodies cannot read signals; pass them as arguments", (file, number, 0, None))
            try:
                python = re.sub(r"\b_r(\d+)_\b", lambda found: lookups[int(found[1])], _toPython("".join(pieces)))
                if sourceIndex in lambdaArgs:
                    python = f"lambda {', '.join(lambdaArgs[sourceIndex])}: {python}"
                code = compile(python, file, "eval")
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
                registry.Workspace(path.parent).importFile(path)

        cls._filename, cls._title, cls._options, cls._params = filename, title, options, params
        cls._tran, cls._elements, cls._meas, cls._probes, cls._codes = tran, elements, meas, probes, codes
        cls._funcs, cls._measures, cls._warnings = funcs, measures, notes

    @classmethod
    def defaultParameters(cls, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """``.param`` values (``overrides`` replace them before dependent values are
        evaluated), then the derived run settings the frontend needs."""
        if not hasattr(cls, "_codes"):
            raise TypeError(f"{cls.__name__} has no netlist: class X(NetlistSystem, netlistFile=...)")
        overrides = dict(overrides or {})
        env, p = dict(_FUNCTIONS), {}
        for name, _, index in cls._funcs:
            env[name] = eval(cls._codes[index][0], env)
        eventName = cls._meas["name"].lower() if cls._meas else None
        code, file, number = None, cls._filename, 0
        try:
            # .param values in any order: a value that uses one not known yet waits for it.
            waiting, failure = list(cls._params), None
            while waiting:
                deferred = []
                for name, index in waiting:
                    code, _, file, number = cls._codes[index]
                    try:
                        p[name] = overrides.pop(name) if name in overrides else float(np.real(eval(code, env)))
                    except NameError as error:
                        deferred.append((name, index))
                        failure = (error, file, number)
                        continue
                    env[name.lower()] = p[name]
                if len(deferred) == len(waiting):
                    raise SyntaxError(f"{failure[0]}", (failure[1], failure[2], 0, None))
                waiting = deferred
            p = {name: p[name] for name, _ in cls._params}
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
        # Skipped cards and options: warned here, so they reach whoever builds the system.
        for message in cls._warnings:
            warnings.warn(f"{cls.__name__}: {message}", stacklevel=2)
        p = cls.defaultParameters() if p is None else p
        env = {**_FUNCTIONS, **{str(name).lower(): value for name, value in p.items()}}
        for name, _, index in cls._funcs:
            env[name] = eval(cls._codes[index][0], env)
        eventName = cls._meas["name"].lower() if cls._meas else None
        measureNames = {spec["name"].lower() for spec in cls._measures}
        for code, _, file, number in cls._codes:
            unknown = set(code.co_names) - env.keys() - _RUNTIME_NAMES - {eventName} - measureNames
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

        # Branches: plain R and L cards on one node pair merge into one series RL branch;
        # a combined card (L ... R=, R ... L=) is a branch of its own.
        branches: dict[tuple[str, ...], dict[str, Any]] = {}
        capacitance, capacitorLine = np.zeros(count), [None] * count
        voltageDevices, currentDevices = [None] * count, [[] for _ in range(count)]
        references: dict[str, Any] = {}
        devices: dict[str, Any] = {}
        for element in cls._elements:
            kind, name, file, number = element["kind"], element["name"], element["file"], element["line"]
            if kind in ("r", "l", "c"):
                value = float(eval(cls._codes[element["value"]][0], env))
                if value < 0 or (kind == "c" and value == 0):
                    requirement = "positive" if kind == "c" else "zero or positive"
                    raise SyntaxError(f"{name}: value must be {requirement}", (file, number, 0, None))
            if kind in ("r", "l"):
                values = {kind.upper(): value}
                if "series" in element:
                    other = "R" if kind == "l" else "L"
                    values[other] = float(eval(cls._codes[element["series"]][0], env))
                    if values[other] < 0:
                        raise SyntaxError(f"{name}: {other} must be zero or positive", (file, number, 0, None))
                key = ("card", name.lower()) if "series" in element else tuple(sorted(element["keys"]))
                branch = branches.setdefault(key, {"nodes": element["keys"], "R": None, "L": None,
                                                   "names": [], "file": file, "line": number})
                if branch[kind.upper()] is not None:
                    raise SyntaxError(f"{name}: a second {kind.upper()} between the same nodes; "
                                       "only one R and one L per node pair (use L ... R=... for a "
                                       "parallel branch)", (file, number, 0, None))
                for part, partValue in values.items():
                    if not perUnit:
                        partValue = partValue / impedanceBase if part == "R" else wBase * partValue / impedanceBase
                    branch[part] = partValue
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
                if device.type == "Voltage":
                    if voltageDevices[node] is not None:
                        raise SyntaxError(f"{name}: bus {busNames[node]} already has a voltage-port device; "
                                           "a bus takes one voltage-port device", (file, number, 0, None))
                    voltageDevices[node] = device
                else:
                    currentDevices[node].append(device)
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
            # A bus without a voltage port is reduced by RLGrid.localDAEreduction.
            if voltageDevices[node] is not None and not currentDevices[node]:
                currentDevices[node] = [components.ZeroCurrent(vbase, sbase, f"{safeName}_zero")]

        branchList = list(branches.values())
        incidence = np.zeros((count, len(branchList)))
        for column, branch in enumerate(branchList):
            incidence[busIndex[branch["nodes"][0]], column] = 1
            incidence[busIndex[branch["nodes"][1]], column] = -1
        branchR = np.asarray([branch["R"] or 0.0 for branch in branchList])
        branchL = np.asarray([branch["L"] or 0.0 for branch in branchList])
        for branch in branchList:
            if not branch["R"] and not branch["L"]:
                raise SyntaxError(f"{branch['names'][0]}: the branch has zero impedance (R = L = 0)",
                                  (branch["file"], branch["line"], 0, None))
            if not branch["L"] and all(voltageDevices[busIndex[node]] is not None for node in branch["nodes"]):
                raise SyntaxError(
                    f"{branch['names'][0]}: a branch without inductance between two buses with voltage "
                    "devices; add an L in series", (branch["file"], branch["line"], 0, None))

        # Source locations for the frontend ("Open system code", "Open instance"): the
        # netlist file, the X card of each device and the first card of each branch.
        # Netlist text (netlist="...") has no file to open.
        self.sourceReference = {"file": cls._filename, "line": 1, "column": 1} if Path(cls._filename).is_file() else None
        self.deviceSources = {
            id(devices[element["name"].lower()]): {"file": element["file"], "line": element["line"], "column": 1}
            for element in cls._elements if element["kind"] == "x" and Path(element["file"]).is_file()
        }
        self.branchSources = [{"file": branch["file"], "line": branch["line"], "column": 1}
                              if Path(branch["file"]).is_file() else self.sourceReference
                              for branch in branches.values()] if self.sourceReference else None
        currentDevices = [None if not items else items[0] if len(items) == 1 else items for items in currentDevices]
        devices.update({"Vdev": voltageDevices, "Idev": currentDevices})
        network = {"A": incidence, "R": branchR, "L": branchL, "busNames": busNames}
        try:
            super().__init__(cls.__name__, network, devices,
                             {"Vbase": vbase, "Sbase": sbase, "frequency_Hz": p["frequency_Hz"]}, references)
        except ValueError as error:
            # RLGrid cannot reduce a bus ("bus a, b: ..."): report it at a card on that bus.
            buses = str(error).split(":")[0].removeprefix("bus ").split(", ")
            card = next((element for element in cls._elements
                         if element["kind"] != "c" and element["nodes"][0] in buses), None)
            location = (card["file"], card["line"]) if card else (cls._filename, 1)
            raise SyntaxError(str(error), (*location, 0, None)) from None
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
        # Columns of V/IB/SB: retained buses, then buses reduced by RLGrid.localDAEreduction;
        # network branches, then the branches of the reduced buses.
        grid = self.rlGrid
        nodeIndex = {node: retained[index] for node, index in busIndex.items() if index in retained}
        nodeIndex.update({node: grid.n + grid.floatingNodes.index(index)
                          for node, index in busIndex.items() if index in grid.floatingNodes})
        survivors = [branch for branch in branchList
                     if busIndex[branch["nodes"][0]] in retained and busIndex[branch["nodes"][1]] in retained]
        branchIndex = {name: rank for rank, branch in enumerate(survivors) for name in branch["names"]}
        branchIndex.update({name: grid.b + grid.floatingColumns.index(column)
                            for column, branch in enumerate(branchList) if column in grid.floatingColumns
                            for name in branch["names"]})
        maps = {"state": stateIndex, "node": nodeIndex, "branch": branchIndex}
        for _, used, file, number in cls._codes:
            for kind, key in used:
                if key not in maps[kind]:
                    reason = {
                        "state": "no such device state (use @xname[state])",
                        "node": "no such bus, or the bus has no path to a source",
                        "branch": "no such R/L, or it has no path to a source",
                    }[kind]
                    raise SyntaxError(f"{key}: {reason}", (file, number, 0, None))
        reference = next((position for position, original in enumerate(self.retainedNodeIndices)
                          if isinstance(voltageDevices[original], components.Infbus)
                          and original != busIndex.get("0")), 0)
        self.netlistEnvironment = {**env, "IDX": stateIndex, "NODE": nodeIndex, "BR": branchIndex,
                                   "REF": reference}
        self._floatingIncidence = incidence[np.ix_(grid.floatingNodes, grid.floatingColumns)]
        self.referenceBusIndex = reference

        # .probe outputs, evaluated by System.calculateSignals.
        def evaluateProbe(code, time_s, state, networkSignals):
            floatingVoltage = networkSignals["floatingVoltage"]
            floatingPower = floatingVoltage * np.conj(networkSignals["floatingBranchCurrent"] @ self._floatingIncidence.T)
            local = {"S": np.asarray(state), "time": np.asarray(time_s),
                     "V": np.hstack((networkSignals["busVoltage"], floatingVoltage)),
                     "IB": np.hstack((networkSignals["branchCurrent"], networkSignals["floatingBranchCurrent"])),
                     "SB": np.hstack((networkSignals["busPower"], floatingPower))}
            return np.real(eval(code, self.netlistEnvironment, local)) * np.ones(len(state))

        self.probes = {name: (lambda time_s, state, net, code=cls._codes[index][0]: evaluateProbe(code, time_s, state, net))
                       for name, index in cls._probes}
        # Post-run .meas: the series they need (calculateSignals) and their times (simulate).
        self._evaluateSeries = evaluateProbe
        self._measureCodes = {
            f"{position}.{role}": cls._codes[spec[role] if role == "expr" else spec[role]["code"]][0]
            for position, spec in enumerate(cls._measures) for role in ("expr", "when", "trig", "targ")
            if role in spec and not (role == "expr" and spec["op"] == "param")
        }
        self.measureSettings = []
        for spec in cls._measures:
            numbers = {key: float(eval(cls._codes[spec[key]][0], env)) for key in ("at", "from", "to") if key in spec}
            numbers.update({f"{kind}.td": float(eval(cls._codes[spec[kind]["td"]][0], env))
                            for kind in ("when", "trig", "targ") if kind in spec and spec[kind]["td"] is not None})
            self.measureSettings.append(numbers)
        if cls._meas:
            self.eventName = cls._meas["name"]
            self.eventDirection = cls._meas["direction"]
            self.eventStartTime_s = (0.0 if cls._meas["from"] is None
                                     else float(eval(cls._codes[cls._meas["from"]][0], env)))

    def calculateSignals(self, time_s: Any, state: Any, networkSignals: dict[str, Any]):
        """``System.calculateSignals`` plus ``measureSeries``: the series of the post-run ``.meas``
        (read by :meth:`simulate`; the frontend does not show them)."""
        signals = super().calculateSignals(time_s, state, networkSignals)
        if getattr(self, "_measureCodes", None):
            signals["measureSeries"] = {key: self._evaluateSeries(code, time_s, state, networkSignals)
                                        for key, code in self._measureCodes.items()}
        return signals

    def simulate(self, time_s: Any, *, saveSnapshot: bool = True, **settings: Any) -> dict[str, Any]:
        """``System.simulate``, then the ``.meas`` results in ``result["measurements"]``
        (``{name: value}``; ``nan`` and a warning when a measurement fails)."""
        cls = type(self)
        result = super().simulate(time_s, saveSnapshot=False, **settings)
        series = result["signals"].pop("measureSeries", {})
        if cls._measures or cls._meas:
            t = np.asarray(result["time_s"], dtype=float)
            values: dict[str, float] = {}
            if cls._meas:
                values[cls._meas["name"]] = float(self.eventTime_s) if np.isfinite(self.eventTime_s) else np.nan
            for position, (spec, numbers) in enumerate(zip(cls._measures, self.measureSettings)):
                operation, failure, value, times = spec["op"], None, np.nan, {}
                # Crossing times of the WHEN / TRIG / TARG conditions, linear between samples.
                for kind in ("when", "trig", "targ"):
                    if kind not in spec:
                        continue
                    y = np.asarray(series[f"{position}.{kind}"], dtype=float)
                    before, after = y[:-1], y[1:]
                    rising, falling = (before < 0) & (after >= 0), (before > 0) & (after <= 0)
                    edge, count = spec[kind]["edge"], spec[kind]["count"]
                    mask = rising if edge == "rise" else falling if edge == "fall" else rising | falling
                    with np.errstate(divide="ignore", invalid="ignore"):
                        crossings = t[:-1] - before * (t[1:] - t[:-1]) / (after - before)
                    crossings = crossings[mask]
                    crossings = crossings[crossings >= numbers.get(f"{kind}.td", -np.inf)]
                    if not crossings.size or (count != "last" and int(count) > crossings.size):
                        failure = f"{kind.upper()} condition not met ({crossings.size} {edge} crossing(s))"
                        break
                    times[kind] = float(crossings[-1] if count == "last" else crossings[int(count) - 1])
                try:
                    if failure is not None:
                        pass
                    elif operation == "param":
                        scope = {**self.netlistEnvironment, **{name.lower(): item for name, item in values.items()}}
                        value = float(np.real(eval(cls._codes[spec["expr"]][0], scope)))
                    elif operation == "trig":
                        value = times["targ"] - times["trig"]
                    else:
                        y = np.asarray(series[f"{position}.expr"], dtype=float)
                        if operation in ("find", "deriv"):
                            at = times["when"] if "when" in spec else numbers["at"]
                            if not t[0] <= at <= t[-1]:
                                failure = f"time {at:g} s is outside the run"
                            else:
                                value = float(np.interp(at, t, np.gradient(y, t) if operation == "deriv" else y))
                        else:
                            start, stop = max(numbers.get("from", t[0]), t[0]), min(numbers.get("to", t[-1]), t[-1])
                            if not start < stop:
                                failure = f"empty window FROM={start:g} TO={stop:g}"
                            else:
                                window = np.r_[start, t[(t > start) & (t < stop)], stop]
                                y = np.interp(window, t, y)
                                value = {
                                    "avg": lambda: np.trapezoid(y, window) / (stop - start),
                                    "rms": lambda: np.sqrt(np.trapezoid(y**2, window) / (stop - start)),
                                    "integ": lambda: np.trapezoid(y, window),
                                    "max": y.max, "min": y.min, "pp": lambda: y.max() - y.min(),
                                }[operation]()
                                value = float(value)
                except Exception as error:  # e.g. PARAM of a name that is not a measurement
                    failure = f"{type(error).__name__}: {error}"
                if failure is not None:
                    warnings.warn(f"{self.name}: .meas {spec['name']} failed: {failure}", stacklevel=2)
                    value = np.nan
                values[spec["name"]] = value
            result["measurements"] = values
        if saveSnapshot:
            _, result["frontendDescriptor"] = self.outputMatFactory.saveSnapshot(result)
        return result

    def eventHandler(self, time_s: float, state: np.ndarray) -> float:
        """The ``.meas ... WHEN`` expression minus its value; the event fires at its zero crossing."""
        if not type(self)._meas:
            return super().eventHandler(time_s, state)
        code = type(self)._codes[type(self)._meas["code"]][0]
        local = {"S": np.asarray(state), "time": time_s}
        if {"V", "IB", "SB"} & set(code.co_names):
            grid = self.rlGrid
            # g = [ib, vn, deviceCurrent, v_floating, i_floating], as complex pairs.
            value = grid.g_core(state, self.getReference(time_s)).reshape(-1, 2) @ np.array([1, 1j])
            branch, voltage, _, floatingVoltage, floatingBranch = np.split(
                value, np.cumsum([grid.b, grid.n, grid.n, len(grid.floatingNodes)]))
            local["V"] = np.r_[voltage, floatingVoltage]
            local["IB"] = np.r_[branch, floatingBranch]
            local["SB"] = local["V"] * np.conj(np.r_[branch @ grid.A.T, floatingBranch @ self._floatingIncidence.T])
        return float(np.real(eval(code, self.netlistEnvironment, local)))
