"""Registry of runnable systems: the built-ins plus ``@system`` classes in a user workspace.

Users mark a system class with :func:`system`::

    from codesimpy import system
    from codesimpy.systems import SMIB__GFL_C_F_L

    @system(id="MySMIB", label="SMIB · my variant")
    class MySMIB(SMIB__GFL_C_F_L):
        ...

:meth:`Registry.discover` (alias :func:`discover`) imports the decorated files of a
:class:`Workspace`, loads its ``.cir`` netlists as :class:`~codesimpy.core.NetlistSystem`
classes and returns a :class:`Registry`. Components a Python system creates itself need
no registration; user components used by a netlist ``X`` line are marked with
:func:`device` (built-in components are found by their class name).
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import traceback
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, TypeVar, overload

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None  # type: ignore[assignment]

T = TypeVar("T", bound=type)

#: ``@device`` classes by lower-case name: the models a netlist ``X`` line can use.
DEVICES: dict[str, type] = {}


@dataclass(frozen=True)
class Meta:
    """What ``@system`` records on a class."""
    id: str
    label: str
    category: str = ""
    description: str = ""
    file: str = ""
    line: int = 0

    ATTRIBUTE = "__codesim__"

    @classmethod
    def of(cls, target: type) -> Meta | None:
        """The class's own ``@system`` metadata; subclasses do not inherit registration."""
        return target.__dict__.get(cls.ATTRIBUTE)


@overload
def system(cls: T, /) -> T: ...
@overload
def system(
    *, id: str | None = None, label: str | None = None,
    category: str = "", description: str | None = None,
) -> Callable[[T], T]: ...
def system(cls=None, /, *, id=None, label=None, category="", description=None):
    """Mark a :class:`~codesimpy.core.System` subclass as runnable from the frontend.

    Use it bare (``@system``) or with metadata (``@system(id=..., label=...)``).
    ``id`` defaults to the class name and must be unique among all systems.
    """
    def apply(target):
        from .core import System

        if not (isinstance(target, type) and issubclass(target, System)):
            raise TypeError(f"@system must decorate a System subclass, got {target!r}")
        if inspect.isabstract(target):
            missing = ", ".join(sorted(target.__abstractmethods__))
            raise TypeError(f"@system {target.__name__} is abstract; implement: {missing}")
        if not callable(getattr(target, "defaultParameters", None)):
            raise TypeError(f"@system {target.__name__} must define defaultParameters(overrides)")
        try:
            file, line = inspect.getsourcefile(target) or "", inspect.getsourcelines(target)[1]
        except (OSError, TypeError):  # no source available (e.g. compiled)
            file, line = "", 0
        systemId = id or target.__name__
        firstDocLine = (inspect.getdoc(target) or "").split("\n")[0]
        setattr(target, Meta.ATTRIBUTE, Meta(
            id=systemId, label=label or systemId, category=category,
            description=description if description is not None else firstDocLine,
            file=file, line=line,
        ))
        return target

    return apply(cls) if cls is not None else apply


def device(cls=None, /, *, name=None):
    """Mark a :class:`~codesimpy.core.Device` subclass as a netlist model.

    Use it bare (``@device``) or with a name (``@device(name="MyGFM")``); the
    name defaults to the class name and is matched case-insensitively by the
    model field of a netlist ``X`` line.
    """
    def apply(target):
        from .core import Device

        if not (isinstance(target, type) and issubclass(target, Device)):
            raise TypeError(f"@device must decorate a Device subclass, got {target!r}")
        if inspect.isabstract(target):
            missing = ", ".join(sorted(target.__abstractmethods__))
            raise TypeError(f"@device {target.__name__} is abstract; implement: {missing}")
        deviceName = name or target.__name__
        existing = DEVICES.get(deviceName.lower())
        if existing is not None and (existing.__module__, existing.__qualname__) != (
            target.__module__, target.__qualname__,
        ):
            raise ValueError(
                f"device name {deviceName!r} is already used by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        setattr(target, "__codesim_device__", deviceName)
        DEVICES[deviceName.lower()] = target
        return target

    return apply(cls) if cls is not None else apply


class Workspace:
    """A user folder: its ``[tool.codesim]`` configuration, the files to load and their import."""

    MODULE_PREFIX = "codesimpy_user"
    SKIPPED_DIRECTORIES = {
        ".git", ".hg", ".svn", ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache",
        "node_modules", "__pycache__", "build", "dist", "site-packages",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @cached_property
    def config(self) -> dict[str, Any]:
        """``[tool.codesim]`` of the workspace ``pyproject.toml`` (empty without one)."""
        pyproject = self.root / "pyproject.toml"
        if tomllib is None or not pyproject.is_file():
            return {}
        with pyproject.open("rb") as stream:
            return tomllib.load(stream).get("tool", {}).get("codesim", {})

    def files(self) -> list[Path]:
        """Files to load: ``[tool.codesim] systems`` globs, else every ``.cir`` netlist and
        every ``.py`` file that defines a class decorated with ``system`` or ``device``."""
        patterns = self.config.get("systems")
        if patterns:
            files = {path.resolve() for pattern in patterns for path in self.root.glob(pattern) if path.is_file()}
            files -= {path.resolve() for pattern in self.config.get("exclude", []) for path in self.root.glob(pattern)}
            return sorted(path for path in files if path.suffix in (".py", ".cir"))
        found = []
        for path in [*self.root.rglob("*.py"), *self.root.rglob("*.cir")]:
            parts = path.relative_to(self.root).parts
            if any(part in self.SKIPPED_DIRECTORIES or part.startswith(".") for part in parts[:-1]):
                continue
            if path.suffix == ".py":
                # Parse (never run) the file and look for @system / @device on a class.
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
                except (SyntaxError, UnicodeDecodeError, ValueError):
                    tree = None  # import it anyway, so the user sees the real error
                except OSError:
                    continue
                decorators = set() if tree is None else {
                    target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                    for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                    for decorator in node.decorator_list
                    for target in [decorator.func if isinstance(decorator, ast.Call) else decorator]
                }
                if tree is not None and not decorators & {"system", "device"}:
                    continue
            found.append(path.resolve())
        return sorted(found)

    def importFile(self, path: Path):
        """Import a file of this workspace as ``codesimpy_user.<relative.path>``; the
        workspace goes on ``sys.path`` so the file can import its sibling modules."""
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        parts = Path(path).relative_to(self.root).with_suffix("").parts
        name = ".".join([self.MODULE_PREFIX, *(
            "".join(char if char.isalnum() or char == "_" else "_" for char in part) for part in parts)])
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    @classmethod
    def purgeModules(cls) -> None:
        """Forget previously imported workspace files so the next import reads them again."""
        for name in [name for name in sys.modules
                     if name == cls.MODULE_PREFIX or name.startswith(cls.MODULE_PREFIX + ".")]:
            del sys.modules[name]


@dataclass
class SystemEntry:
    id: str
    cls: type
    source: str  # "builtin" | "workspace"
    label: str = ""
    category: str = ""
    description: str = ""
    file: str = ""
    line: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "category": self.category,
            "description": self.description, "source": self.source,
            "file": self.file, "line": self.line,
        }


@dataclass
class Registry:
    systems: dict[str, SystemEntry] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    workspace: str | None = None

    def add(self, entry: SystemEntry) -> None:
        existing = self.systems.get(entry.id)
        if existing is not None and existing.cls is not entry.cls:
            where = existing.file or existing.source
            raise ValueError(f"system id {entry.id!r} is already registered by {where}")
        self.systems[entry.id] = entry

    def recordError(self, path: Path, error: BaseException) -> None:
        """Keep a load error of ``path``, with the line in that file where it happened."""
        line = error.lineno if isinstance(error, SyntaxError) and error.lineno else next((
            frame.lineno or 0 for frame in reversed(traceback.extract_tb(error.__traceback__))
            if Path(frame.filename).resolve() == path.resolve()), 0)
        self.errors.append({
            "file": str(path), "line": line,
            "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        })

    def describe(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "systems": [entry.describe() for entry in self.systems.values()],
            "errors": list(self.errors),
        }

    @classmethod
    def discover(cls, workspace: str | Path | None = None) -> Registry:
        """Built-in systems, plus the ``@system`` classes and ``.cir`` netlists found in ``workspace``."""
        from . import systems as builtin
        from .core import NetlistSystem

        registry = cls()
        for name in builtin.__all__:
            systemClass = getattr(builtin, name)
            meta = Meta.of(systemClass)
            registry.add(SystemEntry(
                id=meta.id if meta else name, cls=systemClass, source="builtin",
                label=meta.label if meta else name,
                category=meta.category if meta else "",
                description=meta.description if meta else (inspect.getdoc(systemClass) or "").split("\n")[0],
            ))
        if workspace is None:
            return registry

        folder = Workspace(workspace)
        if not folder.root.is_dir():
            raise NotADirectoryError(f"workspace is not a directory: {folder.root}")
        registry.workspace = str(folder.root)
        Workspace.purgeModules()
        try:
            files = folder.files()
        except Exception as error:  # e.g. malformed pyproject.toml
            registry.recordError(folder.root / "pyproject.toml", error)
            return registry
        # Python files first: they register the user @device models that netlists use.
        for path in sorted(files, key=lambda path: path.suffix == ".cir"):
            try:
                if path.suffix == ".cir":
                    netlistClass = type(path.stem, (NetlistSystem,), {}, netlistFile=str(path))
                    system(id=path.stem, label=netlistClass._title or path.stem, category="Netlist",
                           description=netlistClass._title)(netlistClass)
                    registry.add(SystemEntry(
                        id=path.stem, cls=netlistClass, source="workspace",
                        label=netlistClass._title or path.stem, category="Netlist",
                        description=netlistClass._title, file=str(path), line=1,
                    ))
                    continue
                module = folder.importFile(path)
                for value in list(vars(module).values()):
                    meta = Meta.of(value) if isinstance(value, type) and value.__module__ == module.__name__ else None
                    if meta is not None:
                        registry.add(SystemEntry(
                            id=meta.id, cls=value, source="workspace", label=meta.label,
                            category=meta.category, description=meta.description,
                            file=meta.file or str(path), line=meta.line,
                        ))
            except Exception as error:
                registry.recordError(path, error)
        return registry


discover = Registry.discover
