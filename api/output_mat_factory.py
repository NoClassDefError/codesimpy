"""MATLAB v7.3 data log of a system's raw results (``<model>-snapshot.mat``).

The results are written as they are (NumPy arrays stay arrays, compressed), with no
conversion to the JSON descriptor; :class:`~codesimpy.api.output_json_factory.OutputJsonFactory`
builds that only for the extension.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class OutputMatFactory:
    def __init__(self, system: Any = None, outputDirectory: str | Path | None = None) -> None:
        """[snapshot] The snapshot folder and the raw results logged by :meth:`saveSnapshot`."""
        self.system = system
        self.outputDirectory = Path(outputDirectory) if outputDirectory else Path(__file__).resolve().parents[2] / "codegen"
        self.preserveExisting = True  # restore a previous session's results (saveSnapshot)
        self.latestSimulation: dict[str, Any] = {}
        # Raw results behind the RMS and interaction views of the descriptor.
        self.latestRMSResult: dict[str, Any] | None = None
        self.latestRMSCorrected: dict[str, Any] | None = None
        self.latestInteractionResult: dict[str, Any] | None = None
        self.layout: Any = None  # topology layout of the viewer (layout.save)
        self._restoreChecked = False  # the snapshot file was read once (saveSnapshot)

    def saveSnapshot(self, simulation: dict[str, Any] | None = None, *, layout: Any = None):
        """[snapshot] Write the raw results to ``<model>-snapshot.mat``; return it and the descriptor.

        The file is a compressed MATLAB v7.3 data log (:meth:`saveMat`) of the raw results,
        not of the descriptor: ``system`` (parameters, names, network), ``simulation`` (the
        :meth:`System.simulate` result: ``time_s``, ``state`` N x n, ``signals``, ...),
        ``rms`` / ``rmsCorrected``, ``smallSignal``, ``interaction`` and ``layout``. Read it
        with MATLAB ``load`` or ``mat73.loadmat``. The descriptor, returned for the
        extension, is built by ``system.outputJsonFactory``.

        The first save of a model restores the results of a previous session from the file
        when the parameters are unchanged (the layout always), so they are shown again.
        """
        modelName = re.sub(r'[<>:"/\\|?*]', "-", self.system.name)
        snapshotFile = self.outputDirectory / f"{modelName}-snapshot.mat"
        parameterKey = json.dumps(self.system.outputJsonFactory.jsonSafe(self.system.parameters), sort_keys=True, default=str)
        if self.preserveExisting and not self._restoreChecked and snapshotFile.exists():
            self._restoreChecked = True
            try:
                existing = self.loadMat(snapshotFile)
                restored: dict[str, Any] = {"layout": existing.get("layout")}
                if existing.get("system", {}).get("parameterKey") == parameterKey:
                    if not self.latestSimulation and existing.get("simulation"):
                        restored["simulation"] = existing["simulation"]
                    if not self.system.smallSignal.latestAnalysis and existing.get("smallSignal"):
                        restored["smallSignal"] = existing["smallSignal"]
                    if self.latestRMSResult is None and existing.get("rms"):
                        restored["rms"] = (existing["rms"], existing.get("rmsCorrected"))
                        restored["rmsDescriptor"] = self.system.outputJsonFactory.serializeRMS(*restored["rms"])
                    if self.latestInteractionResult is None and existing.get("interaction"):
                        restored["interaction"] = existing["interaction"]
                        restored["interactionDescriptor"] = self.system.outputJsonFactory.serializeInteractionAnalysis(
                            existing["interaction"])
            except Exception:  # unreadable or from another version: replaced below
                restored = {}
            self.layout = self.layout if self.layout is not None else restored.get("layout")
            if "simulation" in restored:
                self.latestSimulation = restored["simulation"]
            if "smallSignal" in restored:
                self.system.smallSignal.latestAnalysis = restored["smallSignal"]
            if "rms" in restored:
                self.latestRMSResult, self.latestRMSCorrected = restored["rms"]
                self.system.outputJsonFactory.latestRMS = restored["rmsDescriptor"]
            if "interaction" in restored:
                self.latestInteractionResult = restored["interaction"]
                self.system.outputJsonFactory.latestInteractionAnalysis = restored["interactionDescriptor"]
        self._restoreChecked = True
        if simulation is not None:
            self.latestSimulation = simulation
        if layout is not None:
            self.layout = layout

        # The raw results, written as they are (arrays stay arrays).
        variables: dict[str, Any] = {"system": {
            "name": self.system.name,
            "className": f"{type(self.system).__module__}.{type(self.system).__name__}",
            "parameters": self.system.parameters, "parameterKey": parameterKey,
            "stateNames": self.system.stateNames, "referenceNames": self.system.referenceNames,
            "base": self.system.base,
            "network": {key: self.system.network[key] for key in ("A", "R", "L", "busNames")},
        }}
        if self.latestSimulation:
            variables["simulation"] = {
                key: value for key, value in self.latestSimulation.items() if key != "frontendDescriptor"}
        if self.latestRMSResult is not None:
            variables["rms"] = self.latestRMSResult
            if self.latestRMSCorrected is not None:
                variables["rmsCorrected"] = self.latestRMSCorrected
        if self.system.smallSignal.latestAnalysis:
            variables["smallSignal"] = self.system.smallSignal.latestAnalysis
        if self.latestInteractionResult is not None:
            variables["interaction"] = self.latestInteractionResult
        if self.layout is not None:
            variables["layout"] = self.layout
        self.saveMat(snapshotFile, variables)

        descriptor = self.system.outputJsonFactory.getDescriptor(self.latestSimulation)
        if self.layout is not None:
            descriptor["layout"] = self.layout
        return snapshotFile, descriptor

    @staticmethod
    def saveMat(path: str | Path, variables: dict[str, Any]) -> Path:
        """[snapshot] Write ``variables`` as a compressed MATLAB v7.3 file, atomically.

        ``mat73`` only reads MAT files, so the v7.3 layout is written with h5py: a 512-byte
        MATLAB header, then one HDF5 object per variable with ``MATLAB_class`` attributes.
        Values are written as they are, without conversion to lists:

        * ``np.ndarray``: float -> double, integer -> int64, bool -> logical, complex ->
          complex double; MATLAB sees the NumPy shape (a 1-D array as a column). Arrays of
          64 or more elements are gzip-compressed (with shuffle), which MATLAB reads.
        * ``dict``: struct (a key that is not a MATLAB identifier, ``bus-1``, becomes
          ``bus_1``; the original key is kept in the ``codesim_key`` attribute).
        * ``str``: char; ``bool``: logical; other scalars: 1 x 1; ``None``: ``[]``.
        * list of numbers: double row; any other list: cell row (``finalValues{k}.Bus``).
        * anything else (callables, SymPy objects): left out, named in ``codesim_skipped``.

        The ``codesim_*`` attributes (ignored by MATLAB) let :meth:`loadMat` restore the
        original Python values.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(suffix=".mat", dir=path.parent)
        os.close(handle)
        scalar = (int, float, complex, np.number)
        emptyDims = np.zeros(2, dtype=np.uint64)  # MATLAB stores an empty array as its dims
        try:
            with h5py.File(temporary, "w", userblock_size=512, libver="earliest") as file:
                references = file.create_group("#refs#")  # cell elements live here
                counter = [0]

                def write(parent, name: str, value: Any, key: str | None = None):
                    """Write ``value`` as ``parent[name]`` (``key``: the original dict key);
                    return the HDF5 object, or None when the value is not data."""
                    attributes: dict[str, Any] = {}
                    if isinstance(value, (bool, np.bool_)):
                        attributes["codesim_type"] = np.bytes_("scalar")
                        value = np.asarray(value)
                    elif isinstance(value, scalar):
                        attributes["codesim_type"] = np.bytes_(
                            "int" if isinstance(value, (int, np.integer)) else "scalar")
                        value = np.asarray(value)
                    elif isinstance(value, (list, tuple)) and value and all(
                            isinstance(item, scalar) and not isinstance(item, (bool, np.bool_))
                            for item in value):
                        attributes["codesim_type"] = np.bytes_("list")
                        value = np.asarray(value).reshape(1, -1)
                    if isinstance(value, np.ndarray) and value.dtype.kind in "OUS":
                        value = value.tolist()  # strings or objects: a cell
                    if isinstance(value, dict):
                        target = parent.create_group(name)
                        fields, skipped = [], []
                        for itemKey, item in value.items():
                            itemKey = str(itemKey)
                            # MATLAB field names: a letter, then letters, digits, _; at most 63.
                            field = re.sub(r"\W", "_", itemKey, flags=re.ASCII)[:63]
                            if not field or not field[0].isalpha():
                                field = ("x" + field)[:63]
                            base, suffix = field, 1
                            while field in fields:
                                field, suffix = f"{base[:58]}_{suffix}", suffix + 1
                            if write(target, field, item, itemKey if field != itemKey else None) is None:
                                skipped.append(itemKey)
                            else:
                                fields.append(field)
                        attributes["MATLAB_class"] = np.bytes_("struct")
                        # MATLAB_fields: one variable-length array of single characters per field.
                        fieldNames = np.empty(len(fields), dtype=h5py.vlen_dtype(np.dtype("S1")))
                        for index, field in enumerate(fields):
                            fieldNames[index] = np.frombuffer(field.encode("ascii"), dtype="S1")
                        target.attrs.create("MATLAB_fields", fieldNames)
                        if skipped:
                            attributes["codesim_skipped"] = np.bytes_(", ".join(skipped).encode("utf-8"))
                    elif isinstance(value, str):
                        if value:
                            codes = np.frombuffer(value.encode("utf-16-le"), dtype=np.uint16)
                            target = parent.create_dataset(name, data=codes.reshape(-1, 1))
                        else:
                            target = parent.create_dataset(name, data=emptyDims)
                            attributes["MATLAB_empty"] = np.uint8(1)
                        attributes["MATLAB_class"] = np.bytes_("char")
                        attributes["MATLAB_int_decode"] = np.int32(2)
                    elif value is None:
                        target = parent.create_dataset(name, data=emptyDims)
                        attributes.update(MATLAB_class=np.bytes_("double"), MATLAB_empty=np.uint8(1),
                                          codesim_type=np.bytes_("none"))
                    elif isinstance(value, (list, tuple)):
                        if value:
                            cell = np.empty((len(value), 1), dtype=h5py.ref_dtype)
                            for index, item in enumerate(value):
                                counter[0] += 1
                                written = write(references, f"r{counter[0]}", item)
                                if written is None:  # not data: an empty cell element
                                    written = write(references, f"r{counter[0]}", None)
                                cell[index, 0] = written.ref
                            target = parent.create_dataset(name, data=cell)
                        else:
                            target = parent.create_dataset(name, data=emptyDims)
                            attributes["MATLAB_empty"] = np.uint8(1)
                        attributes["MATLAB_class"] = np.bytes_("cell")
                    elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                        kind = value.dtype.kind
                        attributes["MATLAB_class"] = np.bytes_(
                            {"b": "logical", "i": "int64", "u": "int64"}.get(kind, "double"))
                        attributes["codesim_shape"] = np.asarray(value.shape, dtype=np.int64)
                        if value.size == 0:
                            target = parent.create_dataset(name, data=emptyDims)
                            attributes["MATLAB_empty"] = np.uint8(1)
                        else:
                            # MATLAB is column-major: HDF5 holds the transpose. A 1-D array
                            # is an N x 1 column (HDF5 shape (1, N)), a scalar 1 x 1.
                            data = value.reshape(1, -1) if value.ndim < 2 else value.T
                            if kind == "c":
                                complexData = np.empty(data.shape, dtype=[("real", "<f8"), ("imag", "<f8")])
                                complexData["real"], complexData["imag"] = data.real, data.imag
                                data = complexData
                            else:
                                data = data.astype({"b": np.uint8, "i": np.int64, "u": np.int64}.get(kind, np.float64))
                            compressed = ({"compression": "gzip", "compression_opts": 4, "shuffle": True}
                                          if data.size >= 64 else {})
                            target = parent.create_dataset(name, data=data, **compressed)
                    else:
                        return None
                    for attributeName, attributeValue in attributes.items():
                        target.attrs[attributeName] = attributeValue
                    if key is not None:
                        target.attrs["codesim_key"] = np.bytes_(key.encode("utf-8"))
                    return target

                for name, value in variables.items():
                    write(file, name, value)
            # MATLAB header in the HDF5 user block: text, subsystem offset, version 0x0200, 'IM'.
            text = ("MATLAB 7.3 MAT-file, Platform: PCWIN64, Created on: "
                    + time.strftime("%a %b %d %H:%M:%S %Y") + " HDF5 schema 1.00 .")
            header = text.encode("ascii").ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
            with open(temporary, "r+b") as stream:
                stream.write(header.ljust(512, b"\x00"))
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def loadMat(path: str | Path) -> dict[str, Any]:
        """[snapshot] Read a file written by :meth:`saveMat` back into the original values
        (arrays with their shape and dtype, using the ``codesim_*`` attributes)."""
        with h5py.File(path, "r") as file:

            def read(node) -> Any:
                matlabClass, codesimType = (
                    value.decode() if isinstance(value, bytes) else value
                    for value in (node.attrs.get("MATLAB_class"), node.attrs.get("codesim_type")))
                if isinstance(node, h5py.Group):
                    names = ([bytes(field).decode("ascii") for field in node.attrs["MATLAB_fields"]]
                             if "MATLAB_fields" in node.attrs else list(node))
                    return {
                        (node[name].attrs["codesim_key"].decode("utf-8")
                         if "codesim_key" in node[name].attrs else name): read(node[name])
                        for name in names
                    }
                empty = bool(node.attrs.get("MATLAB_empty", 0))
                if matlabClass == "char":
                    return "" if empty else node[()].reshape(-1).astype(np.uint16).tobytes().decode("utf-16-le")
                if matlabClass == "cell":
                    return [] if empty else [read(file[reference]) for reference in node[()].reshape(-1)]
                if codesimType == "none":
                    return None
                shape = tuple(int(size) for size in node.attrs.get("codesim_shape", ()))
                dtype = {"logical": bool, "int64": np.int64}.get(matlabClass, np.float64)
                if empty:
                    return np.zeros(shape, dtype=dtype)
                data = node[()]
                if data.dtype.names:
                    data = data["real"] + 1j * data["imag"]
                else:
                    data = data.astype(dtype)
                if codesimType == "list":
                    return data.reshape(-1).tolist()
                data = data.reshape(shape) if len(shape) < 2 else data.T
                if codesimType == "int":
                    return int(data)
                if codesimType == "scalar":
                    return data.item()
                return data

            return {
                (file[name].attrs["codesim_key"].decode("utf-8") if "codesim_key" in file[name].attrs else name):
                    read(file[name])
                for name in file if name != "#refs#"
            }
