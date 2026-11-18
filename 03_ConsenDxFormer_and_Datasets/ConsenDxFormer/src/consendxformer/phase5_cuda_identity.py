"""Lazy Linux CUDA Driver API identity only; no contexts, allocations or forwards.

CUDA 12.1 declarations: CUresult/CUdevice are C int; CUuuid is char[16].
Linux CUDAAPI uses the C calling convention (ctypes.CDLL, not WinDLL).
CPU and offline paths use only the pure parsers/validator below.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
import uuid

MAPPING_METHOD = "cuda_driver_uuid_pci_v2"
IDENTITY_SCHEMA = "phase5_cuda_driver_identity_v1"
DRIVER_API = "libcuda.so.1:cuDeviceGetUuid_v2+cuDeviceGetPCIBusId"
UUID_PATTERN = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
PCI_BUFFER_BYTES = 32  # API needs >=13; also accommodates eight-digit domains + NUL.


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]  # char[16] layout, without signed conversion.


def canonical_uuid(value):
    if not isinstance(value, str) or re.fullmatch(UUID_PATTERN, value) is None:
        raise ValueError("Missing/malformed full GPU UUID; MIG is unsupported")
    parsed = uuid.UUID(value[4:])
    if parsed.int == 0:
        raise ValueError("Zero GPU UUID is invalid")
    return "GPU-" + str(parsed)


def pci_components(value):
    """Require full domain:bus:device.function, retaining raw text separately."""
    if not isinstance(value, str) or re.fullmatch(r"(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value) is None:
        raise ValueError("Missing/malformed full PCI address")
    parts = [int(v, 16) for v in re.split(r"[:.]", value)]
    if parts[2] > 31:
        raise ValueError("Invalid PCI device component")
    return parts


def validate_driver_identity(evidence, physical):
    """Recorded evidence only. Never loads or queries a driver."""
    if (not isinstance(evidence, dict) or evidence.get("schema") != IDENTITY_SCHEMA
            or evidence.get("api") != DRIVER_API
            or type(evidence.get("logical_ordinal")) is not int or evidence["logical_ordinal"] != 0
            or type(evidence.get("device_count")) is not int or evidence["device_count"] != 1):
        raise ValueError("Missing/invalid direct CUDA identity evidence")
    raw = evidence.get("uuid_bytes_hex")
    if not isinstance(raw, str) or re.fullmatch(r"[0-9a-f]{32}", raw) is None:
        raise ValueError("Missing/malformed driver UUID bytes")
    actual = canonical_uuid(evidence.get("uuid"))
    if actual != "GPU-" + str(uuid.UUID(bytes=bytes.fromhex(raw))) or actual != canonical_uuid(physical.get("uuid")):
        raise ValueError("Direct CUDA UUID differs from selected physical GPU")
    parts = pci_components(evidence.get("pci_bus_id"))
    recorded = evidence.get("pci_components")
    if (not isinstance(recorded, list) or any(type(v) is not int for v in recorded)
            or recorded != parts or parts != pci_components(physical.get("pci_bus_id"))):
        raise ValueError("Direct CUDA PCI address differs from selected physical GPU")


def _load_driver():
    # Only the installed driver SONAME through the system loader. Never search
    # toolkit paths, stubs, download a library, try alternatives or fall back.
    # Toolkit stubs cannot pass cuInit; its return is checked before enumeration.
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Direct CUDA identity currently requires Linux; no device fallback")
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError("Installed NVIDIA libcuda.so.1 unavailable") from exc


def query_driver_identity():
    """Query isolated ordinal zero via the installed driver; no model/probe calls."""
    canonical_uuid(os.environ.get("CUDA_VISIBLE_DEVICES"))
    driver = _load_driver()
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetUuid_v2": [ctypes.POINTER(CUuuid), ctypes.c_int],
        "cuDeviceGetPCIBusId": [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
    }
    functions = {}
    for name, args in signatures.items():
        try:
            function = getattr(driver, name)
        except AttributeError as exc:
            raise RuntimeError(f"Required CUDA driver symbol unavailable: {name}") from exc
        function.argtypes, function.restype = args, ctypes.c_int
        functions[name] = function

    def checked(name, *args):
        result = functions[name](*args)
        if result != 0:
            raise RuntimeError(f"CUDA driver {name} failed with CUresult={result}")

    checked("cuInit", 0)
    count, device = ctypes.c_int(-1), ctypes.c_int(-1)
    checked("cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise ValueError("Exactly one isolated CUDA driver device is required")
    checked("cuDeviceGet", ctypes.byref(device), 0)  # Never the NVIDIA management index.
    identifier = CUuuid()
    checked("cuDeviceGetUuid_v2", ctypes.byref(identifier), device)
    # Nonzero sentinel ensures a missing terminator is detected, not supplied by us.
    buffer = (ctypes.c_char * PCI_BUFFER_BYTES)(*([b"!"] * PCI_BUFFER_BYTES))
    checked("cuDeviceGetPCIBusId", buffer, PCI_BUFFER_BYTES, device)
    raw_buffer = bytes(buffer)
    if b"\0" not in raw_buffer:
        raise ValueError("Unterminated CUDA PCI address buffer")
    try:
        pci = raw_buffer.split(b"\0", 1)[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Non-ASCII CUDA PCI address") from exc
    raw_uuid = bytes(identifier.bytes)
    value = canonical_uuid("GPU-" + str(uuid.UUID(bytes=raw_uuid)))
    return {"schema": IDENTITY_SCHEMA, "api": DRIVER_API, "logical_ordinal": 0,
            "device_count": count.value, "uuid_bytes_hex": raw_uuid.hex(), "uuid": value,
            "pci_bus_id": pci, "pci_components": pci_components(pci)}
