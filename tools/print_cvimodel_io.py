#!/usr/bin/env python3

import ctypes
import json
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int32,
    c_size_t,
    c_uint8,
    c_uint64,
    c_void_p,
)

CVI_DIM_MAX = 6
DEFAULT_LIB_PATH = "/mnt/system/lib/libcviruntime.so"


class CviShape(Structure):
    _fields_ = [("dim", c_int32 * CVI_DIM_MAX), ("dim_size", c_size_t)]


class CviTensor(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("shape", CviShape),
        ("fmt", c_int32),
        ("count", c_size_t),
        ("mem_size", c_size_t),
        ("sys_mem", POINTER(c_uint8)),
        ("paddr", c_uint64),
        ("mem_type", c_int32),
        ("qscale", c_float),
        ("zero_point", c_int),
        ("pixel_format", c_int32),
        ("aligned", c_bool),
        ("mean", c_float * 3),
        ("scale", c_float * 3),
        ("owner", c_void_p),
        ("reserved", c_char * 32),
    ]


def load_runtime():
    lib_path = os.environ.get("CVIRUNTIME_LIB_PATH", DEFAULT_LIB_PATH)
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as err:
        raise SystemExit(
            f"failed to load libcviruntime from {lib_path}: {err}. "
            "Run this script on the Milk-V target or set CVIRUNTIME_LIB_PATH."
        ) from err
    lib.CVI_NN_RegisterModel.argtypes = [c_char_p, POINTER(c_void_p)]
    lib.CVI_NN_RegisterModel.restype = c_int
    lib.CVI_NN_GetInputOutputTensors.argtypes = [
        c_void_p,
        POINTER(POINTER(CviTensor)),
        POINTER(c_int32),
        POINTER(POINTER(CviTensor)),
        POINTER(c_int32),
    ]
    lib.CVI_NN_GetInputOutputTensors.restype = c_int
    lib.CVI_NN_CleanupModel.argtypes = [c_void_p]
    lib.CVI_NN_CleanupModel.restype = c_int
    lib.CVI_NN_GetModelTarget.argtypes = [c_void_p]
    lib.CVI_NN_GetModelTarget.restype = c_char_p
    lib.CVI_NN_GetModelVersion.argtypes = [c_void_p, POINTER(c_int32), POINTER(c_int32)]
    lib.CVI_NN_GetModelVersion.restype = c_int
    return lib


def tensor_to_dict(tensor: CviTensor) -> dict:
    return {
        "name": (tensor.name or b"").decode(errors="replace"),
        "dims": [int(tensor.shape.dim[i]) for i in range(int(tensor.shape.dim_size))],
        "fmt": int(tensor.fmt),
        "count": int(tensor.count),
        "mem_size": int(tensor.mem_size),
        "qscale": float(tensor.qscale),
        "zero_point": int(tensor.zero_point),
        "pixel_format": int(tensor.pixel_format),
        "mean": [float(x) for x in tensor.mean],
        "scale": [float(x) for x in tensor.scale],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/model.cvimodel", file=sys.stderr)
        return 1

    lib = load_runtime()
    model = c_void_p()
    rc = lib.CVI_NN_RegisterModel(sys.argv[1].encode(), byref(model))
    if rc != 0 or not model.value:
        print(f"CVI_NN_RegisterModel failed rc={rc}", file=sys.stderr)
        return 1

    inputs = POINTER(CviTensor)()
    outputs = POINTER(CviTensor)()
    input_num = c_int32()
    output_num = c_int32()
    rc = lib.CVI_NN_GetInputOutputTensors(
        model, byref(inputs), byref(input_num), byref(outputs), byref(output_num)
    )
    if rc != 0:
        lib.CVI_NN_CleanupModel(model)
        print(f"CVI_NN_GetInputOutputTensors failed rc={rc}", file=sys.stderr)
        return 1

    major = c_int32()
    minor = c_int32()
    lib.CVI_NN_GetModelVersion(model, byref(major), byref(minor))

    print(
        json.dumps(
            {
                "target": (lib.CVI_NN_GetModelTarget(model) or b"").decode(errors="replace"),
                "version": [int(major.value), int(minor.value)],
                "inputs": [tensor_to_dict(inputs[i]) for i in range(int(input_num.value))],
                "outputs": [tensor_to_dict(outputs[i]) for i in range(int(output_num.value))],
            },
            indent=2,
        )
    )
    lib.CVI_NN_CleanupModel(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
