"""
极简 ONNX 写入器（只为教学：本机没有 onnx/torch 包）。

ONNX 文件本质是一个 protobuf 消息 ModelProto：
  ModelProto
    ├─ ir_version / opset_import      版本信息（算子集版本决定每个算子的语义）
    └─ graph: GraphProto
         ├─ input / output           输入输出的名字、类型、形状
         ├─ initializer              权重（TensorProto，raw_data 里是二进制数组）
         └─ node[]                   计算节点：op_type + 输入名 + 输出名 + 属性
节点之间靠“张量名字”连接，形成一张有向无环图。
在你自己的电脑上请用官方 onnx 包（onnx.helper.make_node 等），不要用这个文件。
"""
import struct
import numpy as np

FLOAT, UINT8, INT8, INT64 = 1, 2, 3, 7
_NP = {FLOAT: np.float32, UINT8: np.uint8, INT8: np.int8, INT64: np.int64}


def _varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wt):
    return _varint((field << 3) | wt)


def f_int(field, v):
    return _key(field, 0) + _varint(int(v))


def f_bytes(field, b):
    if isinstance(b, str):
        b = b.encode()
    return _key(field, 2) + _varint(len(b)) + b


def f_float(field, v):
    return _key(field, 5) + struct.pack("<f", v)


def tensor(name, arr, dtype):
    arr = np.ascontiguousarray(arr, dtype=_NP[dtype])
    msg = b"".join(f_int(1, d) for d in arr.shape)
    msg += f_int(2, dtype) + f_bytes(8, name) + f_bytes(9, arr.tobytes())
    return msg


def value_info(name, dtype, shape):
    dims = b""
    for d in shape:
        dim = f_bytes(2, d) if isinstance(d, str) else f_int(1, d)
        dims += f_bytes(1, dim)
    tensor_type = f_int(1, dtype) + f_bytes(2, dims)
    return f_bytes(1, name) + f_bytes(2, f_bytes(1, tensor_type))


def attr(name, v):
    m = f_bytes(1, name)
    if isinstance(v, float):
        m += f_float(2, v) + f_int(20, 1)
    elif isinstance(v, int):
        m += f_int(3, v) + f_int(20, 2)
    else:                                   # list of ints
        m += b"".join(f_int(8, x) for x in v) + f_int(20, 7)
    return m


def node(op, inputs, outputs, name, **attrs):
    m = b"".join(f_bytes(1, i) for i in inputs)
    m += b"".join(f_bytes(2, o) for o in outputs)
    m += f_bytes(3, name) + f_bytes(4, op)
    m += b"".join(f_bytes(5, attr(k, v)) for k, v in attrs.items())
    return m


def model(nodes, initializers, inputs, outputs, opset=13, name="g"):
    g = b"".join(f_bytes(1, n) for n in nodes) + f_bytes(2, name)
    g += b"".join(f_bytes(5, t) for t in initializers)
    g += b"".join(f_bytes(11, i) for i in inputs)
    g += b"".join(f_bytes(12, o) for o in outputs)
    opset_msg = f_bytes(1, "") + f_int(2, opset)
    return f_int(1, 8) + f_bytes(2, "lesson3") + f_bytes(7, g) + f_bytes(8, opset_msg)
