"""
第3课：视觉模型部署全流程（训练 → 导出ONNX → 推理引擎运行 → 精度对齐 → INT8量化 → 测速）

任务：割草机前视相机的 32×32 小图块，判断“有没有低矮障碍物”（玩具、水管、石块），
      正好补第2课点云对 12cm 低矮物体检不好的短板。
本机没有 PyTorch，所以：
  - 用 numpy 手写一个小 CNN 并训练（代替 PyTorch）
  - 用 onnx_mini_writer 手写 ONNX（代替 torch.onnx.export）
  - 手工做 INT8 量化（代替 onnxruntime.quantization.quantize_static）
推理引擎用的是真实的 ONNX Runtime。

运行：python3 lesson3_deploy.py   （约 1–2 分钟）
"""
import time
import numpy as np
import cv2
import onnxruntime as ort
import onnx_mini_writer as W

rng = np.random.default_rng(3)
MEAN = np.array([0.40, 0.50, 0.30], np.float32)       # RGB 均值（训练时统计的）
STD = np.array([0.20, 0.20, 0.20], np.float32)


# ===============================================================
# 0. 造数据：草地图块，一半有障碍物
# ===============================================================
def grass_patch():
    img = np.zeros((32, 32, 3), np.uint8)
    base = rng.integers(60, 120)
    img[:] = (base * 0.5, base + 30, base * 0.35)        # RGB 草绿底色
    for _ in range(rng.integers(40, 80)):                # 草叶纹理
        x, y = rng.integers(0, 32, 2)
        g = int(rng.integers(70, 200))
        cv2.line(img, (int(x), int(y)), (int(x + rng.integers(-3, 4)), int(y - rng.integers(3, 8))),
                 (g // 3, g, g // 4), 1)
    if rng.random() < 0.3:                               # 干扰：泥土斑块 / 阴影（不是障碍物）
        c = (110, 80, 50) if rng.random() < 0.5 else (20, 35, 15)
        cv2.circle(img, tuple(int(v) for v in rng.integers(4, 28, 2)), int(rng.integers(3, 8)), c, -1)
    return img


KIND_P = None   # 各类障碍物的出现比例；None = 四类均匀


def add_obstacle(img):
    kind = rng.choice(4, p=KIND_P) if KIND_P is not None else rng.integers(0, 4)
    cx, cy = (int(v) for v in rng.integers(8, 24, 2))
    if kind == 0:   # 彩色玩具
        col = [(220, 40, 40), (40, 80, 220), (230, 210, 40), (240, 240, 240)][rng.integers(0, 4)]
        cv2.rectangle(img, (cx - 2, cy - 2), (cx + int(rng.integers(1, 4)), cy + 2), col, -1)
    elif kind == 1:  # 绿色水管：颜色接近草，难
        pts = np.array([[cx - 10, cy], [cx - 3, cy - 4], [cx + 4, cy + 2], [cx + 10, cy - 2]], np.int32)
        cv2.polylines(img, [pts], False, (int(rng.integers(30, 70)), int(rng.integers(110, 160)), int(rng.integers(30, 60))), 2)
    elif kind == 2:  # 灰色石块
        g = int(rng.integers(110, 170))
        cv2.ellipse(img, (cx, cy), (int(rng.integers(3, 6)), int(rng.integers(2, 4))), 0, 0, 360, (g, g, g), -1)
    else:            # 深色鞋子/刺猬
        cv2.ellipse(img, (cx, cy), (4, 3), int(rng.integers(0, 180)), 0, 360, (60, 45, 35), -1)
    return img, kind


def blades(img, n):
    for _ in range(n):                                   # 前景草叶：会遮住一部分障碍物
        x, y = rng.integers(0, 32, 2)
        g = int(rng.integers(70, 200))
        cv2.line(img, (int(x), int(y)), (int(x + rng.integers(-3, 4)), int(y - rng.integers(4, 10))),
                 (g // 3, g, g // 4), 1)


def photometric(img):
    """真实相机：光照忽明忽暗、传感器噪声、运动模糊"""
    f = img.astype(np.float32) * rng.uniform(0.55, 1.45)
    f += rng.normal(0, 10, f.shape)
    if rng.random() < 0.4:
        f = cv2.GaussianBlur(f, (3, 3), 0)
    return np.clip(f, 0, 255).astype(np.uint8)


def make_dataset(n):
    X, y, kinds = [], [], []
    for _ in range(n):
        img = grass_patch()
        if rng.random() < 0.5:
            img, k = add_obstacle(img)
            y.append(1); kinds.append(k)
        else:
            y.append(0); kinds.append(-1)
        blades(img, int(rng.integers(5, 25)))
        X.append(photometric(img))
    return np.array(X), np.array(y), np.array(kinds)     # X: N×32×32×3 uint8 RGB


def preprocess(imgs_rgb):
    """训练和部署必须用【完全相同】的前处理：/255 → 减均值除方差 → HWC 转 NCHW"""
    x = imgs_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ===============================================================
# 1. numpy 小 CNN（代替 PyTorch）
#    Conv3×3(3→8)-ReLU-MaxPool2 → Conv3×3(8→16)-ReLU-MaxPool2 → GAP → FC(16→2)
# ===============================================================
def conv_fwd(x, w, b):
    N, C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # N,C,H,W,3,3
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * Wd, C * 9)
    out = cols @ w.reshape(len(w), -1).T + b
    return out.reshape(N, H, Wd, -1).transpose(0, 3, 1, 2), cols


def conv_bwd(dout, x, w, cols):
    N, C, H, Wd = x.shape
    F = len(w)
    d = dout.transpose(0, 2, 3, 1).reshape(-1, F)
    dw = (d.T @ cols).reshape(w.shape)
    db = d.sum(0)
    dcols = (d @ w.reshape(F, -1)).reshape(N, H, Wd, C, 3, 3).transpose(0, 3, 4, 5, 1, 2)
    dxp = np.zeros((N, C, H + 2, Wd + 2), np.float32)
    for i in range(3):
        for j in range(3):
            dxp[:, :, i:i + H, j:j + Wd] += dcols[:, :, i, j]
    return dxp[:, :, 1:-1, 1:-1], dw, db


def pool_fwd(x):
    N, C, H, Wd = x.shape
    r = x.reshape(N, C, H // 2, 2, Wd // 2, 2)
    out = r.max(axis=(3, 5))
    mask = r == out[:, :, :, None, :, None]
    return out, mask


def pool_bwd(dout, mask):
    d = mask * dout[:, :, :, None, :, None]
    N, C, h, _, w, _ = d.shape
    return d.reshape(N, C, h * 2, w * 2)


def init_params(c1=8, c2=16):
    he = lambda *s: (rng.standard_normal(s) * np.sqrt(2 / np.prod(s[1:]))).astype(np.float32)
    return {"w1": he(c1, 3, 3, 3), "b1": np.zeros(c1, np.float32),
            "w2": he(c2, c1, 3, 3), "b2": np.zeros(c2, np.float32),
            "wf": he(2, c2), "bf": np.zeros(2, np.float32)}


def forward(p, x, keep=False):
    c1, cols1 = conv_fwd(x, p["w1"], p["b1"]); r1 = np.maximum(c1, 0); p1, m1 = pool_fwd(r1)
    c2, cols2 = conv_fwd(p1, p["w2"], p["b2"]); r2 = np.maximum(c2, 0); p2, m2 = pool_fwd(r2)
    g = p2.mean(axis=(2, 3))
    logits = g @ p["wf"].T + p["bf"]
    if keep:
        return logits, (x, c1, cols1, m1, p1, c2, cols2, m2, p2, g)
    return logits, {"in": x, "relu1": r1, "pool1": p1, "relu2": r2, "pool2": p2}


def train(p, X, y, epochs=12, bs=64, lr=3e-3):
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    t = 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        loss_sum = 0
        for s in range(0, len(X), bs):
            idx = order[s:s + bs]
            xb, yb = X[idx], y[idx]
            logits, (x, c1, cols1, m1, p1, c2, cols2, m2, p2, g) = forward(p, xb, keep=True)
            z = logits - logits.max(1, keepdims=True)
            prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            loss_sum += -np.log(prob[np.arange(len(yb)), yb] + 1e-9).sum()
            dl = prob; dl[np.arange(len(yb)), yb] -= 1; dl /= len(yb)
            grads = {"wf": dl.T @ g, "bf": dl.sum(0)}
            dg = dl @ p["wf"]
            dp2 = np.broadcast_to(dg[:, :, None, None] / (p2.shape[2] * p2.shape[3]), p2.shape)
            dr2 = pool_bwd(dp2, m2) * (c2 > 0)
            dp1, grads["w2"], grads["b2"] = conv_bwd(dr2, p1, p["w2"], cols2)
            dr1 = pool_bwd(dp1, m1) * (c1 > 0)
            _, grads["w1"], grads["b1"] = conv_bwd(dr1, x, p["w1"], cols1)
            t += 1                                          # Adam
            for k in p:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 3 == 2 or ep == epochs - 1:
            print(f"  epoch {ep + 1:2d}  loss {loss_sum / len(X):.3f}")
    return p


# ===============================================================
# 2. 导出 ONNX（FP32）
# ===============================================================
def export_fp32(p, path, hw=32):
    inits = [W.tensor(k, v, W.FLOAT) for k, v in p.items()]
    nodes = [
        W.node("Conv", ["input", "w1", "b1"], ["c1"], "conv1", pads=[1, 1, 1, 1]),
        W.node("Relu", ["c1"], ["r1"], "relu1"),
        W.node("MaxPool", ["r1"], ["p1"], "pool1", kernel_shape=[2, 2], strides=[2, 2]),
        W.node("Conv", ["p1", "w2", "b2"], ["c2"], "conv2", pads=[1, 1, 1, 1]),
        W.node("Relu", ["c2"], ["r2"], "relu2"),
        W.node("MaxPool", ["r2"], ["p2"], "pool2", kernel_shape=[2, 2], strides=[2, 2]),
        W.node("GlobalAveragePool", ["p2"], ["gap"], "gap"),
        W.node("Flatten", ["gap"], ["flat"], "flatten", axis=1),
        W.node("Gemm", ["flat", "wf", "bf"], ["logits"], "fc", transB=1),
    ]
    ins = [W.value_info("input", W.FLOAT, ["N", 3, hw, hw])]
    outs = [W.value_info("logits", W.FLOAT, ["N", 2])]
    open(path, "wb").write(W.model(nodes, inits, ins, outs))


# ===============================================================
# 3. INT8 量化（QDQ 格式，和 onnxruntime 静态量化产物同一种格式）
#    权重：每个输出通道一个 scale，对称 int8
#    激活：用校准数据统计 min/max，非对称 uint8
#    在 Conv 前后插 QuantizeLinear/DequantizeLinear，ORT 会把它们融合成 int8 卷积
# ===============================================================
def act_qparams(lo, hi):
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = (hi - lo) / 255.0
    zp = int(np.clip(round(-lo / scale), 0, 255))
    return np.float32(scale), np.uint8(zp)


def export_int8(p, calib_x, path, pct=100.0, hw=32):
    _, acts = forward(p, calib_x)
    q = {}
    for name in ["in", "relu1", "pool1", "relu2"]:
        a = acts[name].ravel()
        lo, hi = np.percentile(a, 100 - pct), np.percentile(a, pct)   # pct<100 时裁掉极端值
        q[name] = act_qparams(lo, hi)
    inits, nodes = [], []

    def qdq(src, dst, key):
        s, z = q[key]
        inits.extend([W.tensor(f"{key}_s", s, W.FLOAT), W.tensor(f"{key}_z", z, W.UINT8)])
        nodes.append(W.node("QuantizeLinear", [src, f"{key}_s", f"{key}_z"], [f"{key}_q"], f"q_{key}"))
        nodes.append(W.node("DequantizeLinear", [f"{key}_q", f"{key}_s", f"{key}_z"], [dst], f"dq_{key}"))

    def qweight(name):
        w = p[name]
        s = np.abs(w).reshape(len(w), -1).max(1) / 127.0                   # 每通道 scale
        wq = np.clip(np.round(w / s[:, None, None, None]), -127, 127).astype(np.int8)
        inits.extend([W.tensor(f"{name}_q", wq, W.INT8), W.tensor(f"{name}_s", s, W.FLOAT),
                      W.tensor(f"{name}_z", np.zeros(len(w)), W.INT8)])
        nodes.append(W.node("DequantizeLinear", [f"{name}_q", f"{name}_s", f"{name}_z"], [f"{name}_dq"],
                            f"dq_{name}", axis=0))

    inits += [W.tensor(k, p[k], W.FLOAT) for k in ("b1", "b2", "wf", "bf")]
    qweight("w1"); qweight("w2")
    qdq("input", "x_dq", "in")
    nodes.append(W.node("Conv", ["x_dq", "w1_dq", "b1"], ["c1"], "conv1", pads=[1, 1, 1, 1]))
    nodes.append(W.node("Relu", ["c1"], ["r1"], "relu1"))
    qdq("r1", "r1_dq", "relu1")
    nodes.append(W.node("MaxPool", ["r1_dq"], ["p1"], "pool1", kernel_shape=[2, 2], strides=[2, 2]))
    qdq("p1", "p1_dq", "pool1")
    nodes.append(W.node("Conv", ["p1_dq", "w2_dq", "b2"], ["c2"], "conv2", pads=[1, 1, 1, 1]))
    nodes.append(W.node("Relu", ["c2"], ["r2"], "relu2"))
    qdq("r2", "r2_dq", "relu2")
    nodes += [
        W.node("MaxPool", ["r2_dq"], ["p2"], "pool2", kernel_shape=[2, 2], strides=[2, 2]),
        W.node("GlobalAveragePool", ["p2"], ["gap"], "gap"),
        W.node("Flatten", ["gap"], ["flat"], "flatten", axis=1),
        W.node("Gemm", ["flat", "wf", "bf"], ["logits"], "fc", transB=1),
    ]
    ins = [W.value_info("input", W.FLOAT, ["N", 3, hw, hw])]
    outs = [W.value_info("logits", W.FLOAT, ["N", 2])]
    open(path, "wb").write(W.model(nodes, inits, ins, outs))
    return q


# ===============================================================
# 4. 工具：测速 & 评估
# ===============================================================
def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                         # 模拟端侧单核，结果更稳定
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def bench(fn, n=300, warmup=30):
    for _ in range(warmup):                            # 预热：首次运行有内存分配、图优化开销
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return np.percentile(ts, 50), np.percentile(ts, 99)


KIND_NAMES = {0: "彩色玩具", 1: "绿色水管", 2: "灰色石块", 3: "深色鞋子"}


def report(name, logits, y, kinds):
    pred = logits.argmax(1)
    acc = (pred == y).mean()
    fp = ((pred == 1) & (y == 0)).mean() / (y == 0).mean()
    per = "  ".join(f"{KIND_NAMES[k]}{(pred[kinds == k] == 1).mean():.2f}" for k in range(4))
    print(f"  {name:22s} 准确率 {acc:.3f} | 草地误报率 {fp:.3f} | 各类检出率 {per}")
    return pred


def main():
    print("== 0 造数据 ==")
    Xtr, ytr, _ = make_dataset(4000)
    Xte, yte, kte = make_dataset(2000)
    print(f"  训练 {len(Xtr)} 张，测试 {len(Xte)} 张，32×32 RGB")
    cv2.imwrite("lesson3_samples.png", cv2.cvtColor(
        cv2.resize(np.vstack([np.hstack(Xte[i * 8:(i + 1) * 8]) for i in range(4)]), None, fx=4, fy=4,
                   interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))

    print("== 1 训练（numpy 代替 PyTorch）==")
    p = train(init_params(), preprocess(Xtr), ytr)
    xte = preprocess(Xte)
    ref_logits, _ = forward(p, xte)
    report("numpy 原模型", ref_logits, yte, kte)

    print("== 2 导出 ONNX 并用 ONNX Runtime 推理 ==")
    export_fp32(p, "obstacle_fp32.onnx")
    s32 = session("obstacle_fp32.onnx")
    ort32 = s32.run(None, {"input": xte})[0]
    print(f"  精度对齐：numpy vs ORT 输出最大绝对误差 {np.abs(ort32 - ref_logits).max():.2e}，"
          f"预测一致率 {(ort32.argmax(1) == ref_logits.argmax(1)).mean():.4f}")
    report("ORT FP32", ort32, yte, kte)

    print("== 3 前处理写错会怎样（部署最常见的 bug）==")
    bgr = Xte[..., ::-1]                                # cv2.imread 读出来是 BGR
    report("忘了 BGR→RGB", s32.run(None, {"input": preprocess(bgr)})[0], yte, kte)
    raw = np.ascontiguousarray((Xte.astype(np.float32) / 255).transpose(0, 3, 1, 2))
    report("忘了减均值除方差", s32.run(None, {"input": raw})[0], yte, kte)

    print("== 4 INT8 量化 ==")
    calib = preprocess(Xtr[:200])                       # 校准集：几百张有代表性的图就够
    for pct in (100.0, 99.9):
        q = export_int8(p, calib, f"obstacle_int8_p{pct}.onnx", pct=pct)
        s8 = session(f"obstacle_int8_p{pct}.onnx")
        o8 = s8.run(None, {"input": xte})[0]
        print(f"  [校准用 {pct}% 分位数]  relu2 激活 scale={q['relu2'][0]:.4f}  "
              f"与FP32预测一致率 {(o8.argmax(1) == ort32.argmax(1)).mean():.4f}")
        report(f"ORT INT8 (p{pct})", o8, yte, kte)

    print("== 5 测速（batch=1，单线程，p50/p99 毫秒）==")
    one = xte[:1]
    s8 = session("obstacle_int8_p100.0.onnx")
    for name, s in [("FP32", s32), ("INT8", s8)]:
        p50, p99 = bench(lambda: s.run(None, {"input": one}))
        print(f"  推理 {name}: p50 {p50:.3f} ms  p99 {p99:.3f} ms")
    frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    def pre():                                          # 真实链路：整帧缩放 + 前处理
        small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        return preprocess(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)[None])
    p50, p99 = bench(pre)
    print(f"  前处理(640×480→32×32): p50 {p50:.3f} ms  p99 {p99:.3f} ms")
    import os
    for f in ("obstacle_fp32.onnx", "obstacle_int8_p100.0.onnx"):
        print(f"  文件大小 {f}: {os.path.getsize(f)} 字节")


if __name__ == "__main__":
    main()
