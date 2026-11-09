"""
第3课作业讲解（约 4–5 分钟）
  作业1：水管检不出，试三种改法
  作业2：校准集不具代表性会怎样
  作业3：模型变大后，INT8 会不会更快
运行：python3 lesson3_homework.py   （与 lesson3_deploy.py、onnx_mini_writer.py 同目录）
"""
import warnings
import numpy as np
import lesson3_deploy as D

warnings.filterwarnings("ignore")


def fresh_data(kind_p=None):
    D.rng = np.random.default_rng(3)                    # 固定种子，保证对比公平
    D.KIND_P = kind_p
    Xtr, ytr, _ = D.make_dataset(4000)
    D.KIND_P = None                                     # 测试集始终是四类均匀，和基线一样
    Xte, yte, kte = D.make_dataset(2000)
    return Xtr, ytr, Xte, yte, kte


def run(name, epochs=12, c1=8, c2=16, kind_p=None):
    Xtr, ytr, Xte, yte, kte = fresh_data(kind_p)
    D.rng = np.random.default_rng(7)
    p = D.train(D.init_params(c1, c2), D.preprocess(Xtr), ytr, epochs=epochs)
    logits, _ = D.forward(p, D.preprocess(Xte))
    D.report(name, logits, yte, kte)
    return p, Xtr, Xte


print("=================== 作业1：水管 ===================")
p_base, Xtr, Xte = run("基线 12轮 8/16通道")
run("训练更久 30轮", epochs=30)
run("模型加宽 16/32通道", c1=16, c2=32)
run("训练集多放水管(50%)", kind_p=[1 / 6, 1 / 2, 1 / 6, 1 / 6])


print("\n=================== 作业2：校准集 ===================")
D.export_fp32(p_base, "hw_fp32.onnx")
s32 = D.session("hw_fp32.onnx")
D.rng = np.random.default_rng(3)
Xtr2, _, _ = D.make_dataset(4000)
Xte2, yte2, kte2 = D.make_dataset(2000)
xte = D.preprocess(Xte2)
o32 = s32.run(None, {"input": xte})[0]
bright = Xtr2.reshape(len(Xtr2), -1).mean(1)
dark10 = Xtr2[np.argsort(bright)[:10]]
print(f"  测试集平均亮度 {Xte2.mean():.0f}，暗光校准图平均亮度 {dark10.mean():.0f}")
for name, calib in [("200张随机图", Xtr2[:200]), ("10张最暗的图", dark10)]:
    q = D.export_int8(p_base, D.preprocess(calib), "hw_int8.onnx")
    o8 = D.session("hw_int8.onnx").run(None, {"input": xte})[0]
    agree = (o8.argmax(1) == o32.argmax(1)).mean()
    print(f"  [{name}] 输入层 scale={q['in'][0]:.4f}  relu2 scale={q['relu2'][0]:.4f}  "
          f"与FP32一致率 {agree:.4f}")
    D.report(f"INT8 {name}", o8, yte2, kte2)


print("\n=================== 作业3：模型多大时 INT8 才划算 ===================")
for hw, c1, c2 in [(32, 8, 16), (128, 32, 64), (256, 64, 128)]:
    D.rng = np.random.default_rng(0)
    p = D.init_params(c1, c2)                           # 只测速度，不需要训练
    x1 = (np.random.default_rng(1).standard_normal((1, 3, hw, hw))).astype(np.float32)
    D.export_fp32(p, "hw_speed_fp32.onnx", hw=hw)
    D.export_int8(p, x1, "hw_speed_int8.onnx", hw=hw)
    s_a, s_b = D.session("hw_speed_fp32.onnx"), D.session("hw_speed_int8.onnx")
    f50, _ = D.bench(lambda: s_a.run(None, {"input": x1}), n=50, warmup=5)
    i50, _ = D.bench(lambda: s_b.run(None, {"input": x1}), n=50, warmup=5)
    macs = (hw * hw * c1 * 27 + (hw // 2) ** 2 * c2 * c1 * 9) / 1e6
    print(f"  输入{hw}×{hw} 通道{c1}/{c2}  计算量 {macs:7.1f} M乘加 | "
          f"FP32 {f50:7.3f} ms  INT8 {i50:7.3f} ms  INT8/FP32 = {i50 / f50:.2f}")
