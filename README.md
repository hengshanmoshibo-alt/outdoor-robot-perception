# 户外移动机器人感知：点云预处理、地面分割、模型部署与相机雷达融合

面向割草机器人等户外低速移动机器人的感知学习项目。在公开数据集（KITTI / SemanticKITTI）上，从点云预处理、地面分割，到视觉模型的 ONNX 部署与 INT8 量化，再到相机–激光雷达目标级融合，逐步实现并量化评测。

> 状态：进行中。下表中“待测”的数字会在实验完成后填入，所有数字均为本人实测。

## 进度

| 模块 | 内容 | 状态 |
|---|---|---|
| `cpp/pcl_hello` | PCL 环境验证 | 待测 |
| `cpp/src/preprocess.cpp` | 裁剪 → 体素 → SOR → RANSAC 去地面 → 欧式聚类，逐步计时 | 待测 |
| `scripts/eval_ground.py` | SemanticKITTI 上 RANSAC vs Patchwork++ 地面分割评测 | 待测 |
| `deploy/` | YOLO 导出 ONNX、自写前后处理、精度对齐、INT8 静态量化 | 未开始 |
| `fusion/` | KITTI raw 上点云投影到图像，检测框 + 点云测距 | 未开始 |
| `simulation/` | 合成数据上的原理验证实验（见下文说明） | 完成 |

## 结果（待填写）

### 预处理耗时（KITTI 序列 __，__ 帧平均，CPU：__）

| 步骤 | 输出点数 | 耗时 (ms) |
|---|---|---|
| 原始 | __ | — |
| 范围裁剪 | __ | __ |
| 体素降采样 (0.10m) | __ | __ |
| 统计离群点去除 | __ | __ |
| RANSAC 去地面 | __ | __ |
| 欧式聚类 | __ 个簇 | __ |

### 地面分割（SemanticKITTI 序列 __，__ 帧）

地面类别口径：road / parking / sidewalk / other-ground / lane-marking / terrain；忽略未标注与植被。

| 方法 | 精确率 | 召回率 | F1 | 耗时中位数 (ms) |
|---|---|---|---|---|
| 单平面 RANSAC | __ | __ | __ | __ |
| Patchwork++ | __ | __ | __ | __ |

### 模型部署（待做）

| 版本 | mAP | 延迟 p50/p99 (ms) | 文件大小 |
|---|---|---|---|
| FP32 | __ | __ | __ |
| INT8 | __ | __ | __ |

## 运行

```bash
# 依赖（Ubuntu 22.04）
sudo apt install -y libpcl-dev libeigen3-dev cmake build-essential
pip install numpy scipy matplotlib open3d opencv-python onnxruntime

# C++ 预处理
cd cpp && mkdir -p build && cd build && cmake .. && make -j
./preprocess <KITTI velodyne 目录> ../../results/preprocess.csv

# 地面分割评测（需先在 Patchwork++ 仓库里 make pyinstall）
python3 scripts/eval_ground.py --root <SemanticKITTI根目录> --seq 08 --frames 50 --out results/ground_08.csv
```

## simulation/：原理验证实验

`simulation/` 里是用 numpy / scipy / OpenCV / ONNX Runtime 在**合成数据**上做的四组实验，用来理解原理和参数取舍，数字不代表真实数据上的表现：

1. 点云基础：裁剪、体素、SOR、RANSAC、聚类，以及参数取舍实验
2. 起伏地形上的分区地面拟合（Patchwork 思想的简化实现）、自适应高度检查、IMU 姿态校正
3. 小 CNN 训练 → 手写 ONNX → ONNX Runtime 推理 → 精度对齐 → INT8 QDQ 量化 → 测速
4. 相机–雷达投影、目标级融合、外参和时间同步误差分析

运行：`cd simulation && python3 lesson1_pointcloud_basics.py`（其余同理，需与同目录文件一起运行）。

## 已知局限

（实验完成后填写：失败案例、参数敏感性、与真实割草场景的差距等。）
