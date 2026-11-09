# 模型部署（计划第 8–10 天）

1. 取 YOLO 最小尺寸预训练模型，导出 ONNX。
2. 自写 OpenCV 前处理（letterbox、BGR→RGB、归一化、NCHW）和后处理（解码、NMS）。
3. 用 ONNX Runtime 推理，与原框架输出做精度对齐（最大绝对误差、框的一致率）。
4. `onnxruntime.quantization.quantize_static` 做 INT8（QDQ 格式、按通道），用几百张有代表性的图校准。
5. 对比 FP32 / INT8 的 mAP、延迟（p50/p99，含前后处理）、文件大小。
