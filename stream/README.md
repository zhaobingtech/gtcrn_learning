# 流式 GTCRN
这里使用的流式转换方法与 [TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE) 仓库中的流式转换方法一致。然而，由于 GTCRN 的超低复杂度，我们仅使用 ONNX 进行部署，并在 CPU 上执行推理。

## 致谢
我想感谢 [Linghao Yu](https://github.com/SherryYu33) 解决了简化 [gtcrn.onnx](file://D:\10_Python\gtcrn_learning\stream\onnx_models\gtcrn.onnx) 文件时的一个 bug。更多详情请参阅 [issue-3](https://github.com/Xiaobin-Rong/gtcrn/issues/3)。

# Streaming GTCRN
The streaming conversion method used here is consistent with the streaming conversion in the [TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE) repository. However, we only deploy it using ONNX and perform inference on the CPU, thanks to the ultra-low complexity of GTCRN.

## Acknowledgement
I would like to express my gratitude to [Linghao Yu](https://github.com/SherryYu33) for resolving a bug in simplifying the `gtcrn.onnx` file. More details can be found in [issue-3](https://github.com/Xiaobin-Rong/gtcrn/issues/3).

