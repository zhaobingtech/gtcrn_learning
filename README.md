# GTCRN

本仓库是 ICASSP2024 论文的官方实现：GTCRN：一种需要超低计算资源的语音增强模型。

音频示例请访问：GTCRN 的音频示例。

## 新闻
- [2025-5-27] 在 H-GTCRN 中发布了一个适用于低信噪比条件的轻量级混合双通道 SE 系统。
- [2025-3-13] 感谢 Fangjun Kuang，构建了一个快速推理网页，请参阅 web 页面。
- [2025-3-10] 感谢 Fangjun Kuang，GTCRN 现在得到了 sherpa-onnx 的支持，请参见此处。
- [2025-3-05] 提出了一种改进的超轻量 SE 模型，名为 UL-UNAS，请参见仓库和 arxiv。

## 关于 GTCRN
分组时间卷积循环网络 (GTCRN) 是一种需要超低计算资源的语音增强模型，仅有 48.2 K 参数和每秒 33.0 MMACs。实验结果表明，我们提出的模型不仅超越了 RNNoise（一个具有类似计算负担的典型轻量级模型），而且与近期需要显著更高计算资源的基线模型相比也具有竞争力。

注意：

论文中报告的复杂度为 23.7K 参数和每秒 39.6 MMACs；然而，我们在本文档中将这些数值更新为 48.2K 参数和每秒 33.0 MMACs。此修改是由于包含了 ERB 模块的参数。即使 ERB 模块的参数是不可学习的，但在计入这些参数后，参数数量增加到了 48.2K。通过用简单连接代替低频维度上线性带到 ERB 带的不变映射，每秒的 MAC 数减少到了 33 MMACs。
在分组 RNN 中的显式特征重排层，通过特征混洗实现，可能导致模型无法流式处理。因此，我们摒弃了这一设计，并通过 DPGRNN 中后续的全连接层隐式地实现了特征重排。

## 性能
实验显示 GTCRN 不仅在 VCTK-DEMAND 和 DNS3 数据集上大幅超越 RNNoise，而且与几个计算开销显著更高的基线模型相比也表现出竞争力。

### 表 1：VCTK-DEMAND 测试集上的性能

| | Para. (M) | MACs (G/s) | SISNR | PESQ | STOI |
|:--:|:-------:|:--------:|:---:|:--:|:--:|
| Noisy | - | - | 8.45 | 1.97 | 0.921 |
| RNNoise (2018) | 0.06 | 0.04 | - | 2.29 | - |
| PercepNet (2020) | 8.00 | 0.80 | - | 2.73 | - |
| DeepFilterNet (2022) | 1.80 | 0.35 | 16.63 | 2.81 | 0.942 |
| S-DCCRN (2022) | 2.34 | - | - | 2.84 | 0.940 |
| GTCRN (proposed) | 0.05 | 0.03 | 18.83 | 2.87 | 0.940 |

### 表 2：DNS3 盲测测试集上的性能

| | Para. (M) | MACs (G/s) | DNSMOS-P.808 | BAK | SIG | OVRL |
|:--:|:-------:|:--------:|:----------:|:-:|:-:|:--:|
| Noisy | - | - | 2.96 | 2.65 | 3.20 | 2.33 |
| RNNoise (2018) | 0.06 | 0.04 | 3.15 | 3.45 | 3.00 | 2.53 |
| S-DCCRN (2022) | 2.34 | - | 3.43 | - | - | - |
| GTCRN (proposed) | 0.05 | 0.03 | 3.44 | 3.90 | 3.00 | 2.70 |

## 预训练模型

预训练模型位于 `checkpoints` 文件夹中，分别在 DNS3 和 VCTK-DEMAND 数据集上进行了训练。

推理过程在 [infer.py](file://D:\10_Python\gtcrn_learning\infer.py) 中展示。

## 流式推理

在 `stream` 文件夹中提供了支持流式处理的 GTCRN，其在第12代 Intel(R) Core(TM) i5-12400 CPU @ 2.50 GHz 上实现了惊人的实时因子 (RTF) **0.07**。

## 相关仓库

[SEtrain](https://github.com/Xiaobin-Rong/SEtrain): 基于深度神经网络语音增强的训练代码模板。

[TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE): 如何将语音增强模型转换为流式格式并使用 ONNX 或 TensorRT 进行部署的示例。

# GTCRN
This repository is the official implementation of the ICASSP2024 paper: [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources](https://ieeexplore.ieee.org/document/10448310). 

Audio examples are available at [Audio examples of GTCRN](https://htmlpreview.github.io/?https://github.com/Xiaobin-Rong/gtcrn_demo/blob/main/index.html).

## 🔥 News
- [**2025-5-27**] A lightweight hybrid dual-channel SE system adapted for low-SNR conditions is released in [H-GTCRN](https://github.com/Max1Wz/H-GTCRN).
- [**2025-3-13**] A quick inference web is built thanks to [Fangjun Kuang](https://github.com/csukuangfj), see in [web](https://huggingface.co/spaces/k2-fsa/wasm-speech-enhancement-gtcrn).
- [**2025-3-10**] Now GTCRN is supported by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) thanks to [Fangjun Kuang](https://github.com/csukuangfj), see [here](https://github.com/k2-fsa/sherpa-onnx/pull/1977).
- [**2025-3-05**] An improved ultra-lightweight SE model named **UL-UNAS** is proposed, see in [repo](https://github.com/Xiaobin-Rong/ul-unas) and [arxiv](https://arxiv.org/abs/2503.00340).

## About GTCRN
Grouped Temporal Convolutional Recurrent Network (GTCRN) is a speech enhancement model requiring ultralow computational resources, featuring only **48.2 K** parameters and **33.0 MMACs** per second.
Experimental results show that our proposed model not only surpasses RNNoise, a typical lightweight model with similar computational burden, 
but also achieves competitive performance when compared to recent baseline models with significantly higher computational resources requirements.

Note:
* The complexity reported in the paper is **23.7K** parameters and **39.6 MMACs** per second; however, we update these values to **48.2K** parameters and **33.0 MMACs** per second here. This modification is due to the inclusion of the ERB module. When accounting for the parameters of the ERB module (even though they are unlearnable), the parameter count increases to 48.2K. By replacing the invariant mapping from linear bands to ERB bands in the low-frequency dimension with simple concatenation instead of matrix multiplication, the MACs per second are reduced to 33 MMACs.
* The explicit feature rearrangement layer in the grouped RNN, which is implemented by feature shuffle, can result in an unstreamable model. Therefore, we discard it and implicitly achieve feature rearrangement through the following FC layer in the DPGRNN.

## Performance
Experiments show that GTCRN not only outperforms RNNoise by a substantial margin on the VCTK-DEMAND and DNS3 dataset, but also achieves competitive performance compared to several baseline models with significantly higher computational overhead.

**Table 1**: Performance on VCTK-DEMAND test set
|    |Para. (M)|MACs (G/s)|SISNR|PESQ|STOI|
|:--:|:-------:|:--------:|:---:|:--:|:--:|
|Noisy|-|-|8.45|1.97|0.921
|RNNoise (2018)|0.06|0.04|-|2.29|-|
|PercepNet (2020)|8.00|0.80|-|2.73|-|
|DeepFilterNet (2022)|1.80|0.35|16.63|2.81|**0.942**|
|S-DCCRN (2022)|2.34|-|-|2.84|0.940|
|GTCRN (proposed)|**0.05**|**0.03**|**18.83**|**2.87**|0.940|
<br>

**Table 2**: Performance on DNS3 blind test set.
|    |Para. (M)|MACs (G/s)|DNSMOS-P.808|BAK|SIG|OVRL|
|:--:|:-------:|:--------:|:----------:|:-:|:-:|:--:|
|Noisy|-|-|2.96|2.65|**3.20**|2.33|
|RNNoise (2018)|0.06|0.04|3.15|3.45|3.00|2.53|
|S-DCCRN (2022)|2.34|-|3.43|-|-|-|
|GTCRN (proposed)|**0.05**|**0.03**|**3.44**|**3.90**|3.00|**2.70**|

## Pre-trained Models
Pre-trained models are provided in `checkpoints` folder, which were trained on DNS3 and VCTK-DEMAND datasets, respectively.

The inference procedure is presented in `infer.py`.

## Streaming Inference
A streaming GTCRN is provided in `stream` folder, which demonstrates an impressive real-time factor (RTF) of **0.07** on the 12th Gen Intel(R) Core(TM) i5-12400 CPU @ 2.50 GHz.

## Related Repositories
[SEtrain](https://github.com/Xiaobin-Rong/SEtrain): A training code template for DNN-based speech enhancement.

[TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE): An example of how to convert a speech enhancement model into a streaming format and deploy it using ONNX or TensorRT.
