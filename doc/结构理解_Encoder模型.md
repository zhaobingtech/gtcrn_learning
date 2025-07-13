# 编码器结构理解



这段代码定义了 GTCRN 模型的 **编码器部分**，其主要作用是逐步提取并压缩输入特征，输出用于解码器重建的中间特征表示。下面我将从结构和流程两个角度详细解释这段代码。

---

## 🧱 编码器整体结构

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([...])
```


- [Encoder](file://D:\10_Python\gtcrn_learning\gtcrn.py#L435-L473) 继承自 `nn.Module`
- 所有卷积层都放在一个 `nn.ModuleList` 中：`self.en_convs`
- 一共包含 **5 个卷积块（ConvBlock 或 GTConvBlock）**

---

## 🔬 输入张量格式

输入张量 [x](file://D:\10_Python\gtcrn_learning\doc\sfe_understanding.py#L11-L11) 的形状为：

```python
(B, C, T, F)
```


| 符号 | 含义 |
|------|------|
| B | Batch size（批量大小） |
| C | Channel（通道数） |
| T | Time（时间帧数） |
| F | Frequency（频率点数） |

---

## 📦 卷积块详解

### 1️⃣ 第一个卷积块

```python
ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False)
```


#### 输入通道数：
- `3*3`: 来自 [SFE](file://D:\10_Python\gtcrn_learning\gtcrn.py#L92-L120) 输出，输入被扩展成了 3×3=9 通道（每个子带 × kernel_size）

#### 输出通道数：
- `16`: 编码后第一层的特征维度

#### 卷积参数：
- `kernel_size=(1,5)`：高度方向不压缩，宽度方向使用 5 点滤波器
- `stride=(1,2)`：在频率轴（F）上进行下采样
- `padding=(0,2)`：保持时间轴不变，频率轴填充保证输出尺寸一致

#### 功能：
- 在频率轴进行降维（F → F/2）
- 提取初步频谱特征

---

### 2️⃣ 第二个卷积块

```python
ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False)
```


#### 输入输出通道数：
- 都是 `16`，即不做通道变换

#### 使用分组卷积：
- `groups=2`：将通道分成 2 组分别处理，减少计算量

#### 卷积参数：
- 同上一个模块，继续在频率轴做下采样（F → F/4）

#### 功能：
- 压缩频率分辨率
- 分组卷积降低参数量

---

### 3️⃣ 第三个卷积块（GTConvBlock）

```python
GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False)
```


#### 结构说明：
- 使用 [GTConvBlock](file://D:\10_Python\gtcrn_learning\gtcrn.py#L228-L298)，结合 SFE、1×1 卷积、深度可分离卷积、注意力机制
- `kernel_size=(3,3)`：时频联合建模
- `dilation=(1,1)`：普通空洞因子

#### 功能：
- 提取局部时频关系
- 不改变分辨率（T, F 不变）

---

### 4️⃣ 第四个卷积块（空洞卷积）

```python
GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False)
```


#### 空洞因子：
- `dilation=(2,1)`：在时间轴扩大感受野，增强上下文建模能力

#### 功能：
- 提取更长时间依赖
- 保持分辨率不变

---

### 5️⃣ 第五个卷积块（更大空洞因子）

```python
GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
```


#### 空洞因子：
- `dilation=(5,1)`：进一步扩大时间轴上的感受野

#### 功能：
- 捕捉长时语音依赖
- 保留原始分辨率

---

## 🔄 前向传播流程

```python
def forward(self, x):
    en_outs = []
    for i in range(len(self.en_convs)):
        x = self.en_convs[i](x)
        en_outs.append(x)
    return x, en_outs
```


### 流程说明：

1. **输入特征**：来自 SFE 层输出，形状 `(B, 9, T, 129)`
2. **经过第一个 ConvBlock**：
   - 下采样频率轴 → `(B, 16, T, 65)`
3. **第二个 ConvBlock**：
   - 再次下采样频率轴 → `(B, 16, T, 33)`
4. **第三个 GTConvBlock**：
   - 保持分辨率不变 → `(B, 16, T, 33)`
5. **第四个 GTConvBlock**：
   - 同上 → `(B, 16, T, 33)`
6. **第五个 GTConvBlock**：
   - 同上 → `(B, 16, T, 33)`

### 最终输出：
- `x`: `(B, 16, T, 33)`，作为编码器最终输出特征
- `en_outs`: 存储每一层输出的列表，用于后续解码器中的跳跃连接（skip connections）

---

## 📈 参数变化图示（假设输入为 `(B, 9, T, 129)`）

| 层 | 类型 | 输入通道 | 输出通道 | 时间轴(T) | 频率轴(F) | 描述 |
|----|------|----------|-----------|------------|-------------|------|
| 0 | ConvBlock | 9 | 16 | 不变 | 129 → 65 | 频率下采样 |
| 1 | ConvBlock | 16 | 16 | 不变 | 65 → 33 | 再次频率下采样 |
| 2 | GTConvBlock | 16 | 16 | 不变 | 33 → 33 | 局部时频建模 |
| 3 | GTConvBlock | 16 | 16 | 不变 | 33 → 33 | 更大感受野 |
| 4 | GTConvBlock | 16 | 16 | 不变 | 33 → 33 | 最大感受野 |

---

## 🧠 总结功能与设计思想

### ✅ 特征提取流程：
1. 通过 SFE 提取子带特征
2. 编码器中逐步压缩频率维度（129 → 33），同时保持时间分辨率
3. 使用 GTConvBlock 提高模型对长时依赖的建模能力
4. 每一层输出保存下来，供解码器跳跃连接使用

### 🎯 设计目标：
- **轻量化**：使用分组卷积、深度可分离卷积、点卷积等结构
- **多尺度建模**：通过不同空洞因子的 GTConvBlock 获取不同感受野
- **信息保留**：通过跳跃连接传递细节信息到解码器

---

## 📊 编码器在整体模型中的位置

```python
feat, en_outs = self.encoder(feat)
```


- `feat`: 编码器最后一层输出的特征，送入 DPGRNN 进一步处理
- `en_outs`: 解码器中跳跃连接所需的中间特征列表

---

## 📚 相关组件参考

| 组件 | 功能 |
|------|------|
| [ConvBlock](file://D:\10_Python\gtcrn_learning\gtcrn.py#L122-L173) | 标准卷积 + BN + PReLU |
| [GTConvBlock](file://D:\10_Python\gtcrn_learning\gtcrn.py#L228-L298) | 子带提取 + 点卷积 + 深度可分离卷积 + 注意力机制 |
| [SFE](file://D:\10_Python\gtcrn_learning\gtcrn.py#L92-L120) | 子带特征提取模块 |
| [DPGRNN](file://D:\10_Python\gtcrn_learning\gtcrn.py#L300-L397) | 双路径 RNN，捕捉时频域长期依赖 |
| [Decoder](file://D:\10_Python\gtcrn_learning\gtcrn.py#L400-L434) | 上采样重建频谱掩码 |

---

## 📌 示例前向传播过程（具体数值）

假设输入形状：`(1, 9, 63, 129)`

| 层 | 输出形状 |
|----|----------|
| 输入 | `(1, 9, 63, 129)` |
| ConvBlock 0 | `(1, 16, 63, 65)` |
| ConvBlock 1 | `(1, 16, 63, 33)` |
| GTConvBlock 2 | `(1, 16, 63, 33)` |
| GTConvBlock 3 | `(1, 16, 63, 33)` |
| GTConvBlock 4 | `(1, 16, 63, 33)` |

最终输出：`feat.shape = (1, 16, 63, 33)`  
跳跃连接特征：`en_outs[0~4]`，每层 `(1, 16, 63, 65) ~ (1, 16, 63, 33)`

---

## ✅ 总结表格

| 编码器层级 | 操作类型 | 输入通道 | 输出通道 | 时间轴变化 | 频率轴变化 | 是否下采样 |
|-----------|----------|----------|-----------|--------------|--------