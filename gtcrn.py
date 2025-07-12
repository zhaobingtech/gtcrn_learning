"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        # 保存 kernel size 参数
        self.kernel_size = kernel_size
        # 定义 unfold 操作，用于提取子带特征
        # 在频率轴（第2维）上进行滑动窗口操作
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size),
                               stride=(1, stride),
                               padding=(0, (kernel_size-1)//2))

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (Tensor): 输入张量，形状为 (B,C,T,F)，其中 B 是批量大小，
                        C 是通道数，T 是时间帧数，F 是频率点数。

        返回:
            Tensor: 提取子带特征后的输出张量
        """
        # 使用 unfold 操作提取局部块并重塑张量
        xs = self.unfold(x).reshape(x.shape[0],
                                   x.shape[1]*self.kernel_size,
                                   x.shape[2],
                                   x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention

    功能说明
    TRA (Temporal Recurrent Attention) 模块是一种时序注意力机制，它结合了 GRU 和全连接层来学习输入特征的时间依赖性，
    并将注意力权重应用到原始输入特征上。
    组件解释
    att_gru: GRU 循环神经网络，用于捕捉时间序列的长期依赖关系
    att_fc: 全连接层，将 GRU 输出的高维特征映射回原始通道数
    att_act: Sigmoid 激活函数，生成注意力权重，控制各个时间帧的重要性
    流程解析
    首先计算输入特征每个时间帧的能量（功率）
    然后使用 GRU 提取时序特征
    接着通过全连接层和激活函数生成注意力权重
    最后将注意力权重应用到原始输入特征上，强调重要时间帧的特征，抑制不重要的特征
        """
    def __init__(self, channels):
        super().__init__()
        # 定义 GRU 循环神经网络，用于时序建模
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        # 全连接层，将 GRU 的输出映射回原始通道数
        self.att_fc = nn.Linear(channels*2, channels)
        # Sigmoid 激活函数，生成注意力权重
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (Tensor): 输入张量，形状为 (B,C,T,F)，其中 B 是批量大小，
                        C 是通道数，T 是时间帧数，F 是频率点数。

        返回:
            Tensor: 应用注意力机制后的输出张量，形状与输入相同
        """
        # 计算每个时间帧的能量（功率），形状变为 (B,C,T)
        zt = torch.mean(x.pow(2), dim=-1)

        # 通过 GRU 处理时间序列特征，提取时序依赖关系
        at = self.att_gru(zt.transpose(1,2))[0]

        # 通过全连接层降维，并恢复原始通道数，调整维度顺序回到 (B,C,T)
        at = self.att_fc(at).transpose(1,2)

        # 应用 Sigmoid 激活函数生成注意力权重
        at = self.att_act(at)

        # 添加一个维度扩展注意力权重到 (B,C,T,1)
        At = at[..., None]  # (B,C,T,1)

        # 将注意力权重应用到输入特征上，进行特征增强
        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        # 是否使用反卷积
        self.use_deconv = use_deconv
        # 计算需要填充的时间步数
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        # 根据是否使用反卷积选择相应的卷积模块
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        # 子带特征提取模块
        self.sfe = SFE(kernel_size=3, stride=1)

        # 第一个点卷积层（1x1卷积）
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        # 批归一化层
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        # 激活函数
        self.point_act = nn.PReLU()

        # 深度可分离卷积层
        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        # 批归一化层
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        # 激活函数
        self.depth_act = nn.PReLU()

        # 第二个点卷积层（1x1卷积）
        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        # 批归一化层
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)

        # 时序循环注意力模块
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        # 将两个张量堆叠在一起
        x = torch.stack([x1, x2], dim=1)
        # 调整维度顺序并保持内存连续
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        # 重新排列张量形状
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        # 将输入张量分为两部分
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        # 应用子带特征提取
        x1 = self.sfe(x1)
        # 第一个点卷积操作
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        # 对时间轴进行填充
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        # 深度可分离卷积操作
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        # 第二个点卷积操作
        h1 = self.point_bn2(self.point_conv2(h1))

        # 应用时序循环注意力
        h1 = self.tra(h1)

        # 对处理后的特征和原始的第二部分进行混洗操作
        x =  self.shuffle(h1, x2)
        
        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        # 隐藏层大小
        self.hidden_size = hidden_size
        # RNN 层数
        self.num_layers = num_layers
        # 是否使用双向 RNN
        self.bidirectional = bidirectional

        # 第一个 GRU，处理输入特征的一半通道
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)
        # 第二个 GRU，处理输入特征的另一半通道
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        前向传播函数

        参数:
            x (Tensor): 输入张量，形状为 (B, seq_length, input_size)，其中 B 是批量大小，
                        seq_length 是序列长度，input_size 是输入特征维度。
            h (Tensor): 初始隐藏状态，形状为 (num_layers, B, hidden_size)

        返回:
            y (Tensor): 输出张量，形状为 (B, seq_length, hidden_size)
            h (Tensor): 最终隐藏状态，形状为 (num_layers, B, hidden_size)
        """
        # 如果没有提供初始隐藏状态，则初始化为零
        if h == None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)

        # 将输入特征分为两部分
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        # 将隐藏状态分为两部分
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        # 确保张量在内存中是连续的
        h1, h2 = h1.contiguous(), h2.contiguous()

        # 分别通过两个 GRU 进行处理
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)

        # 将输出和隐藏状态拼接在一起
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)

        return y, h

    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        # 输入特征维度
        self.input_size = input_size
        # 宽度（频率子带数量）
        self.width = width
        # 隐藏层大小
        self.hidden_size = hidden_size

        # Intra RNN 模块，用于处理每个时间帧内的频率依赖关系
        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        # 全连接层，用于处理 Intra RNN 的输出
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        # 层归一化，应用于 Intra RNN 的输出
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter RNN 模块，用于处理时间帧之间的依赖关系
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        # 全连接层，用于处理 Inter RNN 的输出
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        # 层归一化，应用于 Inter RNN 的输出
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (Tensor): 输入张量，形状为 (B, C, T, F)，其中 B 是批量大小，
                        C 是通道数，T 是时间帧数，F 是频率点数。

        返回:
            Tensor: 经过 Dual-path RNN 处理后的输出张量，形状为 (B, C, T, F)
        """
        ## Intra RNN
        # 调整输入张量的维度以便进行时间帧内处理
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        # 将时间帧和批量合并，以便进行独立处理
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        # 应用 Intra RNN 进行时间帧内处理
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        # 应用全连接层
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        # 恢复时间帧和批量维度
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        # 应用层归一化
        intra_x = self.intra_ln(intra_x)
        # 将原始输入与 Intra RNN 输出相加（残差连接）
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        # 调整维度以便进行时间帧间处理
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        # 将时间帧和批量合并，以便进行独立处理
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        # 应用 Inter RNN 进行时间帧间处理
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        # 应用全连接层
        inter_x = self.inter.fc(inter_x)      # (B*F,T,C)
        # 恢复时间帧和批量维度
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        # 调整维度以匹配后续操作
        inter_x = inter_x.permute(0, 2, 1, 3)   # (B,T,F,C)
        # 应用层归一化
        inter_x = self.inter_ln(inter_x)
        # 将 Intra RNN 输出与 Inter RNN 输出相加（残差连接）
        inter_out = torch.add(intra_out, inter_x)

        # 调整维度以匹配输入格式
        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)

        return dual_out



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器由多个卷积块组成，逐步提取并压缩特征
        self.en_convs = nn.ModuleList([
            # 第一个卷积块：输入通道为 3*3（经过 SFE 处理后的特征），输出通道为 16
            # 使用 (1,5) 的 kernel 进行频率轴上的下采样，padding 保持时间维度不变
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),

            # 第二个卷积块：输入输出通道均为 16，使用分组卷积（groups=2）减少计算量
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),

            # 第三个卷积块：使用 GTConvBlock 提取时频特征，不进行下采样
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),

            # 第四个卷积块：空洞卷积，扩大感受野，提升上下文建模能力
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),

            # 第五个卷积块：进一步扩大空洞因子，增强长时依赖建模
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        """
        前向传播函数
        #
        参数:
            x (Tensor): 输入张量，形状为 (B, C, T, F)，其中 B 是批量大小，
                        C 是通道数，T 是时间帧数，F 是频率点数。

        返回:
            Tensor: 编码器最后一层的输出特征
            List[Tensor]: 各层输出特征组成的列表，用于解码器中的跳跃连接
        """
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        """
        应用复数比值掩码来增强频谱。

        参数:
            mask (Tensor): 掩码张量，形状为 (B, 2, T, F)，其中 B 是批量大小，
                           T 是时间帧数，F 是频率点数。
            spec (Tensor): 原始频谱张量，形状为 (B, 2, T, F)

        返回:
            Tensor: 增强后的频谱张量，形状为 (B, 2, T, F)
        """
        # 计算增强后的实部
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        # 计算增强后的虚部
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        # 将实部和虚部堆叠在一起，形成复数频谱
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    def __init__(self):
        super().__init__()
        # ERB 模块用于将输入特征转换为耳蜗谱图
        self.erb = ERB(65, 64)
        # SFE 模块用于提取子带特征
        self.sfe = SFE(3, 1)

        # 编码器模块
        self.encoder = Encoder()

        # DPGRNN 模块，两个重复的模块用于处理编码后的特征
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)

        # 解码器模块
        self.decoder = Decoder()

        # 掩码生成模块
        self.mask = Mask()

    def forward(self, spec):
        """
        前向传播函数

        参数:
            spec (Tensor): 输入频谱，形状为 (B, F, T, 2)，其中 B 是批量大小，
                           F 是频率点数，T 是时间帧数，2 表示实部和虚部。

        返回:
            Tensor: 增强后的频谱，形状为 (B, F, T, 2)
        """
        spec_ref = spec  # 保存原始频谱用于后续计算

        # 提取实部并调整维度
        spec_real = spec[..., 0].permute(0, 2, 1)
        # 提取虚部并调整维度
        spec_imag = spec[..., 1].permute(0, 2, 1)
        # 计算频谱幅度
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        # 构建输入特征，包含幅度、实部和虚部
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        # 应用 ERB 滤波器组进行特征转换
        feat = self.erb.bm(feat)  # (B,3,T,129)
        # 应用 SFE 模块提取子带特征
        feat = self.sfe(feat)     # (B,9,T,129)

        # 编码器处理特征
        feat, en_outs = self.encoder(feat)

        # 第一个 DPGRNN 模块处理编码后的特征
        feat = self.dpgrnn1(feat) # (B,16,T,33)
        # 第二个 DPGRNN 模块进一步处理特征
        feat = self.dpgrnn2(feat) # (B,16,T,33)

        # 解码器处理特征，并结合编码器的输出
        m_feat = self.decoder(feat, en_outs)

        # 应用逆 ERB 滤波器组恢复频谱维度
        m = self.erb.bs(m_feat)

        # 应用掩码生成增强后的频谱
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1)) # (B,2,T,F)
        # 调整维度以匹配输入格式
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        return spec_enh



if __name__ == "__main__":
    model = GTCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print(flops, params)

    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)
    
    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    x2 = torch.stft(x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y1 = model(x1)[0]
    y2 = model(x2)[0]
    y1 = torch.istft(y1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y2 = torch.istft(y2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    
    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
