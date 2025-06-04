# 基于 Vision Transformer 的细晶 SEM 图像分析模型及 Hall-Petch 关系整合说明文档

当前的案例仅针对细晶强化下的SEM微观结构的识别，并基于Hall-Petch本构构建的深度学习框架以
更加精确识别微观结构信息，同时标定出主导强化机制参数。通过本案例的搭建，模型将拓展至更为复杂的
微观结构体系，并涵盖多种机制融合的本构ODE，以替代传统本构过多且无法完全描述的情况。

# 概述

本项目旨在构建一个基于 Vision Transformer（ViT）的模型，用于细晶扫描电子显微镜（SEM）图
像的分析。该模型不仅能够预测晶粒尺寸分布，还能结合 Hall-Petch 本构关系计算材料的屈服强度。
以下是对代码的详细解释和 Hall-Petch 关系的说明。

# Hall-Petch 本构关系说明

Hall-Petch 关系描述了多晶体材料的屈服强度与晶粒尺寸之间的关系，其数学表达式为：

```math
\sigma_{y} = \sigma_{0} + kd^{\frac{-1}{2}}
```
其中：
- $\sigma_{y}$ 是材料的屈服强度
- $\sigma_{0}$ 是摩擦应力，代表位错在晶体中运动时所受到的阻力。
- $k$ 是 Hall-Petch常数，反映了晶界对位错运动的阻碍能力。
- $d$ 是平均晶粒尺寸。

该关系表明，材料的屈服强度随着晶粒尺寸的减小而增加，这是因为较小的晶粒具有更多的晶界，
晶界能够阻碍位错的运动，从而提高材料的强度。

# Script 解释

## 导入相关库

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```
这些库提供了深度学习所需的基本功能，如张量操作、神经网络模块等。

## 定义 SEMVIT 类

```python
class SEMViT(nn.Module):
    """基于Vision Transformer的细晶SEM图像分析模型，整合Hall-Petch关系"""
    def __init__(self, 
                 image_size=224, 
                 patch_size=16, 
                 num_classes=1,  # 预测屈服强度
                 dim=768, 
                 depth=12, 
                 heads=12, 
                 mlp_dim=3072,
                 dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # 假设SEM图像为RGB，实际应用中可能需要调整
        
        # 用于晶粒边界检测的预处理卷积层
        self.pre_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 假设输入为单通道灰度图
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # 转换为3通道
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # ViT基础结构
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer编码器
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # 用于晶粒尺寸分布估计的头部
        self.to_grain_size = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),  # 假设估计64个不同的晶粒尺寸区间
            nn.Softmax(dim=-1)
        )
        
        # Hall-Petch关系参数
        self.sigma_0 = nn.Parameter(torch.tensor(10.0))  # 摩擦应力
        self.k = nn.Parameter(torch.tensor(0.5))  # Hall-Petch常数
        
        # 初始化权重
        self.apply(self._init_weights)
```

- 初始化参数：定义了模型的输入图像尺寸、patch 大小、类别数、维度、深度、头数等参数。
- 预处理卷积层：对输入的单通道灰度图进行卷积操作，增强晶粒边界特征，并将其转换为 3 通道图像。
- ViT 基础结构：包括 patch 嵌入、位置编码、分类标记和 dropout 层。
- Transformer 编码器：由多个 Transformer 块组成，用于对输入的特征进行编码。
- 晶粒尺寸分布估计头部：将分类标记的输出映射到 64 个不同的晶粒尺寸区间，并使用 Softmax 函数输出概率分布。
- Hall-Petch 关系参数：定义了摩擦应力 $\sigma_{0}$ 和 Hall-Petch 常数 $k$，并将其作为可学习的参数。
- 权重初始化：使用 Xavier 初始化方法对线性层的权重进行初始化，对 LayerNorm 层的权重和偏置进行常数初始化。



### 归一化方法

```python
self.pre_conv = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),  # 对第一个卷积层的输出(32通道)进行归一化
    nn.ReLU(),
    nn.Conv2d(32, 3, kernel_size=3, padding=1),
    nn.BatchNorm2d(3),   # 对第二个卷积层的输出(3通道)进行归一化
    nn.ReLU()
)
```

- 卷积操作通过滑动窗口提取局部特征，不同通道的特征可能具有不同的尺度和分布。
- BatchNorm 的适配性：
    -   对每个通道单独计算均值和方差，保持了空间维度的特征关系。
    -   在训练过程中，统计量（均值、方差）基于当前批次计算；在推理时，使用训练阶段累积的全局统计量。



| 方法&#xA;          | 计算方式&#xA;               | 适用场景&#xA;                |
| ---------------- | ----------------------- | ------------------------ |
| **BatchNorm**    | 对每个批次的每个通道单独归一化&#xA;    | 大规模数据、CNN&#xA;           |
| **LayerNorm**    | 对每个样本的所有通道和空间位置归一化&#xA; | 序列模型（如 Transformer）&#xA; |
| **InstanceNorm** | 对每个样本的每个通道单独归一化&#xA;    | 风格迁移、生成模型&#xA;           |
| **GroupNorm**    | 将通道分组后归一化&#xA;          | 小批次训练、目标检测&#xA;          |

在你的模型中，`nn.BatchNorm2d` 是最适合卷积层的选择，因为它能有效处理图像的通道间差异。


## 定义前向传播方法

```python
    def forward(self, x):
        # 预处理：增强晶粒边界特征
        x = self.pre_conv(x)
        
        # Patch嵌入
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加分类标记
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 提取分类标记的输出
        cls_output = x[:, 0]
        
        # 预测晶粒尺寸分布
        grain_size_dist = self.to_grain_size(cls_output)
        
        # 基于Hall-Petch关系计算屈服强度
        # 假设我们有一个预设的晶粒尺寸区间列表
        grain_sizes = torch.linspace(1e-6, 100e-6, 64).to(x.device)  # 从1微米到100微米
        mean_grain_size = torch.sum(grain_size_dist * grain_sizes, dim=1, keepdim=True)
        
        # Hall-Petch公式: sigma_y = sigma_0 + k*d^(-1/2)
        d_inv_sqrt = torch.rsqrt(mean_grain_size)
        yield_strength = self.sigma_0 + self.k * d_inv_sqrt
        
        return {
            'grain_size_distribution': grain_size_dist,
            'mean_grain_size': mean_grain_size,
            'yield_strength': yield_strength,
            'hall_petch_params': {'sigma_0': self.sigma_0, 'k': self.k}
        }
```

- 预处理：对输入图像进行卷积操作，增强晶粒边界特征。
- Patch 嵌入：将图像分割成多个 patch，并将其嵌入到高维向量空间中。
- 添加分类标记：在输入特征中添加一个分类标记，用于后续的分类任务。
- Transformer 编码：使用 Transformer 编码器对输入特征进行编码。
- 预测晶粒尺寸分布：提取分类标记的输出，并通过全连接层预测晶粒尺寸分布。
- 计算屈服强度：根据预测的晶粒尺寸分布计算平均晶粒尺寸，并使用 Hall-Petch 公式计算屈服强度。
- 返回结果：返回预测的晶粒尺寸分布、平均晶粒尺寸、屈服强度和 Hall-Petch 关系参数。

## 定义 TransformerBlock 类

```python
class TransformerBlock(nn.Module):
    """Transformer模块，包含自注意力和前馈网络"""
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 自注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = residual + x
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
```

- 初始化参数：定义了 Transformer 块的维度、头数、前馈网络的维度和 dropout 率。
- 自注意力机制：使用多头自注意力机制对输入特征进行加权求和。
- 前馈网络：使用全连接层和 GELU 激活函数对自注意力的输出进行非线性变换。
- 残差连接：在自注意力和前馈网络中使用残差连接，有助于缓解梯度消失问题。

## 定义 HallPetchLoss 类

```python
class HallPetchLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, predictions, targets):
        # 晶粒尺寸分布的KL散度损失
        dist_loss = self.kl_div(
            torch.log(predictions['grain_size_distribution'] + 1e-10), 
            targets['grain_size_distribution']
        )
        
        # 屈服强度的MSE损失
        strength_loss = self.mse(
            predictions['yield_strength'], 
            targets['yield_strength']
        )
        
        # 总损失
        total_loss = self.alpha * dist_loss + (1 - self.alpha) * strength_loss
        return total_loss
```

- 初始化参数：定义了损失函数的权重系数 $\alpha$ 以及均方误差损失函数 $MSE$ 和 $KL$ 散度损失函数。
- 计算损失：分别计算晶粒尺寸分布的 $KL$ 散度损失和屈服强度的 $MSE$ 损失，并根据权重系数计算总损失。

## 总结

本模型通过结合 ***Vision Transformer*** 和 ***Hall-Petch*** 关系，实现了对细晶 $SEM$ 图像的分析和屈服强度的预测。
模型的主要步骤包括**图像预处理**、**Patch 嵌入**、**Transformer 编码**、**晶粒尺寸分布预测**和**屈服强度**计算。同时，
使用自定义的损失函数来优化模型的性能。 