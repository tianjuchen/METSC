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