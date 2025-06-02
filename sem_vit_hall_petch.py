import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
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

# 自定义损失函数：结合晶粒尺寸分布和Hall-Petch预测误差
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