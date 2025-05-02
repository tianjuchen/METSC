# METSC microstructure estimation on additive manufactured metallic materials

## Background
We would like to use the transformer to learn the details of the microstructural image subjected
to various creep conditions. And eventually try to build a relation of a time series prediction
model.

## METSC 框架中 IVIM 模型解码器算法解析
在 METSC 框架里，针对 IVIM 模型的解码器算法包含稀疏编码和网络构建两个主要部分，具体公式步骤如下：

**稀疏编码**

将 IVIM 模型线性化：$z_{FC}=\Phi x+\eta$，$z_{FC}$由 Transformer 编码器获取的不同 b 值下编码 dMRI 信号构成，$\Phi$是字典向量，$x$是字典系数向量，$\eta$是噪声项。

构建字典向量：$\Phi=\left[\Phi_{D}, \Phi_{D^{\ast}}\right]$，$x=\left[x_{1-f}, x_{f}\right]^{T}$ 。

归一化处理：为避免分母为 0，将信号归一化到 \[0,1] 区间，$x=\frac{x+\tau}{\| x+\tau\| _{1}}$，$x_{1-f}=\frac{x_{1-f}+\tau}{\left\| x_{1-f}+\tau\right\| _{1}}$，$x_{f}=\frac{x_{f}+\tau}{\left\| x_{f}+\tau\right\| _{1}}$，$\tau = 1e^{-10}$ 。

计算模型参数：$f=I_{1}x$，$D=\frac{\Phi I_{2}x_{1-f}}{I_{2}x_{1-f}}$，$D^{\ast}=\frac{\Phi I_{3}x_{f}}{I_{1}x_{f}}$，$I_{1}$、$I_{2}$ 、$I_{3}$是特定矩阵。

**网络构建**

建立字典的目标函数：$\min _{x}\| y-\Phi x\| _{2}^{2}+\beta\| x\| _{0}$，$\beta$控制矩阵$x$的稀疏性。

采用迭代硬阈值（IHT）方法优化：$x^{k + 1}=H_{M}(x^{k}+W^{H}(y - \Phi x^{k}))$，$W=\Phi^{H}$，$S=I-\Phi^{H} \Phi$，$H_{M}$是非线性算子 。

简化非线性算子：在 IVIM 模型中，因模型参数非负，$H_{M}(x)=\max(x - \lambda, 0)$，$\lambda$是正阈值。

估计模型参数：训练字典后，依据$f=I_{1}x$、$D=\frac{\Phi I_{2}x_{1-f}}{I_{2}x_{1-f}}$和$D^{\ast}=\frac{\Phi I_{3}x_{f}}{I_{1}x_{f}}$估计参数。
