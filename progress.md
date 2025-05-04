# METSC microstructure estimation on additive manufactured metallic materials

## Background
We would like to use the transformer to learn the details of the microstructural image subjected
to various creep conditions. And eventually try to build a relation of a time series prediction
model.

## Decoder algorithm for the IVIM model
In the METSC framework, the decoder algorithm for the IVIM model consists of two main parts: sparse coding and network construction. The specific formula steps are as follows:

**Sparse Coding**

Linearize the IVIM model: $z_{FC}=\Phi x+\eta$, where $z_{FC}$ is composed of the encoded dMRI signals at different b - values obtained by the Transformer encoder, $\Phi$ is the dictionary vector, $x$ is the dictionary coefficient vector, and $\eta$ is the noise term.

Construct the dictionary vector: $\Phi=\left[\Phi_{D}, \Phi_{D^{\ast}}\right]$, $x=\left[x_{1 - f}, x_{f}\right]^{T}$.

Normalization processing: To avoid a denominator of 0, the signals are normalized to the interval $[0,1]$. 
```math
x =\frac{x+\tau}{\lvert x+\tau \rvert_{1}}
```
```math
x_{1 - f}=\frac{x_{1 - f}+\tau}{\lvert x_{1 - f}+\tau \rvert_{1}}
```
```math
x_{f}=\frac{x_{f}+\tau}{ \lvert x_{f}+\tau \rvert_{1}}
```
wherein, $\tau = 1e^{-10}$.

Calculate the model parameters: $f = I_{1}x$, $D=\frac{\Phi I_{2}x_{1 - f}}{I_{2}x_{1 - f}}$, $D^{\ast}=\frac{\Phi I_{3}x_{f}}{I_{1}x_{f}}$, where $I_{1}$, $I_{2}$, and $I_{3}$ are specific matrices.

## IVIM Coding Illustration
This code example shows how these classes can be used to implement a possible sparse 
coding process for the IVIM model, including linear transformation, dictionary 
handling, sparse reconstruction, and parameter mapping. However, it should be noted 
that the code does not strictly implement the paper's formulas precisely.

### Linearizing the IVIM Model

```math
z_{FC}=\Phi x+\eta
```

There is no direct implementation of this formula in the code. 
However, the convolution operation in the `W_layer` class might be related 
to the calculation of the weights $\Phi$, which is used to perform a linear 
transformation on the input signals.

```python
class W_layer(nn.Module):
    def __init__(self,input):
        super(W_layer, self).__init__()
        W_block = [
            nn.Conv2d(in_channels=60, out_channels=input, kernel_size=1, stride=1, bias=True),
        ]
        self.W_block = nn.Sequential(*W_block)
    def forward(self, x):
        return self.W_block(x)
```

### Constructing the Dictionary Vectors

```math
\Phi=\left[\Phi_{D}, \Phi_{D^{\ast}}\right]
```
```math
x=\left[x_{1-f}, x_{f}\right]^{T}
```
There is no direct construction of such dictionary vectors in the code. 
But the `Dictionary_Block` class might be used to handle operations related 
to the dictionary. The convolutional layer inside it might be used for some 
transformation of the dictionary vectors.
```python
class Dictionary_Block(nn.Module):
    def __init__(self,input):
        super(Dictionary_Block, self).__init__()
        Dict_block= [
            nn.Threshold(0.001, 0, inplace=True),
            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, stride=1, bias=True),
        ]
        self.Dict_block = nn.Sequential(*Dict_block)
    def forward(self, x):
        return self.Dict_block(x)
```

### Normalization Processing

```math
x=\frac{x+\tau}{\lvert x+\tau \rvert_{1}}
```
```math
x_{1-f}=\frac{x_{1-f}+\tau}{\lvert x_{1-f}+\tau \rvert_{1}}
```
```math
x_{f}=\frac{x_{f}+\tau}{\lvert x_{f}+\tau\rvert_{1}}
```

There is no direct implementation of normalization in the code. 
However, the `nn.Threshold` layer might be related to avoiding division 
by zero or handling the data range to some extent.

```python
class Dictionary_Block(nn.Module):
    def __init__(self,input):
        super(Dictionary_Block, self).__init__()
        Dict_block= [
            nn.Threshold(0.001, 0, inplace=True),
            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, stride=1, bias=True),
        ]
        self.Dict_block = nn.Sequential(*Dict_block)
    def forward(self, x):
        return self.Dict_block(x)
```

### Calculating the Model Parameters

```math
f=I_{1}x
```
```math
D=\frac{\Phi I_{2}x_{1-f}}{I_{2}x_{1-f}}
```
```math
D^{\ast}=\frac{\Phi I_{3}x_{f}}{I_{1}x_{f}}
```

The `Mapping` class in the code might be related to calculating the model
parameters. It processes the input through convolutional operations and 
concatenates the results, which might be used to calculate different model 
parameters.

```python
class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()
        model = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600,out_channels=1,kernel_size=1),
        ]
        model1 = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600,out_channels=1,kernel_size=1),
        ]
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
    def forward(self, x):
        output1 = self.model(x)
        output2 = self.model1(x)
        output = torch.cat((output1, output2), dim=1)
        return output
```

**Network Construction**

Establish the objective function of the dictionary: 
```math
\min_{x} \| y-\Phi x \|_{2}^{2}+\beta \| x \|_{0}
```
where $\beta$ controls the sparsity of matrix $x$.

Use the Iterative Hard Thresholding (IHT) method for optimization: 
```math
x^{k + 1}=H_{M}(x^{k}+W^{H}(y - \Phi x^{k}))
```
where $W=\Phi^{H}$, $S=I-\Phi^{H} \Phi$, and $H_{M}$ is a nonlinear operator.

Simplify the nonlinear operator: In the IVIM model, since the model parameters are non - negative, $H_{M}(x)=\max(x - \lambda, 0)$, where $\lambda$ is a positive threshold.

Estimate the model parameters: After training the dictionary $Phi$ and $x$, 
the parameters are estimated based on 
```math
f = I_{1}x
```
```math
D=\frac{\Phi I_{2}x_{1 - f}}{I_{2}x_{1 - f}}
```
```math
D^{\ast}=\frac{\Phi I_{3}x_{f}}{I_{1}x_{f}}
```

One possible approach of the code can be:
```python
import torch
import torch.nn as nn

class Dictionary_Block(nn.Module):
    def __init__(self, input):
        super(Dictionary_Block, self).__init__()
        Dict_block = [
            nn.Threshold(0.001, 0, inplace=True),
            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, stride=1, bias=True),
        ]
        self.Dict_block = nn.Sequential(*Dict_block)

    def forward(self, x):
        return self.Dict_block(x)

class W_layer(nn.Module):
    def __init__(self, input):
        super(W_layer, self).__init__()
        W_block = [
            nn.Conv2d(in_channels=60, out_channels=input, kernel_size=1, stride=1, bias=True),
        ]
        self.W_block = nn.Sequential(*W_block)

    def forward(self, x):
        return self.W_block(x)

class SparseReconstruction(nn.Module):
    def __init__(self):
        super(SparseReconstruction, self).__init__()
        input = 601
        self.dcblock1 = nn.Sequential(
            Dictionary_Block(input)
        )
        self.wblock = nn.Sequential(
            W_layer(input)
        )
        self.activ = nn.Sequential(
            nn.Threshold(0.001, 0, inplace=True),
        )

    def forward(self, x):
        x1 = self.wblock(x)
        y1 = self.dcblock1(x1)
        y1 = x1 + y1
        for i in range(8):
            y1 = x1 + self.dcblock1(y1)
        y1 = self.activ(y1)
        return y1

class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()
        model = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600, out_channels=1, kernel_size=1),
        ]
        model1 = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600, out_channels=1, kernel_size=1),
        ]
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)

    def forward(self, x):
        output1 = self.model(x)
        output2 = self.model1(x)
        output = torch.cat((output1, output2), dim=1)
        return output

# 假设模型已经训练好
sparse_reconstruction = SparseReconstruction()
mapping = Mapping()

# 输入数据
input_tensor = torch.randn(1, 60, 10, 10)

# 进行稀疏重建
sparse_output = sparse_reconstruction(input_tensor)

# 进行参数映射
mapped_output = mapping(sparse_output)

# 这里简单假设 mapped_output 中的两个通道分别对应 f 和 D 的近似值
f_estimated = mapped_output[:, 0, :, :]
D_estimated = mapped_output[:, 1, :, :]

# 对于 D*，由于代码中未明确体现，这里只是示意
# 假设存在一个函数来计算 D*
def calculate_D_star(sparse_output):
    # 这里只是占位实现，需要根据具体公式修改
    return torch.randn_like(f_estimated)

D_star_estimated = calculate_D_star(sparse_output)

print(f"Estimated f shape: {f_estimated.shape}")
print(f"Estimated D shape: {D_estimated.shape}")
print(f"Estimated D* shape: {D_star_estimated.shape}")    
```
## NODDI model

**Sparse Code**

The NODDI (Neurite Orientation Dispersion and Density Imaging) model is a 
magnetic resonance imaging (MRI) model used to quantitatively describe the 
microstructure of biological tissues. The application of sparse coding in 
the NODDI model can help estimate the model parameters more effectively 
from the observed signals. Below, we'll describe the sparse coding process 
for the NODDI model, combining general ideas with sample code.

### Overview of the NODDI Model

The NODDI model typically consists of three main components: isotropic 
water diffusion, intra - neurite diffusion, and extra - neurite diffusion. 
Its signal model can be expressed as:
```math
S(b,\theta)=f_{iso}S_{iso}(b)+f_{nd}S_{nd}(b,\theta)+(1 - f_{iso}-f_{nd})S_{ec}(b,\theta)
```
where $S(b,\theta)$ is the observed signal under the diffusion - sensitive factor 
$b$ and diffusion direction $\theta$, $f_{iso}$ is the volume fraction of 
isotropic water, $f_{nd}$ is the volume fraction of intra - neurite, and $S_{iso}(b)$,
$S_{nd}(b,\theta)$, and $S_{ec}(b,\theta)$ are the signal attenuation 
functions of isotropic, intra - neurite, and extra - neurite, respectively.

### Sparse Coding Concept

The goal of sparse coding is to find a set of sparse coefficients such 
that the observed signal can be approximated by a linear combination of 
these coefficients and dictionary vectors. For the NODDI model, we can 
represent the observed signal $S$ as:
```math
S = \Phi x+\varepsilon
```
where $\Phi$ is the dictionary matrix, $x$ is the sparse coefficient vector, 
and $\varepsilon$ is the noise term.

### Example Code Implementation

```python
import numpy as np
from sklearn.linear_model import Lasso

# Simulate the generation of NODDI model observed signals
def generate_noddi_signal(f_iso, f_nd, b_values, theta):
    # Simple example: Assume the signal attenuation functions for isotropic, intra - neurite, and extra - neurite
    S_iso = np.exp(-b_values * 0.003)
    S_nd = np.exp(-b_values * 0.001)
    S_ec = np.exp(-b_values * 0.002)
    S = f_iso * S_iso + f_nd * S_nd + (1 - f_iso - f_nd) * S_ec
    return S

# Generate the dictionary matrix
def generate_dictionary(b_values, theta_values):
    num_b = len(b_values)
    num_theta = len(theta_values)
    # Assume the dictionary matrix consists of signals with different parameter combinations
    dictionary = []
    for f_iso in np.linspace(0, 1, 10):
        for f_nd in np.linspace(0, 1 - f_iso, 10):
            for theta in theta_values:
                signal = generate_noddi_signal(f_iso, f_nd, b_values, theta)
                dictionary.append(signal)
    dictionary = np.array(dictionary).T
    return dictionary

# Sparse coding
def sparse_coding(S, Phi):
    # Use Lasso regression for sparse coding
    lasso = Lasso(alpha=0.01)
    lasso.fit(Phi, S)
    x = lasso.coef_
    return x

# Parameter settings
b_values = np.linspace(0, 3000, 10)
theta_values = np.linspace(0, np.pi, 5)
f_iso_true = 0.2
f_nd_true = 0.3

# Generate the observed signal
S_obs = generate_noddi_signal(f_iso_true, f_nd_true, b_values, theta_values[0])

# Generate the dictionary matrix
Phi = generate_dictionary(b_values, theta_values)

# Perform sparse coding
x_sparse = sparse_coding(S_obs, Phi)

print("Sparse coefficients:", x_sparse)    
```

### Code Explanation
- **`generate_noddi_signal` function**: Simulates the generation of NODDI model observed 
signals, calculating the signal values based on the given $f_{iso}$, $f_{nd}$, $b$ values,
and $\theta$.
- **`generate_dictionary` function**: Generates the dictionary matrix $\Phi$ by 
traversing different combinations of $f_{iso}$, $f_{nd}$, and $\theta$, calculating 
the corresponding signals and using them as column vectors of the dictionary.
- **`sparse_coding` function**: Uses Lasso regression from the `sklearn` library 
for sparse coding to find the sparse coefficient vector $x$.
- **Main program**: Sets the parameters, generates the observed signal and the 
dictionary matrix, then performs sparse coding and outputs the sparse coefficients.

### Notes
- The signal attenuation functions $S_{iso}$, $S_{nd}$, and $S_{ec}$ in the example 
are simplified forms. In practical applications, they need to be adjusted according 
to the specific definition of the NODDI model.
- The generation method of the dictionary matrix can be optimized according to 
specific requirements, such as increasing the number of sampling points of parameters 
or using a more reasonable parameter range.
- Different algorithms can be chosen for sparse coding, such as Orthogonal 
Matching Pursuit (OMP). Lasso regression is just one of the commonly used methods.

**Sometimes we still need Chinese lol**
```python
import numpy as np
from sklearn.linear_model import Lasso


# 1. Overview of the NODDI Model
# 模拟生成 NODDI 模型的观测信号
def generate_noddi_signal(f_iso, f_nd, b_values, theta):
    """
    模拟生成 NODDI 模型在给定参数下的观测信号。
    :param f_iso: 各向同性水的体积分数
    :param f_nd: 神经突内的体积分数
    :param b_values: 扩散敏感因子
    :param theta: 扩散方向
    :return: 观测信号
    """
    # 简单示例：假设各向同性、神经突内和神经突外的信号衰减函数
    S_iso = np.exp(-b_values * 0.003)
    S_nd = np.exp(-b_values * 0.001)
    S_ec = np.exp(-b_values * 0.002)
    S = f_iso * S_iso + f_nd * S_nd + (1 - f_iso - f_nd) * S_ec
    return S


# 2. Sparse Coding Concept
# 生成字典矩阵
def generate_dictionary(b_values, theta_values):
    """
    生成 NODDI 模型的字典矩阵。
    :param b_values: 扩散敏感因子
    :param theta_values: 扩散方向
    :return: 字典矩阵
    """
    num_b = len(b_values)
    num_theta = len(theta_values)
    # 假设字典矩阵由不同参数组合的信号组成
    dictionary = []
    for f_iso in np.linspace(0, 1, 10):
        for f_nd in np.linspace(0, 1 - f_iso, 10):
            for theta in theta_values:
                signal = generate_noddi_signal(f_iso, f_nd, b_values, theta)
                dictionary.append(signal)
    dictionary = np.array(dictionary).T
    return dictionary


# 稀疏编码
def sparse_coding(S, Phi):
    """
    使用 Lasso 回归进行稀疏编码。
    :param S: 观测信号
    :param Phi: 字典矩阵
    :return: 稀疏系数向量
    """
    # 使用 Lasso 回归进行稀疏编码
    lasso = Lasso(alpha=0.01)
    lasso.fit(Phi, S)
    x = lasso.coef_
    return x


# 主程序
if __name__ == "__main__":
    # 参数设置
    b_values = np.linspace(0, 3000, 10)
    theta_values = np.linspace(0, np.pi, 5)
    f_iso_true = 0.2
    f_nd_true = 0.3

    # 生成观测信号
    S_obs = generate_noddi_signal(f_iso_true, f_nd_true, b_values, theta_values[0])

    # 生成字典矩阵
    Phi = generate_dictionary(b_values, theta_values)

    # 进行稀疏编码
    x_sparse = sparse_coding(S_obs, Phi)

    print("Sparse coefficients:", x_sparse)
    
```

## 基于 NODDI 模型的代码实现与参数计算推测

The equations (23) - (31) in the paper are used to calculate the parameters 
of the NODDI model, including $v_{iso}$ (the volume fraction of cerebrospinal 
fluid), $v_{ic}$ (the intracellular volume fraction), $\kappa$ (a parameter 
related to the orientation dispersion of neurites), and $OD$ (orientation dispersion). 
In the code implementation, although there is no explicit code that exactly 
corresponds to the formulas one - to - one, it can be speculated that 
the `Mapping` class and the `SparseReconstruction` class are related to them. 
The following is a detailed explanation combining the formulas and the code:

### Review of the Formulas

```math
v_{iso}=I_{6}x_{i}
```
```math
x_{t}=\frac{x_{t}+\tau}{\lvert x_{t}+\tau\rvert_{1}}
```
```math
v_{ic}=\frac{\Phi_{t}I_{7}x_{t}}{I_{9}x_{t}}
```
```math
\kappa=\frac{\Phi_{t}I_{8}x_{t}}{I_{9}x_{t}}
```
```math
I_{6} \in \mathbb{R}^{1 \times i}
```
```math
I_{7} \in \mathbb{R}^{2j \times 2j}
```
```math
I_{8} \in \mathbb{R}^{2j \times 2j}
```
```math
I_{9} \in \mathbb{R}^{1 \times 2j} 
```
```math
OD=\frac{2}{\pi} \arctan (\frac{1}{\kappa})
```

### Speculation on Code Implementation

#### `SparseReconstruction` Class

This class is responsible for performing sparse reconstruction on 
the input data to obtain an output containing dictionary coefficients. 
In the sparse coding of the NODDI model, this output may correspond to 
$x_{t}$ and $x_{i}$ in the formulas. By processing the input data through 
multiple iterations, it realizes the transformation from the original 
data to the sparse representation, providing a basis for the subsequent 
calculation of NODDI model parameters.

```python
class SparseReconstruction(nn.Module):
    def __init__(self):
        super(SparseReconstruction, self).__init__()
        input = 601
        self.dcblock1 = nn.Sequential(
            Dictionary_Block(input)
        )
        self.wblock = nn.Sequential(
            W_layer(input)
        )
        self.activ = nn.Sequential(
            nn.Threshold(0.001, 0, inplace=True),
        )
    def forward(self,x):
        x1 = self.wblock(x)
        y1 = self.dcblock1(x1)
        y1 = x1 + y1
        for i in range (8):
            y1 = x1 + self.dcblock1(y1)
        y1 = self.activ(y1)
        return y1
```

#### `Mapping` Class

This class processes the output of the `SparseReconstruction` class 
to calculate the parameters of the NODDI model. The operations of the 
convolutional layers and the threshold layer in it may be used to implement 
the matrix multiplications and normalization operations in the formulas.

```python
class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()
        model = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600,out_channels=1,kernel_size=1),
        ]
        model1 = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600,out_channels=1,kernel_size=1),
        ]
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
    def forward(self, x):
        output1 = self.model(x)
        output2 = self.model1(x)
        output = torch.cat((output1, output2), dim=1)
        return output
```

#### Specific Calculation Corresponding

Suppose the output of `SparseReconstruction` is `sparse_output`, which 
contains dictionary coefficient information. In the `Mapping` class, the 
convolutional layers in `model` and `model1` may be used to calculate the 
intermediate results related to $v_{iso}$, $v_{ic}$, and $\kappa$ respectively. 
For example, the convolutional operation in `model` may correspond to matrix 
multiplications such as $I_{6}x_{i}$, $\Phi_{t}I_{7}x_{t}$, and 
$\Phi_{t}I_{8}x_{t}$; the threshold layer may be related to the normalization 
operation in the formulas, such as the normalization in formula (24). Finally, 
after concatenation, the `output` may contain the calculated parameters 
$v_{iso}$, $v_{ic}$, and $\kappa$. As for the calculation of $OD$, it may not be 
directly reflected in the code, but in practical applications, it can be calculated 
according to formula (31) after obtaining $\kappa$.

The actual code may also need to be combined with specific training processes 
and data processing steps. The above is only a speculative explanation based 
on the given code structure.


## VIM 和 NODDI 解码器的网络构建及各组件作用

The following uses the provided code to illustrate the network construction 
for IVIM and NODDI decoders as described in section 2.2.2.3, analyzing how 
different code components contribute to building the decoder networks.

### General Network Construction Concepts for IVIM and NODDI Decoders

For both IVIM and NODDI decoders, the network construction aims to process 
the input data related to diffusion - weighted signals and estimate the 
relevant model parameters (such as diffusion coefficients, volume fractions, etc.). 
The key is to design layers and operations that can handle the sparse representation 
of data and perform necessary calculations based on the model equations.

### Code Components and Their Roles in Network Construction

#### `Dictionary_Block` Class
```python
class Dictionary_Block(nn.Module):
    def __init__(self, input):
        super(Dictionary_Block, self).__init__()
        Dict_block = [
            nn.Threshold(0.001, 0, inplace=True),
            nn.Conv2d(in_channels=input, out_channels=input, kernel_size=1, stride=1, bias=True),
        ]
        self.Dict_block = nn.Sequential(*Dict_block)

    def forward(self, x):
        return self.Dict_block(x)
```

This class plays a fundamental role in both IVIM and NODDI decoder networks. 
In the context of network construction:

- **Sparsity Enforcement**: The `nn.Threshold` layer helps in enforcing sparsity in 
the data representation. For both IVIM and NODDI models, sparse coding is 
crucial to represent the complex diffusion signals with a minimal set of 
coefficients. By setting values below `0.001` to `0`, it promotes a sparse 
solution, which is in line with the requirements of the sparse - based decoding 
process.

- **Dictionary - Related Operations**: The `nn.Conv2d` with a `1x1` kernel size can 
be seen as an operation related to the dictionary. It may be used to transform 
the input data in a way that aligns with the dictionary vectors, effectively 
performing operations similar to projecting the data onto the dictionary elements. 
This is important for both IVIM and NODDI as the dictionary is used to represent 
the signals in a more interpretable form for parameter estimation.

#### `W_layer` Class
```python
class W_layer(nn.Module):
    def __init__(self, input):
        super(W_layer, self).__init__()
        W_block = [
            nn.Conv2d(in_channels=60, out_channels=input, kernel_size=1, stride=1, bias=True),
        ]
        self.W_block = nn.Sequential(*W_block)

    def forward(self, x):
        return self.W_block(x)
```
This layer is likely involved in the initial transformation of the input data 
for the decoder networks:
- **Data Projection**: The `1x1` convolutional operation in this layer can 
project the input data with `60` channels to a different number of channels 
specified by `input`. In the network construction for IVIM and NODDI decoders, 
this projection step may be used to adjust the data's feature space. 
For example, it could be used to transform the raw diffusion - weighted 
signal data into a space where the subsequent dictionary - based operations 
and parameter calculations become more meaningful. It helps in extracting and 
preparing the relevant features for further processing in the decoder network.


#### `SparseReconstruction` Class
```python
class SparseReconstruction(nn.Module):
    def __init__(self):
        super(SparseReconstruction, self).__init__()
        input = 601
        self.dcblock1 = nn.Sequential(
            Dictionary_Block(input)
        )
        self.wblock = nn.Sequential(
            W_layer(input)
        )
        self.activ = nn.Sequential(
            nn.Threshold(0.001, 0, inplace=True),
        )

    def forward(self, x):
        x1 = self.wblock(x)
        y1 = self.dcblock1(x1)
        y1 = x1 + y1
        for i in range(8):
            y1 = x1 + self.dcblock1(y1)
        y1 = self.activ(y1)
        return y1
```
This class is a key component in the network construction for both IVIM and 
NODDI decoders:

- **Integrated Sparse Reconstruction**: It combines the `Dictionary_Block` and 
`W_layer` to perform a comprehensive sparse reconstruction process. By repeatedly 
applying the `Dictionary_Block` operation in a loop, it refines the sparse 
representation of the input data. For IVIM and NODDI decoders, obtaining an 
accurate sparse representation of the diffusion signals is essential for 
subsequent parameter estimation. The multiple iterations help in optimizing 
the sparse coefficients to better fit the model.

- **Sparsity Refinement**: The final `nn.Threshold` layer in the `activ` 
sequence further refines the sparsity of the output. This ensures that the data 
representation is as sparse as possible while still retaining the necessary 
information for parameter calculation. In the context of both models, a sparse 
and accurate representation of the diffusion data is crucial for the decoder 
to estimate the correct model parameters.


#### `Mapping` Class
```python
class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()
        model = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600, out_channels=1, kernel_size=1),
        ]
        model1 = [
            nn.Threshold(0.0001, 0, inplace=False),
            nn.Conv2d(in_channels=600, out_channels=1, kernel_size=1),
        ]
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)

    def forward(self, x):
        output1 = self.model(x)
        output2 = self.model1(x)
        output = torch.cat((output1, output2), dim=1)
        return output
```

This class is mainly responsible for mapping the sparse - reconstructed 
data to the model parameters in the decoder networks:

- **Parameter Calculation**: The convolutional layers in this class are 
likely used to perform calculations related to the model parameters. 
For example, in the case of IVIM, it may calculate parameters such as the 
diffusion coefficient $D$, perfusion fraction $f$, etc., and for NODDI, it may 
calculate parameters like $v_{iso}$, $v_{ic}$, $\kappa$, and $OD$. The `1x1` 
convolutional operations can be seen as implementing matrix - vector multiplications 
similar to those in the model equations to compute the parameters.

- **Output Merging**: By concatenating the outputs of the two branches 
(`model` and `model1`), it combines different parameter - related calculations 
into a single output. This output represents the estimated parameters of the 
IVIM or NODDI model, which is the final result of the decoder network.


### Overall Network Construction for IVIM and NODDI Decoders

For the IVIM decoder network, the input data related to diffusion - weighted 
signals first passes through the `W_layer` for initial transformation. 
Then, it goes through the `SparseReconstruction` class to obtain a sparse 
representation of the data. Finally, the `Mapping` class takes the 
sparse - reconstructed data and calculates the IVIM model parameters such 
as $D$, $f$, etc.

For the NODDI decoder network, the process is similar. The input data is 
first pre - processed by the `W_layer` and then undergoes sparse reconstruction 
in the `SparseReconstruction` class. The `Mapping` class then uses the 
sparse - reconstructed data to calculate NODDI - specific parameters like 
$v_{iso}$, $v_{ic}$, $\kappa$, and $OD$.

In summary, the combination of these code components forms the core of the network 
construction for both IVIM and NODDI decoders, enabling the processing of 
input data and the estimation of the relevant model parameters.
