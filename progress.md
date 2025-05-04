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

### 1. Linearizing the IVIM Model

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

### 2. Constructing the Dictionary Vectors

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

### 3. Normalization Processing

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

### 4. Calculating the Model Parameters

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
