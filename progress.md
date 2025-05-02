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

**Network Construction**

Establish the objective function of the dictionary: $\min _{x} \lvert y-\Phi x \rvert_{2}^{2}+\beta \lvert x \rvert_{0}$, where $\beta$ controls the sparsity of matrix $x$.

Use the Iterative Hard Thresholding (IHT) method for optimization: $x^{k + 1}=H_{M}(x^{k}+W^{H}(y - \Phi x^{k}))$, where $W=\Phi^{H}$, $S=I-\Phi^{H} \Phi$, and $H_{M}$ is a nonlinear operator.

Simplify the nonlinear operator: In the IVIM model, since the model parameters are non - negative, $H_{M}(x)=\max(x - \lambda, 0)$, where $\lambda$ is a positive threshold.

Estimate the model parameters: After training the dictionary, the parameters are estimated based on $f = I_{1}x$, $D=\frac{\Phi I_{2}x_{1 - f}}{I_{2}x_{1 - f}}$, and $D^{\ast}=\frac{\Phi I_{3}x_{f}}{I_{1}x_{f}}$.
