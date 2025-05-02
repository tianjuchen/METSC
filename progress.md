# METSC microstructure estimation on additive manufactured metallic materials

## Background
We would like to use the transformer to learn the details of the microstructural image subjected
to various creep conditions. And eventually try to build a relation of a time series prediction
model.

## METSC 框架中 IVIM 模型解码器算法解析
In the METSC framework, the decoder algorithm for the IVIM model consists of two main parts: sparse coding and network construction. The specific formula steps are as follows:

**Sparse Coding**

Linearize the IVIM model: $z_{FC}=\Phi x+\eta$, where $z_{FC}$ is composed of the encoded dMRI signals at different b - values obtained by the Transformer encoder, $\Phi$ is the dictionary vector, $x$ is the dictionary coefficient vector, and $\eta$ is the noise term.
