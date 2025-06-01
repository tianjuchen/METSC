# 《METSC 中迭代硬阈值（IHT）方法的实现及优势》

\*The Iterative Hard Thresholding (IHT) method is realized in the Transformer structure of the METSC framework through a combination of model-driven network design and sparse coding techniques, as detailed in the provided document. Here's a step-by-step explanation of its implementation:


### **1. Theoretical Foundation of IHT in METSC**

The IHT method is used to solve the sparse reconstruction problem, which is central to the METSC framework. The core objective function is:


$ 
\min_{x} \left\| z_{FC} - \Phi x \right\|_2^2 + \beta \| x \|_0
 $

where $z_{FC}$ is the encoded dMRI signal from the Transformer encoder, $\Phi$ is the dictionary matrix, $x$ is the vector of dictionary coefficients, and $\beta$ controls the sparsity of $x$. The IHT iteration updates $x$ by projecting onto the set of sparse vectors, which is formalized as:


$ 
x^{n+1} = H_M \left[ \Phi^H z_{FC} + (I - \Phi^H \Phi) x^n \right]
 $

where $H_M$ is a hard thresholding operator that sets values below a threshold $\lambda$ to zero .


### **2. Network Architecture for IHT Implementation**

The IHT process is unfolded into a neural network within the METSC decoder, consisting of two main components:


#### **(1) Sparse Representation Stage**



*   **Dictionary Construction**: The dictionary $\Phi$ is constructed based on discretized parameters of dMRI models (e.g., IVIM or NODDI). For IVIM, $\Phi$ is formed by combining discretized diffusion coefficients $D$ and $D^*$ –.


*   **Iterative Thresholding Layers**: The decoder uses a cascaded structure where each layer implements one IHT iteration. Each layer includes:



    *   A linear layer representing $\Phi^H$ and $(I - \Phi^H \Phi)$.


    *   A hard thresholding layer $H_M$ that zeros out coefficients below $\lambda$ –.


#### **(2) Model-Driven Decoder Design**



*   The decoder is a model-driven neural network that integrates the IHT iterations as learnable layers. Specifically:



    *   The weights $W = \Phi^H$ and $S = I - \Phi^H \Phi$ are shared across layers.


    *   The thresholding operator $H_M$ is adapted for non-negative parameters in IVIM (e.g., $x \geq \lambda$ is retained, otherwise zeroed) .


*   This design allows the network to learn the optimal dictionary and thresholding parameters during training, rather than relying on pre-defined values .


### **3. Integration with Transformer Encoder**



*   **Feature Encoding**: The Transformer encoder first processes the dMRI signal into a high-level representation $z_{FC}$, which captures long-range dependencies in the q-space data –.


*   **Sparse Reconstruction**: The encoded features $z_{FC}$ are fed into the IHT-based decoder, which iteratively refines the sparse coefficients $x$ to fit the dictionary model –.


*   **Parameter Mapping**: The sparse coefficients $x$ are normalized and mapped to microstructural parameters (e.g., $v_{iso}$, $v_{ic}$, $\kappa$) using linear combinations defined by the model equations –.


### **4. Adaptation for dMRI Models**



*   **IVIM Model**: For IVIM, the IHT decoder linearizes the bi-exponential model by separating diffusion and perfusion components. The dictionary $\Phi$ is designed to represent the bi-exponential signal, and the coefficients $x$ are normalized to ensure non-negativity and sum-to-one constraints –.


*   **NODDI Model**: For NODDI, the decoder linearizes the multi-compartment signal model, with the dictionary $\Phi$ capturing intracellular, extracellular, and CSF components. The IHT iterations enforce sparsity in the orientation dispersion and volume fraction coefficients –.


### **5. Training and Optimization**



*   The entire METSC framework (Transformer encoder + IHT decoder) is trained end-to-end using the Adam optimizer. The loss function is the mean squared error (MSE) between predicted and ground-truth microstructural parameters .


*   The IHT parameters (e.g., threshold $\lambda$) are learned during training, allowing the network to adapt to the specific characteristics of dMRI data .


### **6. Key Advantages of IHT in METSC**



*   **Sparsity Induction**: IHT enforces sparse representations, which aligns with the physical sparsity of dMRI signals (e.g., few dominant microstructural components) .


*   **Model Bias Integration**: By unfolding the IHT process into the network, METSC incorporates biophysical model knowledge (e.g., IVIM and NODDI equations) as inductive bias, reducing the need for large training datasets .


*   **Acceleration**: The IHT-based decoder enables parameter estimation with downsampled q-space data, achieving up to 11.25× acceleration for NODDI fitting .


### **Summary**

The IHT method is realized in the METSC Transformer through a model-driven decoder that unrolls iterative sparse reconstruction steps into learnable network layers. This integration allows the framework to leverage both the representational power of Transformer and the physical constraints of dMRI models, resulting in accurate and efficient microstructural parameter estimation.