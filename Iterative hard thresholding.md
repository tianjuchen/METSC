# IHT method implementation in METSC

The Iterative Hard Thresholding (IHT) method is realized in the 
Transformer structure of the METSC framework through a combination of 
model-driven network design and sparse coding techniques, as detailed 
in the provided document. Here's a step-by-step explanation of its implementation:


### **Theoretical Foundation of IHT in METSC**

The IHT method is used to solve the sparse reconstruction problem, which is central to the METSC framework. The core objective function is:


$$ \min_{x} \Vert z_{FC} - \Phi x \Vert_{2}^{2} + \beta \Vert x \Vert_{0} $$

where $z_{FC}$ is the encoded dMRI signal from the Transformer encoder, $\Phi$ is the dictionary matrix, $x$ is the vector of dictionary coefficients, and $\beta$ controls the sparsity of $x$. The IHT iteration updates $x$ by projecting onto the set of sparse vectors, which is formalized as:


$$ x^{n+1} = H_{M} [\Phi^{H} z_{FC} + (I - \Phi^{H} \Phi) x^{n}] $$

where $H_M$ is a hard thresholding operator that sets values below a threshold $\lambda$ to zero .


### **Iterative Hard Thresholding Layer Implementation**

The IHT layers are part of the sparse representation stage in the METSC decoder. The key components 
include dictionary operations, iterative updates, and thresholding. Here's a code example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class IHTLayer(nn.Module):
    """Iterative Hard Thresholding layer for sparse reconstruction"""
    def __init__(self, dict_dim, threshold=0.001, use_nonneg=True):
        super(IHTLayer, self).__init__()
        self.threshold = threshold
        self.use_nonneg = use_nonneg
        
        # Dictionary transpose (Φ^H) and (I - Φ^HΦ) operations
        # In practice, these would be learned during training
        self.W = nn.Linear(dict_dim, dict_dim, bias=False)  # Represents Φ^H
        self.S = nn.Linear(dict_dim, dict_dim, bias=False)  # Represents (I - Φ^HΦ)
        
        # Initialize weights to identity for simplicity
        nn.init.eye_(self.W.weight)
        nn.init.eye_(self.S.weight)

    def hard_thresholding(self, x):
        """Hard thresholding operator H_M"""
        if self.use_nonneg:
            # For non-negative parameters (e.g., IVIM)
            return F.threshold(x, self.threshold, 0)
        else:
            # General hard thresholding
            return torch.where(
                torch.abs(x) >= self.threshold, 
                x, 
                torch.zeros_like(x)
            )

    def forward(self, z_FC, x_prev):
        """
        z_FC: Encoded features from Transformer (shape: [batch, dict_dim])
        x_prev: Previous iteration's sparse coefficients (shape: [batch, dict_dim])
        """
        # Compute Φ^H z_FC
        w_z = self.W(z_FC)
        # Compute (I - Φ^HΦ) x_prev
        s_x = self.S(x_prev)
        # Iterative update: x^{n+1} = H_M(Φ^H z_FC + (I - Φ^HΦ) x^n)
        x_next = self.hard_thresholding(w_z + s_x)
        return x_next


class SparseDecoder(nn.Module):
    """Sparse representation decoder with IHT iterations"""
    def __init__(self, dict_dim, num_iterations=8, threshold=0.001, use_nonneg=True):
        super(SparseDecoder, self).__init__()
        self.num_iterations = num_iterations
        # Create multiple IHT layers
        self.iht_layers = nn.ModuleList([
            IHTLayer(dict_dim, threshold, use_nonneg) 
            for _ in range(num_iterations)
        ])
        self.threshold = threshold
        self.use_nonneg = use_nonneg

    def forward(self, z_FC):
        """
        z_FC: Encoded features from Transformer (shape: [batch, dict_dim])
        """
        batch_size = z_FC.shape[0]
        # Initialize sparse coefficients as zeros
        x = torch.zeros(batch_size, z_FC.shape[1]).to(z_FC.device)
        
        # Perform IHT iterations
        for layer in self.iht_layers:
            x = layer(z_FC, x)
        
        # Normalize coefficients (e.g., for IVIM/NODDI)
        x = (x + 1e-10) / torch.norm(x + 1e-10, p=1, dim=1, keepdim=True)
        return x
```
### **Code Explanation**

#### ** **`IHTLayer`** Class**



*   **Dictionary Operations**: The `W` and `S` linear layers represent $\Phi^H$ and $(I - \Phi^H\Phi)$, which are learned during training to fit the dictionary model .


*   **Hard Thresholding**: The `hard_thresholding` method implements the $H_M$ operator, zeroing coefficients below a threshold. For IVIM, non-negativity is enforced .


*   **Iterative Update**: Each layer computes $x^{n+1} = H_M(\Phi^H z_{FC} + (I - \Phi^H\Phi) x^n)$, aligning with the IHT update equation .


#### ** **`SparseDecoder`** Class**



*   **Iteration Stacking**: Multiple `IHTLayer` instances are stacked (e.g., 8 iterations) to unfold the IHT process .


*   **Initialization & Normalization**: Sparse coefficients start as zeros and are normalized after iterations to satisfy constraints (e.g., sum-to-one for volume fractions) .


### **Integration with Transformer Encoder**

The `SparseDecoder` would be integrated into the METSC framework as part of the decoder stage:
```python
class METSC(nn.Module):
    def __init__(self, dict_dim=600, num_iterations=8):
        super(METSC, self).__init__()
        # Transformer encoder (simplified here)
        self.transformer_encoder = nn.Sequential(
            # ... Transformer layers from the paper ...
        )
        # Sparse decoder with IHT layers
        self.sparse_decoder = SparseDecoder(dict_dim, num_iterations)
        # Mapping layer to microstructural parameters
        self.mapping = nn.Sequential(
            nn.Conv2d(dict_dim, 3, kernel_size=1),  # Example for NODDI (v_iso, v_ic, OD)
        )

    def forward(self, x):
        # Transformer encoding
        z_FC = self.transformer_encoder(x)
        # Sparse reconstruction via IHT
        sparse_coeffs = self.sparse_decoder(z_FC)
        # Map to microstructural parameters
        params = self.mapping(sparse_coeffs)
        return params
```

### **Key Alignments with the Document**



1.  **Dictionary Learning**: The `W` and `S` layers correspond to the dictionary operations in Eq. (33)–(35) .


2.  **Thresholding**: The `hard_thresholding` function implements the non-negative thresholding in Eq. (37) .


3.  **Iterative Unfolding**: The `SparseDecoder` with `num_iterations=8` matches the 8 IHT iterations mentioned in the code explanation .


4.  **Normalization**: The post-processing step aligns with Eq. (24) for NODDI and Eq. (11)–(13) for IVIM .


### **Practical Considerations**



*   **Dictionary Initialization**: In practice, the dictionary $\Phi$ is initialized based on discretized model parameters (e.g., IVIM's $D$ and $D^*$) .


*   **Shared Weights**: The `W` and `S` weights are shared across IHT layers, as stated in the document .


*   **Training**: The entire network is trained end-to-end with MSE loss, adapting dictionary parameters and thresholding values .


This implementation demonstrates how the IHT method is translated into executable code, integrating 
model-driven sparsity with Transformer features for dMRI microstructure estimation.


### Network Architecture for IHT Implementation**

The IHT process is unfolded into a neural network within the METSC decoder, consisting of two main components:


#### **Sparse Representation Stage**



*   **Dictionary Construction**: The dictionary $\Phi$ is constructed based on discretized parameters of dMRI models (e.g., IVIM or NODDI). For IVIM, $\Phi$ is formed by combining discretized diffusion coefficients $D$ and $D^*$ –.


*   **Iterative Thresholding Layers**: The decoder uses a cascaded structure where each layer implements one IHT iteration. Each layer includes:



    *   A linear layer representing $\Phi^H$ and $(I - \Phi^H \Phi)$.


    *   A hard thresholding layer $H_M$ that zeros out coefficients below $\lambda$ –.


#### **Model-Driven Decoder Design**



*   The decoder is a model-driven neural network that integrates the IHT iterations as learnable layers. Specifically:



    *   The weights $W = \Phi^H$ and $S = I - \Phi^H \Phi$ are shared across layers.


    *   The thresholding operator $H_M$ is adapted for non-negative parameters in IVIM (e.g., $x \geq \lambda$ is retained, otherwise zeroed) .


*   This design allows the network to learn the optimal dictionary and thresholding parameters during training, rather than relying on pre-defined values .


### **Integration with Transformer Encoder**



*   **Feature Encoding**: The Transformer encoder first processes the dMRI signal into a high-level representation $z_{FC}$, which captures long-range dependencies in the q-space data –.


*   **Sparse Reconstruction**: The encoded features $z_{FC}$ are fed into the IHT-based decoder, which iteratively refines the sparse coefficients $x$ to fit the dictionary model –.


*   **Parameter Mapping**: The sparse coefficients $x$ are normalized and mapped to microstructural parameters (e.g., $v_{iso}$, $v_{ic}$, $\kappa$) using linear combinations defined by the model equations –.


### **Adaptation for dMRI Models**



*   **IVIM Model**: For IVIM, the IHT decoder linearizes the bi-exponential model by separating diffusion and perfusion components. The dictionary $\Phi$ is designed to represent the bi-exponential signal, and the coefficients $x$ are normalized to ensure non-negativity and sum-to-one constraints –.


*   **NODDI Model**: For NODDI, the decoder linearizes the multi-compartment signal model, with the dictionary $\Phi$ capturing intracellular, extracellular, and CSF components. The IHT iterations enforce sparsity in the orientation dispersion and volume fraction coefficients –.


### **Training and Optimization**



*   The entire METSC framework (Transformer encoder + IHT decoder) is trained end-to-end using the Adam optimizer. The loss function is the mean squared error (MSE) between predicted and ground-truth microstructural parameters .


*   The IHT parameters (e.g., threshold $\lambda$) are learned during training, allowing the network to adapt to the specific characteristics of dMRI data .


### **Key Advantages of IHT in METSC**



*   **Sparsity Induction**: IHT enforces sparse representations, which aligns with the physical sparsity of dMRI signals (e.g., few dominant microstructural components) .


*   **Model Bias Integration**: By unfolding the IHT process into the network, METSC incorporates biophysical model knowledge (e.g., IVIM and NODDI equations) as inductive bias, reducing the need for large training datasets .


*   **Acceleration**: The IHT-based decoder enables parameter estimation with downsampled q-space data, achieving up to 11.25× acceleration for NODDI fitting .


### **Summary**

The IHT method is realized in the METSC Transformer through a 
model-driven decoder that unrolls iterative sparse reconstruction 
steps into learnable network layers. This integration allows the framework 
to leverage both the representational power of Transformer and the physical 
constraints of dMRI models, resulting in accurate and efficient microstructural parameter estimation.