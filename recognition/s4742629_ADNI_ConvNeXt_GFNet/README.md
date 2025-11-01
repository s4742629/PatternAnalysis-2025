## ConvNeXt Architecture

The model begins with a **stem** composed of a `4×4` convolutional layer with stride `4`.  
This reduces the input dimensions by a factor of four while increasing the channel depth to `96`, yielding a higher-dimensional feature space.  
This aggressive reduction effectively *patchifies* the image, analogous to the patch embeddings used in Vision Transformers.  
Each `4×4` receptive field acts as an individual patch, similar to the `16×16` patch segmentation used by many ViT models.

Following the stem are **four hierarchical stages** of ConvNeXt blocks.  
Each stage progressively reduces spatial resolution while increasing the number of channels.  
Between stages, a `2×2` convolutional downsampling layer with stride `2` halves the spatial dimensions and doubles the number of channels.  
These layers form a feature pyramid that expands the model’s receptive field, enabling a transition from local to global feature learning.  

After the final stage, a **global average pooling** layer condenses each feature map to a single value per channel.  
**Layer normalization** is applied before the fully connected classification head, which outputs logits for binary classification.  
Optional **dropout** may be applied to the head to mitigate overfitting.

Layer normalization replaces batch normalization, and the **GELU** activation function replaces **ReLU**, producing smoother nonlinear responses and more stable optimization.

---

### Stage Summary

- **Patchify Stem:** `4×4` convolution, stride `4` → `(96 × H/4 × W/4)`
- **Downsample 1:** `2×2` convolution, stride `2` → `(192 × H/8 × W/8)`
- **Downsample 2:** `2×2` convolution, stride `2` → `(384 × H/16 × W/16)`
- **Downsample 3:** `2×2` convolution, stride `2` → `(768 × H/32 × W/32)`

---

## ConvNeXt Block

Each ConvNeXt block begins with a `7×7` **depthwise convolution**, providing a large receptive field to capture broad spatial context.  
A **LayerNorm** operation follows to stabilize gradients and maintain numerical consistency.  
The block then employs an **inverted bottleneck** structure — two `1×1` pointwise convolutions separated by a **GELU** activation.  
The first `1×1` convolution expands the channel dimension by a factor of four, and the second reduces it back to the original size, enabling efficient nonlinear channel mixing while maintaining computational efficiency.

Each block also includes a **residual connection** that adds the input to the output, facilitating gradient flow and improving convergence.  
**DropPath** regularization randomly drops entire residual branches during training, further enhancing generalization.  
Some ConvNeXt implementations also introduce **LayerScale**, a set of small learnable scalars applied to residual outputs to stabilize deep training.

---

### Block Structure

- `7×7` depthwise convolution  
- Layer normalization  
- `1×1` expansion convolution (×4 channels)  
- GELU activation  
- `1×1` contraction convolution (÷4 channels)  
- Residual connection with DropPath regularization
