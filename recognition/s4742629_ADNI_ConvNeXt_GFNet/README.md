# Introduction

The diagnosis of brain illnesses is conventionally carried out through invasive biopsy procedures or manual examination of medical images. However, manual interpretation can introduce subjectivity and human error, potentially leading to inaccurate diagnoses.

This project investigates the viability of ConvNeXt, a state-of-the-art convolutional neural network (CNN) image classifier, for identifying Alzheimer’s disease (AD) from medical imaging data. The model was trained and evaluated using the ADNI dataset, which contains two-dimensional MRI brain scans of both normal cognitive (NC) and Alzheimer’s disease (AD) patients.

# ConvNeXt

ConvNeXt is a modern CNN architecture derived from ResNet, designed to compete with Vision Transformers (ViTs) by adopting several Transformer-inspired design principles while retaining a purely convolutional structure. Its main advantages over ViTs are lower data requirements and reduced computational complexity, making it well-suited for medical imaging tasks where datasets are often limited. These properties allow ConvNeXt to achieve strong performance while minimizing overfitting. Furthermore, through transfer learning (using pretrained weights from large-scale image datasets) ConvNeXt models have demonstrated promising accuracy in classifying MRI brain scans and detecting early signs of Alzheimer’s disease.

## ConvNeXt Architecture

The model begins with a stem composed of a `4×4` convolutional layer with stride `4`. This reduces the input dimensions by a factor of four while increasing the channel depth to `96`, yielding a higher-dimensional feature space. This aggressive reduction effectively *patchifies* the image, analogous to the patch embeddings used in Vision Transformers. Each `4×4` receptive field acts as an individual patch, similar to the `16×16` patch segmentation used by many ViT models.

Following the stem are four hierarchical stages of ConvNeXt blocks. Each stage progressively reduces spatial resolution while increasing the number of channels. Between stages, a `2×2` convolutional downsampling layer with stride `2` halves the spatial dimensions and doubles the number of channels. These layers form a feature pyramid that expands the model’s receptive field, enabling a transition from local to global feature learning.  

After the final stage, a global average pooling layer condenses each feature map to a single value per channel. Layer normalization is applied before the fully connected classification head, which outputs logits for binary classification. Optional dropout may be applied to the head to mitigate overfitting.

Layer normalization replaces batch normalization, and the GELU activation function replaces ReLU, producing smoother nonlinear responses and more stable optimization.

### Stage Summary

- **Patchify Stem:** `4×4` convolution, stride `4` = `(96 × H/4 × W/4)`
- **Downsample 1:** `2×2` convolution, stride `2` = `(192 × H/8 × W/8)`
- **Downsample 2:** `2×2` convolution, stride `2` = `(384 × H/16 × W/16)`
- **Downsample 3:** `2×2` convolution, stride `2` = `(768 × H/32 × W/32)`

## ConvNeXt Block

Each ConvNeXt block begins with a `7×7` depthwise convolution, providing a large receptive field to capture broad spatial context. A LayerNorm operation follows to stabilize gradients and maintain numerical consistency. The block then employs an inverted bottleneck structure: two `1×1` pointwise convolutions with a GELU activation in-between. The first `1×1` convolution expands the channel dimension by a factor of four, and the second reduces it back to the original size, enabling efficient nonlinear channel mixing while maintaining computational efficiency.

Each block also includes a residual connection that adds the input to the output, facilitating gradient flow and improving convergence. DropPath regularization randomly drops entire residual branches during training, further enhancing generalization. Some ConvNeXt implementations also introduce LayerScale, a set of small learnable scalars applied to residual outputs to stabilize deep training.

### Block Structure

- `7×7` depthwise convolution  
- Layer normalization  
- `1×1` expansion convolution (increased by a factor of 4)  
- GELU activation  
- `1×1` contraction convolution (decreased by a factor of 4)  
- Residual connection with DropPath regularization

## ConvNeXt Tiny

ConvNeXt-Tiny is the smallest architecture variant of the ConvNeXt family introduced by Meta. Compared to larger models such as ConvNeXt-Small and ConvNeXt-Large, it employs three ConvNeXt blocks in the first, second, and fourth stages, and nine blocks in the third stage, resulting in an overall structure of `[3, 3, 9, 3]`. Consequently, ConvNeXt-Tiny contains the fewest trainable parameters among the variants and was selected for this application due to the limited size of the available dataset.

# ADNI Dataset

The ADNI dataset contains two-dimensional grayscale MRI brain scan images divided into two categories: normal cognitive (NC) and Alzheimer’s disease (AD) patients. Each image has dimensions of 256 by 240 pixels. Data were collected from 1,051 patients, with 20 distinct scans available per patient. The dataset is provided in two prepared subsets: a training set containing 21,520 samples and a testing set containing 9,000 samples. The class distribution between AD and NC is roughly equal in both subsets, but there is some discrepancy. The dataset is structured within its folder as follows:

```
AD_NC  
├── test  
│   ├── AD  
│   └── NC  
└── train  
    ├── AD  
    └── NC  
```

## Data Split

20% of the pre-prepared training set was set aside as a validation set. The validation set was used for hyperparameter tuning, while the remaining training data were used to train the model’s internal parameters.
Although it is theoretically possible to use the pre-prepared testing set as a validation set, this would reduce the amount of unseen data available for final evaluation. Given the limited availability of medical imaging data and the high stakes associated with diagnostic accuracy, it was essential to preserve a distinct testing set to better represent real-world variability in medical scans.

| Dataset                                | AD Samples | NC Samples | Total  |
|----------------------------------------|-------------|-------------|--------|
| **Training Set**                       | 10,400      | 11,120      | 21,520 |
| **Testing Set**                        | 4,460       | 4,540       | 9,000  |
| **Split Training Set (after split)**   | 8,320       | 8,920       | 17,240 |
| **Validation Set**                     | 2,080       | 2,200       | 4,280  |

To avoid data leakage, the validation set was drawn from the training data rather than from the test set. The initial split of the training data was stratified only by class to maintain similar class proportions across subsets. However, early experiments revealed inflated validation scores due to patient overlap between the training and validation sets. To address this, the dataset was re-split by patient, ensuring that scans from the same individual appeared in only one subset.

## Data Augmentation

Two categories of image transformations were applied during preprocessing.
The first consisted of deterministic transforms, applied consistently to all datasets. These included resizing and padding images to 256×256 pixels and normalizing pixel intensity using a mean of 0.263 and a standard deviation of 0.271, values derived from related studies. Images were also converted to single-channel grayscale within the custom data loader, as the MRI scans were predominantly monochromatic.

```python
train_tf = transforms.Compose([
        transforms.Pad((8, 0, 8, 0)),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.01, 0.05)),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
```

The second category consisted of random augmentations applied only to the training set to increase data diversity and reduce overfitting. These included:
- Random resized crops, horizontal flips, rotations, and affine transformations to simulate variations in orientation and scale found in real medical imaging.
- Random erasing and resized crops to obscure regions of the image, encouraging the model to learn generalized features rather than memorizing specific patterns.
- Gaussian blur and color jitter to mimic variations in brightness and image fidelity, improving robustness to noise and imaging inconsistencies.
These augmentations aimed to enhance regularization and improve the model’s generalizability to unseen medical data.

```python
eval_tf = transforms.Compose([
        transforms.Pad((8, 0, 8, 0)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
```
