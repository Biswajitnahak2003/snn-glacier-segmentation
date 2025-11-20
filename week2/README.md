# Week 2: Data Augmentation and Pipeline Integration

## 1. Theoretical Background
### The Need for Augmentation
Neural networks are prone to **overfitting**, where they memorize training examples rather than learning generalizable features. This is especially true in remote sensing where labeled data is scarce. Data Augmentation artificially expands the dataset size by applying transformations.

### Invariance in Satellite Imagery
Unlike natural photography (where the sky is always up), satellite imagery is **Rotation Invariant**. A glacier looks like a glacier regardless of orientation. This allows us to use aggressive geometric augmentations that would be invalid in other domains.
* **Geometric Transforms:** Flips and rotations teach the model spatial invariance.
* **Elastic Transforms:** Simulates the natural, fluid distortions of ice flow, forcing the model to learn shape boundaries rather than rigid patterns.

## 2. Implementation Details
### Integration with Albumentations
We integrated the `Albumentations` library, known for its speed and support for multispectral data.
1.  **Pipeline Construction:** We defined a composition of transforms:
    * `HorizontalFlip` & `VerticalFlip`: $(p=0.5)$
    * `RandomRotate90`: $(p=0.5)$
    * `GridDistortion` & `ElasticTransform`: To simulate terrain variations.
2.  **Simultaneous Transformation:** A critical implementation detail was ensuring the **Input Image** and the **Segmentation Mask** undergo the *exact same* random transformation. If the image rotates $90^{\circ}$ but the mask does not, the labels become incorrect.
3.  **Tensor Conversion:** The pipeline handles the final conversion from `HWC` (Height-Width-Channel) NumPy arrays to `CHW` (Channel-Height-Width) PyTorch tensors.

---
### License & Author
**Author:** Biswajit Nahak  
**Qualification:** B.Tech ETC @ IIIT BBSR  
**License:** MIT License