# Week 2: Data Augmentation and Pipeline Integration

## 🚀 The Challenge: Fighting Overfitting
Deep learning models are data-hungry, and in remote sensing, high-quality labeled data is scarce. I noticed early on that the model was at risk of **overfitting**—essentially memorizing the training images rather than learning what a glacier actually looks like.

To solve this, I implemented a robust **Data Augmentation** pipeline to artificially expand my dataset and force the model to learn generalized features.

## 🧠 My Strategy: "Space Has No Up"
I realized that satellite imagery has a unique advantage over standard photography: **Rotation Invariance**.
* In a photo of a cat, the sky is usually up.
* In a satellite image of a glacier, orientation doesn't matter. A glacier is a glacier whether it's facing North, South, or diagonally.

I exploited this fact to use aggressive geometric augmentations that would be impossible in other domains.

## 🛠️ Implementation Details
I chose the `Albumentations` library for this task because it is significantly faster than standard tools and handles multi-channel spectral data natively.

### 1. The Augmentation Pipeline
I designed a composition of transforms to simulate various conditions:
* **Geometric Flips:** `HorizontalFlip` and `VerticalFlip` (applied 50% of the time) to teach the model spatial invariance.
* **Rotation:** `RandomRotate90` to ensure the model doesn't learn to expect glaciers in only one orientation.
* **Shape Simulation:** I used `GridDistortion` and `ElasticTransform`. This was a key decision—glaciers are fluid, organic shapes. These transforms simulate natural distortions in the ice flow, forcing the model to focus on boundaries rather than rigid shapes.

### 2. Solving the Alignment Problem
A critical technical challenge in segmentation is **Simultaneous Transformation**. If I rotate the input image by 90°, I *must* rotate the Ground Truth mask by exactly the same amount. If they get out of sync, the labels become garbage.
* *Solution:* I utilized Albumentations' dual-transform capability to ensure the image and mask are always treated as a single, locked pair during transformation.

### 3. Tensor Conversion
Finally, I integrated the format conversion into the pipeline. The script automatically converts the data from the standard NumPy `HWC` (Height-Width-Channel) format to the PyTorch-required `CHW` (Channel-Height-Width) format, streamlining the training loop.

---
## 👤 Author
**Biswajit Nahak** *B.Tech ETC @IIIT Bhubaneswar* [GitHub Profile](https://github.com/Biswajitnahak2003) | [LinkedIn](https://www.linkedin.com/in/biswajit-nahak/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).