# Week 3: Custom Architecture and Full Training Pipeline

## 1. Theoretical Background
### The U-Net Architecture
For semantic segmentation, we require an output resolution equal to the input resolution. We implemented the **U-Net** architecture from scratch, which consists of:
* **Contracting Path (Encoder):** Repeated application of convolutions and pooling to capture context ("What is present?").
* **Expanding Path (Decoder):** Upsampling layers to restore spatial dimensions ("Where is it present?").
* **Skip Connections:** The defining feature of U-Net. High-resolution features from the encoder are concatenated with the decoder, allowing the model to localize fine details (like narrow ice tongues) that would otherwise be lost.

### Training Dynamics
Training a neural network involves minimizing a loss function via Backpropagation.
* **Loss Function:** We use **CrossEntropyLoss**, which calculates the divergence between the predicted class probabilities and the actual pixel labels.
* **Optimization:** We utilize the **AdamW** optimizer, which decouples weight decay from the gradient update, offering better generalization than standard SGD.

## 2. Implementation Details
### Custom `GlacierNet` Architecture
Instead of using a pre-trained black box, we built a modular CNN:
1.  **`DoubleConv` Block:** A reusable module containing `Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU`.
2.  **Bottleneck:** A bridge connecting the encoder and decoder with the highest feature depth (512 channels).
3.  **Input Adaptation:** The first layer was explicitly designed to accept **5 input channels**, avoiding the need for makeshift adapters.

### The Training Loop
We constructed a robust training loop (`train_one_epoch` and `validate`) that:
1.  Iterates through the `DataLoader`.
2.  Moves tensors to the GPU (CUDA).
3.  Performs the Forward Pass and calculates Loss.
4.  Performs Backpropagation and Weight Updates.
5.  Calculates the **Matthews Correlation Coefficient (MCC)** during validation to monitor true model performance, as accuracy can be misleading in imbalanced datasets.

---
### License & Author
**Author:** Biswajit Nahak  
**Qualification:** B.Tech ETC @ IIIT BBSR  
**License:** MIT License