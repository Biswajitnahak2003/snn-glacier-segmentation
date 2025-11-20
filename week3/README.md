# Week 3: Custom Architecture and Full Training Pipeline

## 🧠 The Strategy: Why a Custom U-Net?
This week, I moved from data preparation to the core of the project: the model itself. While transfer learning is popular, I decided to build a **Custom U-Net** from scratch.

Why? Because standard pre-trained models (like ResNet) are designed for 3-channel RGB images (dogs, cats, cars). My satellite data has **5 spectral channels**. Hacking a pre-trained model to accept 5 channels often breaks the pre-learned weights in the first layer. By building from scratch, I created a network native to the domain.



[Image of U-Net architecture]


### My Architecture Design (`GlacierNet`)
I implemented the classic **U-Net** architecture, which is the gold standard for segmentation. It works on a simple principle:
* **The Encoder (Contracting Path):** It aggressively downsamples the image to understand *what* is in it (context), sacrificing spatial resolution.
* **The Decoder (Expanding Path):** It upsamples the features back to the original size to understand *where* the objects are (localization).
* **Skip Connections:** This is the "secret sauce." I wired connections directly from the Encoder to the Decoder. This allows the model to recover fine details—like narrow ice tongues—that would otherwise be lost during downsampling.

## 🛠️ Implementation Details

### 1. Modular Components
I structured the code modularly to make it clean and reusable:
* **`DoubleConv` Block:** The building block of the network. It stacks `Conv2d -> BatchNorm -> ReLU` twice. This provides the non-linearity needed to learn complex features.
* **The Bottleneck:** I designed a bridge connecting the encoder and decoder with 512 feature channels, forcing the model to compress the image content into a rich, abstract representation.
* **Input Adaptation:** Unlike stock models, my first layer accepts **5 input channels**, allowing the model to learn directly from the Near-Infrared and Shortwave-Infrared bands.

### 2. The Training Engine
I constructed a robust training loop (`train_one_epoch` and `validate`) to handle the heavy lifting:
* **Hardware Acceleration:** The script automatically detects and moves tensors to the GPU (CUDA) for faster computation.
* **Optimization:** I chose the **AdamW** optimizer over standard SGD. It decouples weight decay from the gradient update, which generally leads to better generalization on unseen data.
* **Loss Function:** I used **CrossEntropyLoss** to penalize the model for every pixel it misclassified.

### 3. The "True" Metric: MCC
Accuracy is a liar in satellite imagery. If 90% of an image is "Background" and 10% is "Glacier," a model that predicts "Background" for everything is 90% accurate but useless.

To solve this, I implemented the **Matthews Correlation Coefficient (MCC)** tracking. MCC is a robust metric that only goes up if the model correctly predicts *both* the majority (background) and minority (glacier) classes.

---
## 👤 Author
**Biswajit Nahak** *B.Tech ETC @IIIT Bhubaneswar* [GitHub Profile](https://github.com/Biswajitnahak2003) | [LinkedIn](https://www.linkedin.com/in/biswajit-nahak/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).