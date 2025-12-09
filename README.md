# Glacier Segmentation: A Neuromorphic vs. Traditional Deep Learning Study

## 🌍 Project Overview
Glaciers are critical indicators of climate change, but monitoring them requires precise segmentation of satellite imagery. This project utilizes **5-Channel Multi-Spectral Satellite Imagery** to perform 4-class semantic segmentation (Glacier, Debris, Lake, Non-glacier).

The core objective of this project is to conduct a comparative study between two distinct paradigms of Deep Learning:
1.  **Convolutional Neural Networks (CNNs):** The industry standard for high-precision image segmentation.
2.  **Spiking Neural Networks (SNNs):** A bio-inspired approach where neurons communicate via binary "spikes," promising massive energy efficiency.

## 📂 Project Structure
This repository tracks 12 weeks of rigorous experimentation.

| Module | Focus Area | Key Outcome |
| :--- | :--- | :--- |
| **[Week 1-3](week1/)** | **Foundations** | Built custom `GlacierNet` (0.61 MCC). Established 5-channel pipeline. |
| **[Week 4-5](week4&5/)** | **SNN Basics** | Proved theoretical energy efficiency (14x) using `snnTorch` but struggled with accuracy. |
| **[Week 6-7](week6&7/)** | **Transfer Learning** | **VGG16 (CNN)** set the all-time accuracy record (**0.69 MCC**). |
| **[Week 8-9](week8&9/)** | **ResNet SNNs** | Discovered **ResNet50** as the best ResNet SNN (0.47 MCC), contradicting the "shallower is better" hypothesis. |
| **[Week 10](week10/)** | **EfficientNet** | Proved that **EfficientNet-V2** works (0.52 MCC) only if Attention blocks are removed. |
| **[Week 11-12](week11&12/)** | **DeepLabV3+** | Discovered that parallel branching (ASPP) dilutes spike signals, causing failure (0.34 MCC). |

## 🔬 Methodology

### 1. The Data
* **Source:** The dataset utilized in this study was obtained from the **GlacierHack 2025** challenge, organized by the **IEEE InGARSS 2025 Young Professionals (YP) Team**.
* **Input:** 5-Channel TIFs (R, G, B, IR, UV).
* **Output:** 4-Class Segmentation Masks.
* **Preprocessing:** I implemented percentile clipping and min-max scaling to handle the high dynamic range of satellite sensors.

### 2. The CNN Approach (Baseline)
I established a high-performance baseline using U-Net architectures.
* **Custom Models:** Built lightweight U-Nets from scratch to understand feature extraction.
* **Transfer Learning:** Utilized VGG16 encoders pre-trained on ImageNet to leverage learned features, achieving high segmentation accuracy (~0.69 MCC).

### 3. The SNN Approach (Experimental)
I converted our architectures into the spiking domain using `snnTorch`.
* **Temporal Coding:** Input data is repeated over time steps ($T$), allowing neurons to integrate signals and fire binary spikes.
* **Surrogate Gradients:** I utilized surrogate functions (like `ATan`) to enable backpropagation through non-differentiable spikes.
* **Optimization:** We tackled challenges like "Vanishing Spikes" in deep networks by implementing residual connections and fine-tuning neuron decay rates.

## 🏆 Final Results

### 1. Accuracy Champions (MCC)
* 🥇 **CNN - VGG16:** **0.6926** (Best Detail Preservation)
* 🥈 **CNN - EfficientNet-V2:** 0.6692 (Best Parameter Efficiency)
* 🥉 **SNN - EfficientNet-V2:** **0.5250** (Best Spiking Performance)

### 2. Efficiency Champions (Theoretical Energy)
* 🥇 **SNN - EfficientNet-V2:** ~25x efficiency gain vs VGG16 CNN.
* 🥈 **SNN - ResNet50:** ~12x efficiency gain vs ResNet50 CNN.

### 3. Full Comparison Table

| Architecture | CNN Accuracy (MCC) | SNN Accuracy (MCC) | Verdict |
| :--- | :--- | :--- | :--- |
| **Custom U-Net** | 0.6100 | 0.5300 | Good baseline |
| **VGG16** | **0.6926** | 0.4462 | Failed due to lack of residuals |
| **ResNet18** | 0.5500 | 0.3500 | Underfitting |
| **ResNet34** | 0.6213 | 0.3800 | Good balance |
| **ResNet50** | 0.6686 | 0.4731 | Strong CNN, decent SNN |
| **EfficientNet-V2** | 0.6692 | **0.5250** | **Best Modern SNN** (SE Removed) |
| **DeepLabV3+** | 0.6424 | 0.3429 | Failed due to signal dilution |

## 🔬 Scientific Conclusions
1.  **Architecture Matters:** For SNNs, **"Simple is Better."** Complex branching (DeepLab) or Attention mechanisms (EfficientNet-B0) kill the binary signal. Simple, dense residual blocks (EffNet-V2) work best.
2.  **Resolution is King:** VGG16 remains the CNN king because it preserves spatial resolution in the first layer, whereas ResNet/EfficientNet aggressively downsample the input, losing fine glacier cracks that cannot be recovered.
3.  **Viability:** The SNN models demonstrated valid segmentation capabilities (~0.52 MCC) with **>90% energy savings**, making them viable for deployment on power-constrained edge devices like satellites and drones.

## 🛠️ Tech Stack
* **Core:** `Python`, `PyTorch`
* **Neuromorphic:** `snnTorch`
* **Vision:** `OpenCV`, `Albumentations`, `Segmentation Models PyTorch`
* **Analysis:** `Matplotlib`, `Scikit-Learn`

## 🙏 Acknowledgements
I would like to sincerely thank the **IEEE InGARSS 2025 Young Professionals Team** for providing the **GlacierHack 2025 dataset** and the opportunity to use it for academic research.
* *Reference: IEEE InGARSS GlacierHack 2025.*

---
## 👤 Author
**Biswajit Nahak** | B.Tech ETC | @IIIT Bhubaneswar
* [GitHub Profile](https://github.com/Biswajitnahak2003)
* [LinkedIn](https://www.linkedin.com/in/biswajit-nahak/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).