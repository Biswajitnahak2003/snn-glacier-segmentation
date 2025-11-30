# Glacier Segmentation: A Neuromorphic vs. Traditional Deep Learning Study

## 🌍 Project Overview
Glaciers are critical indicators of climate change, but monitoring them requires precise segmentation of satellite imagery. This project utilizes **5-Channel Multi-Spectral Satellite Imagery** to perform 4-class semantic segmentation (Glacier, Debris, Lake, Non-glacier).

The core objective of this Project is to conduct a comparative study between two distinct paradigms of Deep Learning:
1.  **Convolutional Neural Networks (CNNs):** The industry standard for high-precision image segmentation.
2.  **Spiking Neural Networks (SNNs):** A bio-inspired approach where neurons communicate via binary "spikes," promising massive energy efficiency.

## 📂 Project Structure
This repository tracks the evolution of the project over several weeks of experimentation.

| Module | Focus Area | Key Technologies |
| :--- | :--- | :--- |
| **[Week 1](week1/)** | **Data Engineering** | Custom PyTorch Dataset, 5-Band TIF Loading, Robust Normalization |
| **[Week 2](week2/)** | **Augmentation Pipeline** | `Albumentations`, Geometric Transforms, Mask Alignment |
| **[Week 3](week3/)** | **Custom CNN Architecture** | Custom U-Net (`GlacierNet`), AdamW Optimizer, MCC Metric Tracking |
| **[Week 4 & 5](week4&5/)** | **Spiking Neural Networks** | `snnTorch`, Leaky Integrate-and-Fire (LIF), Energy Efficiency Analysis |
| **[Weeks 6 & 7](week6&7/)** | **Transfer Learning** | VGG16 Backbone, Pre-trained Weights, Deep SNN Conversion Challenges |

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

## 🏆 Results

### Accuracy vs. Efficiency
* **Precision:** The CNN models consistently outperformed SNNs in pure segmentation accuracy (MCC), excelling at defining fine boundaries.
* **Efficiency:** The SNN models demonstrated a **~14x theoretical improvement in energy efficiency**. By exploiting sparsity (only ~35% of neurons active), SNNs proved viable for edge deployment where power is limited, even if it comes with a trade-off in precision.

## 🛠️ Tech Stack
* **Core:** `Python`, `PyTorch`
* **Neuromorphic:** `snnTorch`
* **Vision:** `OpenCV`, `Albumentations`, `Segmentation Models PyTorch`
* **Analysis:** `Matplotlib`, `Scikit-Learn`

## 🙏 Acknowledgements
I would like to sincerely thank the **IEEE InGARSS 2025 Young Professionals Team** for providing the **GlacierHack 2025 dataset** and the opportunity to use it for academic research.
* *Reference:* IEEE InGARSS 2025 Young Professionals Team. (2025). *GlacierHack 2025 Dataset: Multi-class Glacier Segmentation*.

---
## 👤 Author
**Biswajit Nahak** | B.Tech ETC | @IIIT Bhubaneswar
* [GitHub Profile](https://github.com/Biswajitnahak2003)
* [LinkedIn](https://www.linkedin.com/in/biswajit-nahak/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
*Note: The dataset used in this project is the property of IEEE InGARSS/GlacierHack and is used here for educational/academic purposes.*