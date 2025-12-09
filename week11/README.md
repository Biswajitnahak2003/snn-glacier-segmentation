# Week 12: The DeepLabV3+ Experiment (Contextual Architectures)

## 🧠 The Hypothesis
After finding success with U-Net architectures (Week 11), we attempted to implement **DeepLabV3+**, the industry standard for semantic segmentation.
* **The Theory:** DeepLab uses **ASPP (Atrous Spatial Pyramid Pooling)** to capture context at multiple scales simultaneously. We hypothesized that this would help the model distinguish between "Ice" and "Snow" better than a standard U-Net.

## 🛠️ Implementation
We built a custom **Spiking ASPP** module and tested it with two powerful backbones:
1.  **ResNet50:** To test a deep, residual backbone.
2.  **EfficientNet-B2:** To test a parameter-efficient backbone with SE blocks removed.

## 📊 Results: The "Signal Dilution" Problem

| Backbone | Model Type | MCC Score | Verdict |
| :--- | :--- | :--- | :--- |
| **ResNet50** | CNN | 0.630 | Good, but lower than VGG16 (0.69). |
| **ResNet50** | **SNN** | **0.326** | **Failure.** Significant drop from U-Net SNN (0.47). |
| **EffNet-B2** | CNN | 0.642 | Strong performance. |
| **EffNet-B2** | **SNN** | **0.343** | **Failure.** Poor signal propagation. |

## 📉 Failure Analysis
Our comparative analysis reveals why DeepLab is unsuitable for SNNs trained with surrogate gradients:
1.  **Parallel Branching (The Killer):** The ASPP module splits the incoming feature map into 5 parallel branches (1x1 conv + 3 dilated convs + pooling).
2.  **Energy Evaporation:** In a Spiking Network, the signal is a sparse train of binary spikes. Splitting this already weak signal into 5 paths reduces the energy in each branch below the firing threshold. The signal effectively "evaporates" inside the ASPP module.

## 🏆 Final Verdict
For Neuromorphic Earth Observation, **U-Net is the superior architecture.** The sequential structure of U-Net concentrates the spiking signal, whereas the parallel structure of DeepLab dilutes it.

---
### 👤 Author
**Biswajit Nahak** *B.Tech ETC Student at IIIT BBSR*

### 📄 License
This project is open-source and available under the [MIT License](../LICENSE).