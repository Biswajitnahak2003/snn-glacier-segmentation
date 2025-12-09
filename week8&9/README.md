# Weeks 8 & 9: The Residual Network Experiments (ResNet)

## 🧠 The Hypothesis: Solving "Vanishing Spikes"
After the experiments with VGG16 in Weeks 6-7, I hit a major roadblock. While the VGG-CNN was excellent (0.69 MCC), the VGG-SNN performed poorly (0.44 MCC).

My analysis pointed to the **"Vanishing Spike" Problem**. In a deep Spiking Neural Network (SNN), binary signals degrade as they pass through many layers. Without a path to preserve the signal, the deeper layers effectively stop firing.

**The Solution:** I turned to **ResNet (Residual Networks)**.

The "Skip Connections" in ResNet act as highways for spikes, theoretically allowing the signal to bypass dead neurons and propagate deeper into the network.

## 🛠️ Implementation Details
I integrated `torchvision`'s ResNet backbones into my Spiking U-Net framework:
1.  **Stem Modification:** ResNet's default input layer (7x7 Conv, Stride 2) creates aggressive downsampling. I modified this to handle the 5-channel satellite input.
2.  **Spiking ResBlocks:** I replaced standard ReLU activations with `Leaky Integrate-and-Fire (LIF)` neurons.
3.  **Tuning:** For the deeper models (34 & 50), I used a steeper surrogate gradient slope ($k=25$) and `OneCycleLR` scheduler to force signal propagation.

## 📊 Results & Analysis

I tested three depths to find the optimal balance for SNNs.

| Architecture | CNN Baseline (MCC) | SNN Experimental (MCC) | Verdict |
| :--- | :--- | :--- | :--- |
| **ResNet18** | 0.55 | 0.35 | **Underfitting.** Too shallow to capture complex glacier features in the spiking domain. |
| **ResNet34** | 0.62 | **0.50** | **🏆 The "Sweet Spot".** The best balance of depth and trainability. |
| **ResNet50** | **0.67** | 0.47 | **Diminishing Returns.** The bottleneck structure helped the CNN, but the SNN struggled with the increased depth (50 layers). |

## 📉 Scientific Conclusion
This study yielded a critical finding: **Depth helps, but only up to a point.**
1.  **The "Goldilocks" Zone:** Unlike VGG (0.44) or ResNet18 (0.35), **ResNet34 (0.50)** provided enough capacity to learn without being so deep that the spikes vanished.
2.  **CNN vs. SNN:** The ResNet50 CNN (0.67) nearly matched the VGG16 baseline, proving that deep residual networks are excellent for this task *if* continuous activations are used. SNNs, however, struggle to utilize that depth.

---
### 👤 Author
**Biswajit Nahak** *B.Tech ETC Student at IIIT BBSR*

### 📄 License
This project is open-source and available under the [MIT License](../LICENSE).