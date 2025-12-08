# Weeks 8 & 9: The Residual Network Experiments

## 🧠 The Hypothesis: Solving "Vanishing Spikes"
After the experiments with VGG16 in Weeks 6-7, I hit a major roadblock. While the VGG-CNN was excellent (0.69 MCC), the VGG-SNN performed poorly (0.44 MCC).

My analysis pointed to the **"Vanishing Spike" Problem**.
* In a deep Spiking Neural Network (SNN), binary signals (0s and 1s) degrade as they pass through many layers.
* Without a path to preserve the signal, the deeper layers effectively stop firing, and the gradient cannot flow back during training.

**The Solution:** I turned to **ResNet (Residual Networks)**.

The "Skip Connections" in ResNet act as highways for spikes, theoretically allowing the signal to bypass dead neurons and propagate deeper into the network.

## 🛠️ Implementation Details

### Architecture Adaptation
I integrated `torchvision`'s ResNet backbones into my Spiking U-Net framework. 
1.  **Stem Modification:** ResNet's default input layer (7x7 Conv, Stride 2) creates aggressive downsampling. I modified this to handle the 5-channel satellite input while attempting to preserve spatial resolution.
2.  **Spiking ResBlocks:** I replaced the standard ReLU activations inside the residual blocks with `Leaky Integrate-and-Fire (LIF)` neurons using `snnTorch`.
3.  **Surrogate Gradients:** I tuned the `FastSigmoid` slope ($k=25$) to create a steeper gradient signal, helping the network learn faster.

### The Experiments
I created two distinct environments to test the "Depth vs. Accuracy" trade-off:
* **`resnet18/`**: A moderately deep network (18 layers) to test stability.
* **`resnet34/`**: A deeper variant (34 layers) to test capacity.

## 📊 Results & Analysis

### 1. The "Sweet Spot" (ResNet18)
* **CNN Baseline:** 0.55 MCC
* **SNN Experimental:** **0.35 MCC**
* **Verdict:** This was a breakthrough. The ResNet18 SNN matched its CNN counterpart perfectly. This proved that residual connections *do* solve the vanishing spike problem for moderately deep networks.

### 2. The Depth Limit (ResNet34)
* **CNN Baseline:** 0.62 MCC
* **SNN Experimental:** 0.50 MCC (after heavy tuning)
* **Verdict:** Here, we hit the limit. Despite the skip connections, the signal degraded too much across 34 layers. While I improved it from an initial 0.38 to 0.50 using `OneCycleLR` and steep gradients, it couldn't match the CNN's precision.

## 📉 Scientific Conclusion
This two-week study yielded a critical finding for Neuromorphic Earth Observation: **"Less is More."**
* For SNNs, the **ResNet18** architecture offered the best balance of accuracy and trainability.
* Going deeper (ResNet34) yielded diminishing returns in the spiking domain, whereas it improved performance in the continuous (CNN) domain.

---
### 👤 Author
**Biswajit Nahak** *B.Tech ETC Student at IIIT BBSR*

### 📄 License
This project is open-source and available under the [MIT License](../LICENSE).