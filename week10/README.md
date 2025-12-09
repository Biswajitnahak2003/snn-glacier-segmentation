# Week 10: The EfficientNet Experiment (Modern Architectures)

## 🧠 The Hypothesis
After identifying the limits of ResNet (0.50 MCC), I attempted to leverage state-of-the-art "Modern" CNN architectures. I chose **EfficientNet**, which uses "Compound Scaling" to balance depth, width, and resolution.

**The Goal:** Can an architecture optimized for parameter efficiency (fewer weights) translate to spiking efficiency (fewer events)?

## 🛠️ Experiment 1: EfficientNet-B0 (The Failure)
I started with the standard EfficientNet-B0 backbone.
* **Result:** The SNN collapsed (**0.16 MCC**).
* **Analysis:** The model failed to learn beyond simple background statistics. I identified two culprits:
    1.  **Depthwise Convolutions:** Sparse connectivity made spike propagation difficult.
    2.  **SE Blocks (Attention):** The "Squeeze-and-Excitation" attention mechanism multiplies channels by a scale factor. In an SNN, this turned into a "Hard Gate," randomly muting entire channels when the attention neuron failed to fire.

## 🛠️ Experiment 2: EfficientNet-V2 "Surgical" (The Success)
To fix this, I pivoted to **EfficientNet-V2**.
1.  **Architecture Change:** V2 uses **Fused-MBConv** (Dense Convolutions) in early layers, which are friendlier to SNNs.
2.  **Architectural Surgery:** I implemented a programmatic "Kill Switch" to remove all SE/Attention blocks, replacing them with Identity layers.

### 📊 Final Results

| Model | CNN Score (MCC) | SNN Score (MCC) | Analysis |
| :--- | :--- | :--- | :--- |
| **EffNet-B0** | 0.67 | 0.16 | Failed due to Attention mechanisms. |
| **EffNet-V2** | **0.67** | **0.525** | **Success!** Surpassed ResNet34 (0.50) and nearly matched the best CNNs. |

## 📉 Scientific Conclusion
This week proved that **Modern CNNs CAN be converted to SNNs**, but they require modification.
* **Attention is dangerous:** Analog attention mechanisms (SE, Transformers) often break binary SNNs unless replaced or heavily tuned.
* **Density matters:** Dense convolutions (V2) propagate spikes far better than Depthwise convolutions (B0).
* **Efficiency:** The **EfficientNet-V2 SNN (0.525)** is now our most efficient high-performing model, beating ResNet34 in accuracy while using fewer parameters.

---
### 👤 Author
**Biswajit Nahak** *B.Tech ETC Student at IIIT BBSR*

### 📄 License
This project is open-source and available under the [MIT License](../LICENSE).