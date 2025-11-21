# Weeks 6 & 7: Transfer Learning and the Deep SNN Challenge

## 🧠 The Strategy: Standing on the Shoulders of Giants
In previous weeks, we pushed custom lightweight models to their limit (~0.61 MCC). To break through this ceiling, I turned to **Transfer Learning**.

Deep Learning models trained on massive datasets (like ImageNet) have already learned how to detect edges, textures, and shapes. My hypothesis was that adapting a pre-trained **VGG16** backbone would significantly boost our accuracy compared to training from scratch.

However, this introduced a massive scientific question: **Can we take a standard pre-trained CNN and successfully convert it into a Spiking Neural Network?**



[Image of VGG16 architecture]


## 🛠️ Implementation & Experiments

### 1. The Control Group: VGG16 CNN (`vgg16_cnn.ipynb`)
First, I needed to establish the "upper bound" of performance. I implemented a standard U-Net using a **VGG16** encoder pre-trained on ImageNet.
* **Input Adaptation:** Since ImageNet weights expect 3 channels (RGB) and we have 5, I modified the first convolutional layer by averaging the pre-trained weights to accommodate the extra spectral bands.
* **Results (`CNN_1.png` & `CNN_history.png`):** The model performed exceptionally well, achieving high MCC scores (~0.69+) and producing crisp segmentation masks. This proved that the VGG16 architecture *can* solve the problem.

### 2. The SNN Conversion (`vgg16_snn.ipynb`)
Next, I attempted a direct conversion using `snnTorch`.
* **Mechanism:** I replaced every **ReLU** activation in the VGG16 backbone with **Leaky Integrate-and-Fire (LIF)** neurons.
* **The Challenge:** Deep SNNs suffer from the **"Vanishing Spike"** problem. Unlike continuous values in a CNN, binary spikes can "die out" as they travel through 16 deep layers.
* **Observation (`snn_1.png`):** The performance dropped significantly compared to the CNN. The deep network struggled to propagate the gradient through time.

### 3. Surgical Fine-Tuning (`vgg16_snn_finetuned.ipynb`)
To rescue the SNN performance, I implemented a specialized training regimen:
* **Gradual Unfreezing:** I froze the pre-trained VGG encoder weights for the first 3 epochs. This allowed the randomly initialized Decoder to stabilize and "learn how to interpret spikes" before we started adjusting the heavy backbone.
* **Hyperparameter Tuning:** I adjusted the neuron decay rate (`beta`) and firing thresholds to encourage signal propagation.
* **Result (`snn_finetuned.png`):** While this improved stability compared to the raw conversion, it highlighted a fundamental limitation: deep networks *without* Residual connections (like VGG) are very difficult to train in the spiking domain.

## 📊 Conclusion
This two-week experiment was crucial for understanding SNN dynamics:
1.  **CNNs** benefit massively from depth (VGG16) and pre-training.
2.  **SNNs** struggle with depth unless architectural highways (like Residual connections) are present.
3.  While the VGG SNN was efficient, the accuracy trade-off was too high, pointing us toward **ResNet** architectures for future improvements.

---
## 👤 Author
**Biswajit Nahak** *B.Tech ETC @IIIT Bhubaneswar* [GitHub Profile](https://github.com/Biswajitnahak2003) | [LinkedIn](https://www.linkedin.com/in/biswajit-nahak/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).