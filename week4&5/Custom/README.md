# Week 4: Spiking Neural Networks & Energy Efficiency

## 🧠 The Paradigm Shift: Why SNNs?
After establishing a strong baseline with the Custom CNN in Week 3, I pivoted to the experimental core of this project: **Neuromorphic Computing**.

While CNNs are accurate, they are computationally expensive. Every neuron computes a value for every pixel, regardless of whether that pixel contains useful information. **Spiking Neural Networks (SNNs)** mimic the biological brain. Neurons only transmit information (spikes) when a threshold is exceeded. This event-driven processing promises massive energy savings.



## 🛠️ Implementation Journey

I utilized the `snnTorch` library to convert my U-Net architecture into the spiking domain. This wasn't a plug-and-play process; it required several iterations to get right.

### 1. The Baseline SNN (`snn.ipynb`)
I started by replacing the standard activations (ReLU) in my U-Net with **Leaky Integrate-and-Fire (LIF)** neurons.
* **Mechanism:** Instead of a single forward pass, the SNN processes the image over $T$ time steps (Temporal Domain).
* **Challenge:** I immediately hit the "Memory Wall." Because the SNN stores the membrane potential history for every time step, the VRAM usage exploded, causing Out-Of-Memory (OOM) crashes.

### 2. Optimization: SNN "Lite" (`snn_1.ipynb`)
To solve the memory constraints, I re-engineered the architecture into a "Lite" version.
* **Reduction:** I reduced the base filter count from 32 to 16.
* **Result:** This allowed me to increase the Time Steps ($T$) without crashing the GPU. More time steps generally lead to better accuracy in SNNs as the signal has more time to propagate.

### 3. Solving Class Collapse (`snn_2.ipynb`)
Visualizing the initial results (`SNN_Lite_sample.png`) revealed a critical issue: the model was predicting "Background" and "Ice" well but completely ignoring "Snow" (the Green class).
* **Diagnosis:** SNNs naturally gravitate towards **sparsity** (silence). Since Snow was a minority class, the network decided it was more efficient to just never spike for it.
* **Solution:** I implemented a **Weighted CrossEntropy Loss**. I assigned a penalty multiplier of **3.0x** to the Snow class, forcing the spiking neurons to react to it. The results (`SNN_Weighted_sample.png`) showed the Green class reappearing.

## ⚡ The Showdown: CNN vs. SNN

The final stage (`comparison.ipynb`) was a head-to-head battle between the Week 3 CNN and the Week 4 SNN.

### Methodology
I didn't just compare accuracy; I calculated the **Theoretical Energy Cost**.
* **CNN Cost:** Calculated based on Multiply-Accumulate (MAC) operations (approx. 4.6pJ per op).
* **SNN Cost:** Calculated based on Accumulate (AC) operations, scaled by the **Sparsity Rate** (how often neurons actually fire).

### 🏆 Results (`comparison.png`)
* **Accuracy:** The CNN is still the king of precision (~0.61 MCC vs 0.53 MCC). The SNN struggles slightly with fine boundaries due to the binary nature of spikes.
* **Efficiency:** The SNN achieved a **sparsity of ~35%**, meaning 65% of the network remained silent during inference.
* **Conclusion:** While we traded off some accuracy, the SNN demonstrated a **~14x improvement in energy efficiency**. This proves viability for deployment on edge devices (satellites/drones) where battery life is more critical than perfect pixel precision.

---
## 👤 Author
**Biswajit Nahak** *B.Tech ETC @IIIT Bhubaneswar* [GitHub Profile](https://github.com/Biswajitnahak2003) | [LinkedIn](https://www.linkedin.com/in/biswajit-nahak/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).