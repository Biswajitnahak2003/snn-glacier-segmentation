# Week 1: Data Ingestion, Processing, and Visualization

## 🚀 Dealing with Satellite Imagery
This week was all about setting up the data pipeline. Unlike standard RGB photos, the satellite images for this project are **5-Channel Multispectral TIFFs**. Working with them wasn't straightforward because:
* **High Dynamic Range:** The sensors capture 16-bit floating-point data. If you try to open these in a standard image viewer, they look pitch black or washed out because they exceed the normal 0-255 color range.
* **Multi-Dimensionality:** Standard libraries like PIL struggle with 5 channels, so I had to implement custom loading logic.

## 🛠️ My Implementation
To get this data ready for a PyTorch model, I had to build a robust pipeline that could load, normalize, and visualize the images efficiently.

### 1. Custom Dataset Class (`GlacierDataset`)
I wrote a custom class inheriting from `torch.utils.data.Dataset` to handle the file mapping.
* **File Mapping:** The script scans the directory and automatically matches the 5 separate band files (`Band1` to `Band5`) for every image ID.
* **Stacking:** I used **OpenCV** to read the bands individually and stacked them into a single 5-channel NumPy array (`H, W, 5`).
* **Normalization (The Tricky Part):** Satellite data is full of "hot pixels" (glare) that skew the data. Simple Min-Max scaling didn't work.
    * *My Solution:* I implemented **Percentile Clipping** (clipping the bottom 2% and top 2% of pixels) before normalizing. This removed the outliers and gave me a clean 0-1 range for the model to learn from.

### 2. Visualization Pipeline
You can't train what you can't see. I wrote a script to sanity-check the data:
* It extracts **Bands 4, 3, and 2** to create a **False Color RGB** composite (since we can't visualize 5 channels at once).
* It overlays the Ground Truth masks to ensure the pixel coordinates align perfectly with the input.

---
---
## 👤 Author
**Biswajit Nahak** *B.Tech ETC Student at IIIT BBSR* [GitHub Profile](https://github.com/Biswajitnahak2003) | [LinkedIn](https://www.linkedin.com/in/biswajit-nahak/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).