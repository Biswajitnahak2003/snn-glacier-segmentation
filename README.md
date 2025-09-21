# SNN for Glacier Segmentation

###  This project is a academic endeavor to develop a Spiking Neural Network (SNN) for the semantic segmentation of glaciers from multi-band satellite imagery. The primary goal is to create a computationally efficient model that leverages the low-latency, event-driven nature of SNNs.

## Project Status

Week 1 (Completed): Environment setup, data exploration, and development of a robust data loading pipeline using glob and rasterio to handle the multi-folder data structure.

Week 2 (completed): Creation of a custom PyTorch Dataset and DataLoader to prepare the data for model training.

week 3 (upcoming): we will create a baseline u-net model

## Dataset
This project uses the proprietary dataset from the GlacierHack 2025 competition.

Structure: The dataset consists of 25 multi-band satellite images. Each image is composed of 5 spectral bands, stored as separate .tif files in their respective folders (Band1 through Band5).

Labels: Ground truth segmentation masks are provided in the label folder.

Access: Due to its proprietary nature, the train/ directory is not included in this repository. To run this project, the train/ folder must be acquired and placed in the root of this project directory.

## Progress

The current codebase includes 3 Jupyter Notebooks:
###### week1:
    visual_1.ipynb: Tryig out visulization of one image.
    week1_eda.ipynb: Demonstrates the initial exploratory data analysis and the core logic for loading and stacking the 5-band images from the complex file structure.
###### week2:
    week2_torchdataloader.ipynb: Created a function that takes images and their masks and returns pytorch tensor for training.

## Next Steps

Week 3: Implement a baseline U-Net model in PyTorch.

Week 4: Train and evaluate the baseline U-Net to establish a performance benchmark (MCC score).

Week 5-8: Convert the U-Net to a Spiking U-Net (S-U-Net) using snnTorch, train it, and compare its performance and efficiency against the baseline.

and more 

## Author
    
Biswajit Nahak
B.Tech | ETC | @IIIT Bhubaneswar
