# Week 3: Baseline U-Net Model and updated week1 and week2 by adding data augmentation
## Objective
  The primary goal of this week was to define and verify the architecture for our baseline deep learning model, a U-Net. 
  This model will serve as the performance benchmark that our final Spiking U-Net will be compared against.
  This document also outlines the data augmentation strategy that was finalized in our data pipeline.

#### Part 1: Data Augmentation
###### What is Data Augmentation?
  Data Augmentation is a technique to artificially create new training data from existing data.
  This is done by applying simple, random transformations like flipping or rotating the original images.
  
###### Why are we adding this for our dataset?
  Our dataset is very small (only 25 images).
  A deep learning model can easily "memorize" these few images, a problem known as overfitting.
  When this happens, the model performs well on the data it has seen but fails completely on new, unseen images. 
  Augmentation helps prevent this.
  
###### How are we adding it?
  We use the powerful albumentations library inside our GlacierDataset class.
  For each image in our training set, we randomly apply the following transformations on-the-fly:

    HorizontalFlip: Flips the image and mask left-to-right.

    VerticalFlip: Flips the image and mask upside-down.

    RandomRotate90: Randomly rotates the image and mask by 90, 180, or 270 degrees.

#### Part 2: U-Net Model Architecture

This diagram outlines the flow of a 5-channel satellite image through our U-Net to produce the final 1-channel segmentation mask.

                                     Input Image (5, 256, 256)
                                               |
                                               V
+------------------------------------------+   |   +---------------------------------------------+
|           ENCODER (Contracting Path)       |   |   |           DECODER (Expanding Path)          |
|                                          |   |   |                                             |
|   [Conv Block x2, BatchNorm, ReLU]       |   |   |   [Conv Block x2, BatchNorm, ReLU]          |
|   `skip1` -> (64 channels, 256x256) -----|---|---|----------------------> [Concatenate] |
|                |                         |   |   |                               ^               |
|                V                         |   |   |                               |               |
|           [MaxPool 2x2]                  |   |   |                      [Up-Conv 2x2]              |
|                |                         |   |   |                               |               |
|                V                         |   |   |   (128 channels, 128x128) <- `d2`            |
|   (64 channels, 128x128)                 |   |   |                               |               |
|                |                         |   |   |                               |               |
|   [Conv Block x2, BatchNorm, ReLU]       |   |   |   [Conv Block x2, BatchNorm, ReLU]          |
|   `skip2` -> (128 channels, 128x128) ----|---|--> [Concatenate]                         |
|                |                         |   |   ^                                     |
|                V                         |   |   |                                     |
|           [MaxPool 2x2]                  |   |   [Up-Conv 2x2]                         |
|                |                         |   |   |                                     |
|                V                         |   |   |   (256 channels, 64x64) <- `b`          |
|   (128 channels, 64x64)                  |   |                                         |
|                                          |   |                                         |
+------------------------------------------+   |   +---------------------------------------------+
                                               |
                                               V
                               +----------------------------------+
                               | BOTTLENECK                       |
                               |                                  |
                               | [Conv Block x2, BatchNorm, ReLU] |
                               | `b` -> (256 channels, 64x64)     |
                               +----------------------------------+
                                               |
                                               V
                               +----------------------------------+
                               | OUTPUT                           |
                               |                                  |
                               | [1x1 Conv] -> (1 channel, 256x256)|
                               |      |                           |
                               |      V                           |
                               | [Sigmoid Activation]             |
                               +----------------------------------+
                                               |
                                               V
                                    Final Mask (1, 256, 256)

###### Explanation of Each Step

  Encoder (Contracting Path): This is the left side of the "U". 
  Its job is to act like a feature extractor.
  It uses convolutional layers to find patterns (like textures and edges) and max pooling layers to shrink the image.
  As the image gets smaller, the network is forced to learn more abstract, high-level concepts about what is in the image, rather than just where it is.

  Bottleneck: This is the lowest point of the "U".
  It holds the most compressed, high-level summary of the image content. 
  At this stage, the model has a strong semantic understanding of the scene (e.g., "this image contains a glacier and mountains").

  Decoder (Expanding Path): This is the right side of the "U". 
  Its job is to take the abstract summary from the bottleneck and "zoom in," using up-convolutional layers (ConvTranspose2d) to increase the image size back to the original. 
  This path rebuilds the image, but this time as a segmentation mask.

  Skip Connections: This is the most important feature of a U-Net.
  The horizontal arrows carry high-resolution feature maps from the encoder directly to the decoder.
  This is critical because the decoder needs to know precisely where to place the glacier boundaries. 
  The bottleneck provides the "what" (glacier), and the skip connection provides the "where" (the exact location and shape).

  Output Layer: After the final decoder block, we have a feature map with many channels (64 in our case).
  A final 1x1 Conv layer is used to collapse these 64 channels into a single output channel.
  A Sigmoid activation function is then applied to ensure the final output is a clean probability map, where each pixel's value is between 0 and 1. 
  This was a key step in stabilizing our model.

###### How to Further Improve This Model
  The current architecture is a strong and stable starting point. To improve performance in Week 4, we will explore the following professional techniques:

  Learning Rate Scheduler: Instead of a fixed learning rate, we will use a scheduler (like ReduceLROnPlateau). 
  This will automatically lower the learning rate when the model's performance on the validation set stops improving, allowing for more precise fine-tuning.

  Advanced Loss Functions: While BCELoss is stable, a combined Dice Loss + BCELoss often yields better MCC scores for segmentation.
  Dice Loss is excellent at handling class imbalance (e.g., if there are far more non-glacier pixels than glacier pixels).

  Hyperparameter Tuning: We can systematically experiment with different learning rates, batch sizes,
  and even the depth of the U-Net (e.g., adding another encoder/decoder block) to find the optimal configuration for our specific dataset.

#### Author 
  Biswajit Nahak

  B.Tech | ETC | @IIIT Bhubaneswar
  
