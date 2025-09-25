# Week 2: Data Pipeline with PyTorch Dataset and DataLoader class
### Objective
  The goal of this week was to prepare our custom, multi-layered glacier dataset for efficient training with a neural network.
  This involved creating a robust data pipeline using PyTorch's Dataset and DataLoader classes.
  
## What did we do?
#### We built a custom class called GlacierDataset which acts as a "recipe book" for our data. It tells PyTorch exactly how to:
  Find the 5 separate band files and 1 mask file for any given sample.Load them using rasterio.(week1)
  Stack the 5 bands into a single 5-channel numpy array and then to pytorch tensor.

### Why is the DataLoader so important?
#### While we could have loaded all the data into a big list, using a DataLoader is the professional standard for two critical reasons:
Memory Efficiency:  

  Instead of loading all 25 high-resolution images into RAM at once, it loads them in small batches (e.g., 4 at a time). 
  This allows us to train models on datasets that are much larger than our computer's memory.

Better Training:
  
  The DataLoader automatically shuffles the data every epoch (training cycle).
  This randomization is crucial for preventing the model from getting "stuck" and helps it learn more effectively.

### How dataset and dataloader works?
DATASET:
  this tells PyTorch what our data is and how to access one sample at a time.
  Basically we create a custom class that inherits Dataset class, where we define 3 functions
- __init__
    This is the constructor. It runs only once when we create the dataset.
    It sets up the list of files to use.
      
- __len__
  This method simply returns the total number of samples in the dataset.
  PyTorch uses this to know how big the dataset is
    
- __getitem__
   This is the most important method. PyTorch calls this to get a SINGLE sample.   

$ then we create an instance of our dataset and then create dataloader

DATALOADER:
  This tells PyTorch how to batch, shuffle, and load multiple samples in parallel.
  
  Basically we define this as a object of of pytorch's DataLoader, 
  which shuld have dataset,which we defined earlier and some hyperparameters.
  
  For ex- 
  
      shuffle = True(this suffles data if true)
      batch_size = integer(this tell how many data we are picking for training at once)
      num_workers = integer(for parallel processing
      )
    
## Outcome
#### We now have a train_loader and a val_loader that can efficiently feed batches of pre-processed, 5-channel image tensors and their corresponding masks to any PyTorch model.

## Author 
  Biswajit Nahak
  B.Tech | ETC | @IIIT Bhubaneswar
