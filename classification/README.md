# Classification 
This directory contains all files relevant for the classification of different smells.

During the semester we tried several models/approaches for classification. Below you can find a summary of those approaches.

### Training new models:
New models can be trained by running the respective files as detailed in the respective sections below. For training the neural networks the ray.tune library was used. The documentation for this library can be found [here](https://ray.readthedocs.io/en/latest/tune.html).

ray.tune automatically generates tensorboard files that can be viewed by launching tensorboard with the respective folder.

# Models
### 1. SmelLSTM
### 2. RNN
### 3. 1dCNN
The 1dCNN used here is based on the WaveNet architecture originally designed to work with audio signals [1]. The network - by design - has a lot of parameters even when we tried to simplify it, it still has >100.000 parameters which was too large a network to train with the little amount of data we have. This is why we ultimately switched to using our RNN and SmelLSTM approaches.
When prototyping the idea for the 1dCNN we tried two different loss function that provided similar results. 
Triplet Loss: The idea behind the triplet loss is to have an anchor datapoint (measurement in our case), a different positive sample from the same class and a negative sample from another class. The network then learns that the anchor and the positive sample should be close together in the latent space and the anchor and the negative sample far away.
We wanted to try this as it would also directly provide a nice visual representation since we can show the resulting latent space in 2 or 3 dimensions using PCA.

The version of the CNN with the triplet loss can be trained by using the train_cnn1d.py file. The tune configuration can be changed as needed for hyperparameter tuning.
Classic Categorical Cross Entropy: The second approach was to directly predict the classes using a cross entropy loss function. The CNN using this loss can be trained using the train_cnn1d_cel.py file.
### 4. Baysian Approach
### 5. kNN

[1] [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
