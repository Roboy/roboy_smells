# Classification 
This directory contains all files relevant for the classification of different smells.

During the semester we tried several models/approaches for classification. Below you can find a summary of those approaches.

# Models
### 1. Recurrent Models
These models have the property to learn from sequential data by taking into account historical data. We assume that our sensor data contains informative patterns in the course of the channels before saturating. This way the classifier can make predictions before the sensor is fully saturated. For the recurrent layer we tried both, simple RNN layers as well as Long short-term memory (LSTM) layers that are able to learn which information should be remembered. [1] 
Both models consist in this recurrent layer followed by 1 - 2 fully connected layers. 
Furthermore we made experiments with both stateful and stateless models. More information about stateful models can be found [here](https://www.tensorflow.org/guide/keras/rnn). 

Our best performing model architecture consists in a stateful LSTM layer followed by a fully connected layer that returns logits for the different classes at each time step, which we call SmelLSTM.

### 2. 1dCNN
The 1dCNN used here is based on the WaveNet architecture originally designed to work with audio signals [2]. The network - by design - has a lot of parameters even when we tried to simplify it, it still has >100.000 parameters which was too large a network to train with the little amount of data we have. This is why we ultimately switched to using our RNN and SmelLSTM approaches.
When prototyping the idea for the 1dCNN we tried two different loss function that provided similar results. 
Triplet Loss: The idea behind the triplet loss is to have an anchor datapoint (measurement in our case), a different positive sample from the same class and a negative sample from another class. The network then learns that the anchor and the positive sample should be close together in the latent space and the anchor and the negative sample far away. [3]
We wanted to try this as it would also directly provide a nice visual representation since we can show the resulting latent space in 2 or 3 dimensions using PCA.

The version of the CNN with the triplet loss can be trained by using the train_cnn1d.py file. The tune configuration can be changed as needed for hyperparameter tuning.
Classic Categorical Cross Entropy: The second approach was to directly predict the classes using a cross entropy loss function. The CNN using this loss can be trained using the train_cnn1d_cel.py file.

### 3. Baysian Approach
For the baysian approach, the saturated sensor values are assumed to be Gaussian distributed. During training mean and covariance matrix of a Gaussian are fit to the underlying data for each class. During inference a new datapoint the parameterized model is used to find the corresponding class with the highest probability. 
We had decent results using this approach but on average they were worse than the results we got using the SmelLSTM

### 4. kNN
The most simple approach we used was a k-nearest-neighbor (kNN). Here we assumed that the saturated sensor values of the same class have a small distance. During inference the classes of the k nearest neighbors are used to predict the class of the new datapoint by performing a (weighted) majority vote. Our best performing model uses 5 neighbors, the euclidean space and a distance weighting.

## Training new models:
New models can be trained by running the respective files and functions as detailed in the respective python files. For all training runs the training data csv files need to be placed into the data_train directory and the validation data into data_val, respectively. 
#### Simple model fitting
The simple models (kNN, naive Bayes) use an internal fit function that is called when a model instance is created (see knn.py and naive_bayes.py). 
#### Neural network training
For training the neural networks the ray.tune library was used for automated training and testing with parallelized hyperparameter search. The hyperparameter search space as well as the location to store the trained models and the checkpoint frequency can be configured in the train_*model_name*.py files. The documentation of the ray.tune library can be found [here](https://ray.readthedocs.io/en/latest/tune.html).
ray.tune automatically generates tensorboard files that can be viewed by launching tensorboard with the respective folder.

[1] [Learning to Forget: Continual Prediction with LSTM](https://www.researchgate.net/publication/12292425_Learning_to_Forget_Continual_Prediction_with_LSTM)

[2] [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

[3] [Triplet Loss](https://en.wikipedia.org/wiki/Triplet_loss)
