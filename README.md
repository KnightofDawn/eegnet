#eegnet
Google DeepMind's WaveNet neural network implementation for epileptic seizures detection in raw iEEG data.


## Intro
This code was developed for the [Kaggle - Melbourne University Seizure Prediction](https://www.kaggle.com/c/melbourne-university-seizure-prediction), where **eegnet_v1 achieved AUC = 0.63 with just ~10 epochs (which took 15h) in [Google Cloud Machine Learning](https://cloud.google.com/ml/)**. No GPUs were used due to unavailability, although its is highly recomended.


#### Features:
- Code developed using [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).
- Data input using [TFRecords](https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#reading-data) with [TF-Slim Dataset Descriptor](https://github.com/tensorflow/models/tree/master/slim)
- Train code runs in single machine or distributed  - [Between-graph replication](https://www.tensorflow.org/versions/r0.12/how_tos/distributed/index.html#replicated-training)
- Scripts to run in gcloud.


## Table of contents

<a href="#Install">Installation and setup</a><br>
<a href='#Motivation'>Motivation</a><br>
<a href='#Data_challenge'>Dataset and challenge</a><br>
<a href='#eegnet_network'>eegnet network</a><br>
<a href='#Train'>Training eegnet</a><br>
<a href='#Eval'>Evaluating</a><br>
<a href='#Submit'>Generate submission</a><br>

## Installation and setup
<a id='Install'></a>


## Motivation
<a id='Motivation'></a>

The intent was from the beggining to use a neural network inspired on [Google DeepMind's WaveNet](https://arxiv.org/pdf/1609.03499.pdf) direclty on raw iEEG data.

Reading the WaveNet paper was truly inspirational: a demonstration of the power of deep neural networks in extracting relevant features directly from raw audio data. **It is a perfect fit for other kinds of challenging raw data such as brain waves!**


## [Dataset and challenge](https://www.kaggle.com/c/melbourne-university-seizure-prediction/data)
<a id='Data_challenge'></a>

- 10 min segments at 400Hz - 240000 points
- 16 input channels
- 5047 training files (80/20 split for validation)
- 1908 test files

#### Challenge
> The challenge is to distinguish between ten minute long data clips covering an hour prior to a seizure, and ten minute iEEG clips of interictal activity. Seizures are known to cluster, or occur in groups. Patients who typically have seizure clusters receive little benefit from forecasting follow-on seizures. For this contest only lead seizures, defined here as seizures occurring four hours or more after another seizure, are included in the training and testing data sets. In order to avoid any potential contamination between interictal, preictal, and post-ictal EEG signals, interictal data segments were restricted to be at least four hours before or after any seizure. Interictal data segments were chosen at random within these restrictions.


## eegnet network
<a id='eegnet_network'></a>

The TF network code can be found in [eegnet/src/eegnet/eegnet_vX.py](https://github.com/projectappia/eegnet/tree/master/src/eegnet).

The main difference between wavenet and eegnet resides in eegnet being trained only with a classifcation loss. Due to the nature of the data, 16 input channels, it was discarded training the network also predicting the next sample point. This compromise, on the other hand, allows eegnet to be applied directly on raw data of 16 input channels, without any companding transformation as in WaveNet.

eegnet uses **dilated convolutions** as opposed to LSTM to model the intrinsic characteristics of the input data. This allows eegnet to be applied to inputs of any size, altough given the input file size of 240000 points, to alleviate computation resourches, it is given the choice to the user of splitting the input file into smaller size segments. As will be explained later, this showed to produce a collateral effect of increasing the _'dataset variability'_, decreasing the learning rate. It was concluded that for this dataset a full input file of 10 min is the best compromise between necessary information for seizure detection and _'dataset variability'_.

As in WaveNet use for speech recognition, an **average pool layer** is placed on top of the **input** and **dilated convolution blocks** to aggregate the activations to a fixed size from which **logits** are extracted using normal concolutions and a fully connected layer.

#### Reasons behind eegnet architecture:
- Bigger average pool size increases accuracy.
- Bigger splits size increases accuracy. In fact, whole file (split 1) is essential to get good results, it is essential for training on whole dataset, otherwise the variability of small splits sizes is too big for model to generalize and learn.
- eegnet_v2, a smaller version with only two dilated blocks, no residual connections, achieved good learning rates, specially because it was possible to do many more epochs than with eegnet_v1. Altough, results on validation and test data (through Kaggle submission) showed that the network didn't generalize well.
- Logits from last dilated block vs logits from skip connections (as in WaveNet) showed slightly worse results in our tests.
- It was attempted to increase learning rate by training with two logits losses: logits from last dilated block and logits from skip connections. No improvements were observed.
- Bigger batch sizes produced better results. Given the input file size, one can easily ran out of RAM with big batch sizes, altough going for online training (batch_size = 1) is not advised. Good results were achieved with batch_size = 7 (limitted by gcloud complex_model_l RAM).
- Introducing more convolutions in logits processing didn't improve results, which makes sense, the main feature extractors and core of this network are the dilated convolution blocks.
- Best initial learning rate determined to be 1e-3, bigger would cause the network not to converge.
- ADAM optimizer produced good results and was choosen.

#### Lessons learnt:
- Weights and biases regularization and dropout are essential to fight overfitting.
- Start train in a small train dataset and small network, if using inception or wavenet blocks, implement blocks but keep number small.
- With stacked convolution layers the total receptive field increases with depth increase. Atrous/dilated convolutions increase the effect even more while keeping computation down.
- With residual connections don't apply activation and normalization functions.
- Almost always use activation and normalization functions between layers, even in compress or expand 1x1 convolutions
- Last fully connected layer before softmax DON'T use activation or normalization functions.
- It's good practice to use a 1x1 feature compressing layer before a normal 3x3/5x5/7x7 convolution. Makes computation faster and uses proven embeddings principle of a compressed data representation.
- ALWAYS normalize input data with MEAN = 0 and STD < 1. STD < 1 is important as well to avoid activation functions saturation.
- Don't look only into minibatch loss, minibatch and validation accuracy are as important. I have seen minibatch loss not decreasing much while accuracy increases.

With more computation resources it would be interesting to try different optimizers and bigger batch sizes.

#### Important:
The main constraint on using eegnet directly on raw data is the computational resources necessary. GPUs are highly recomended but were still unavailable in gcloud at the time of development of this project. AWS is also being investigated but nothing to report at the moment still.

eegnet_v1 achieved the abovementioned results with only a **~10 epochs** of training and having only **6 dilated blocks**. With more epochs and a network with 20+ dilated blocks as WaveNet, we believe the AUC results would have been truly inspiring.


## Training eegnet
<a id='Train'></a>



## Evaluating
<a id='Eval'></a>



## Generate submission
<a id='Submit'></a>


