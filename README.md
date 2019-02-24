# Sentiment-Analysis-of-Netflix-Reviews

In this project a Neural Network model based on Recurrent Neural Networks aims to predict whether a Netflix review conveys a positive or a negative sentiment. For each review the model predicts a percentage value for conveying a particular sentiment.

In case of Recurrent Neural Network I am using Long-Short-Term-Memory (LSTMs) networks. Via input arguments the user can specify whether these LSTMs should be bi-directional or uni-directional. A Dropout wrapper around the LSTMs prevents the overfitting.

## Getting Started
### Prerequisites

TensorFlow 1.5

Python 3.7.1

### Data

### Start the Training of The Model

To start the training of the model run the python script `src\train.py`, with (optionaly) your own hyperparameters to overwrite the existing ones. For example:

      python src\train.py \
            --num_epoch=25 \
            --batch_size=32 \
            --learning_rate=0.0005 \
            --architecture=uni_directional \
            --lstm_units=100  \
            --dropout_keep_prob=0.5 \
            --embedding_size=100  \
       
       
            epoch_nr: 0, train_loss: 0.654, train_acc: 0.629, test_acc: 0.737
            epoch_nr: 1, train_loss: 0.451, train_acc: 0.809, test_acc: 0.753
            epoch_nr: 2, train_loss: 0.294, train_acc: 0.889, test_acc: 0.762
            epoch_nr: 3, train_loss: 0.205, train_acc: 0.930, test_acc: 0.740
            epoch_nr: 4, train_loss: 0.141, train_acc: 0.951, test_acc: 0.754
            epoch_nr: 5, train_loss: 0.108, train_acc: 0.965, test_acc: 0.743
            epoch_nr: 6, train_loss: 0.085, train_acc: 0.973, test_acc: 0.723
            epoch_nr: 7, train_loss: 0.069, train_acc: 0.977, test_acc: 0.738
            epoch_nr: 8, train_loss: 0.055, train_acc: 0.984, test_acc: 0.725
            epoch_nr: 9, train_loss: 0.048, train_acc: 0.986, test_acc: 0.727
            epoch_nr: 10, train_loss: 0.044, train_acc: 0.987, test_acc: 0.723

### Run Inference Tests



## Deployment


### Authors
