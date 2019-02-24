# Sentiment-Analysis-of-Netflix-Reviews

In this project a Neural Network model based on Recurrent Neural Networks aims to predict whether a Netflix review conveys a positive or a negative sentiment. For each review the model predicts a percentage value for conveying a particular sentiment.

In case of Recurrent Neural Network I am using Long-Short-Term-Memory (LSTMs) networks. Via input arguments the user can specify whether these LSTMs should be bi-directional or uni-directional. A Dropout wrapper around the LSTMs prevents the overfitting.

## Getting Started
### Prerequisites

#### TensorFlow 1.5

#### Python 3.7.1

### Data

The data consists of 5000 negative and 5000 positive Netflix reviews. You can examine the reviews in `data/raw/`.

I am using `tf.Data API` to ensure a fast, high performance data input pipeline. `tf.DATA API` works the best if the data is in the  `tf.Records` format.

The cleaned and formatted reviews in `tf.Record` format can be found in `data\tf_records\`.

You can do the cleaning of the data by yourself by executing `src\preprocess\clean_file.py`. To export the data in `tf.Records` format execute `src\data\tf_records_writer.py`.

### Start the Training of The Model

To start the training of the model run the python script `src\train.py`, with (optionaly) your own hyperparameters to overwrite the existing ones. For example:

      python src\train.py \
            --num_epoch=25 \
            --batch_size=32 \
            --learning_rate=0.0005 \
            --architecture=uni_directional (or bi-directional) \
            --lstm_units=100  \
            --dropout_keep_prob=0.5 \
            --embedding_size=100  \
            --required_acc_checkpoint=0.7 \
            
The meaning of these hyperparameters can be found in the documentations of `tf.FLAGS` in `train.py`.        

After the execution, the training of the model should start. You can observe the training loss and the accuracy on the training set and test set. Accuracy gives the ratio of correctly predicted sentiment of a given Netflix Review. You may see results like these:
       
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

During training we can observe overfitting on the training set. Hence a lower value for `dropout_keep_prob` is suggested. 

After the accuracy on the test set reaches the value `required_acc_checkpoint`, the model begins to save checkpoints in `checkpoints/model.ckpt` of the underlying dataflow graph and the parameters of the network.
            

## Deployment

For deployment perposes the model must be exported in the `SavedModel` format. In order to do so execute the script `src\inference.py`:

            python src\inference.py \
                  --export_path_base==model-export/
      
Attention: Other hyparameters must stay the same as during the training of the network.

The created instance of `SavedModel` can be run in a Docker container in a cloud for example. (The documentation for this will be extended in the future.)


