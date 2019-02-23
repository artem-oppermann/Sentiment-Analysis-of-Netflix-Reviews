# Sentiment-Analysis-of-Netflix-Reviews

In this project a Neural Network model based on Recurrent Neural Networks aims to predict whether a Netflix review conveys a positive or a negative sentiment. For each review the model predicts a percentage value for conveying a particular sentiment.

In case of Recurrent Neural Network I am using Long-Short-Term-Memory (LSTMs) networks. Via input arguments the user can specify whether these LSTMs should be bi-directional or uni-directional. A Dropout wrapper around the LSTMs prevents the overfitting.


The model consists of a bidirectional one layer long short term memory reccurent neural network. It is trained on roughly 5000 one-sentence labeled positiv and negativ Netflix reviews. The model's purpose is to learn to recognize a sentiment of a review and classify it whether as a positiv or a negativ review. 

The training of the model needs only about three epoch to reach an accuracy of roughly 75% on the validation set consisting of 1000 reviews. After that the accuracy goes never beyond 78 %. It can be observed that the training accuracy goes very fast towards 90-95% due to overfitting of the model. To prevent this I use a L2 regularization.
Here is an example for the training for 5 epochs and its summarys: 

