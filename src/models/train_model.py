# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:34:17 2018

@author: ARtem Oppermann
"""

import tensorflow as tf
from models.base_model import BaseModel


class TrainModel(BaseModel):
    
    def __init__(self, FLAGS, vocab_size):
        """
        Training class for the deep learning LSTM model for sentiment analysis. 
        Is used only for training. Inference is performance by the special inference class.
        This class inherits from the base class.
        
        :param FLAGS: tf.Flags
        :param vocab_size: number of words in the dataset
        """
        
        super(TrainModel,self).__init__(FLAGS, vocab_size)
             
    def train(self, loss):
        """Training operation.
        
        :param: loss operation to compute cross entropy loss
        """
        
        with tf.name_scope('training_step'):
            
            trainable_variables=tf.trainable_variables() 
            gradients= tf.gradients(loss, trainable_variables) 
            optimizer=tf.train.AdamOptimizer(self.FLAGS.learning_rate)
            train_op=optimizer.apply_gradients(zip(gradients, trainable_variables)) 

        return train_op
    
    def compute_accuracy(self, predict, target):
        """
        :param predict: Softmax activations with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Accuracy mean obtained in current batch
        """
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        return accuracy
        


    def compute_loss(self, scores, target):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Cross entropy loss mean
        """
        
        # Compute cross entropy loss of shape [batch_size]
        with tf.name_scope('cross_entropy'):
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=target, name='cross_entropy')
        
        # Take the mean of the ross entropy loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(cross_entropy_loss, name='loss')
        return loss
    
    
    
    
    
    
    
    
    
    
    
    
    