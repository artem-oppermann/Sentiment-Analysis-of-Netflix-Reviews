# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:34:17 2018

@author: ARtem Oppermann
"""

import tensorflow as tf
from models.base_model import BaseModel

class TrainModel(BaseModel):
    
    def __init__(self):
        
        super(TrainModel,self).__init__()
        

    def optimize(self, x, labels, seq_length):
        
        with tf.variable_scope('inference'):
            logits, _=self.inference(x, seq_length)
        
        accuracy=self._compute_accuracy(logits, labels)
        loss=self._compute_loss(logits, labels)
        train_op = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(loss)
        
        return train_op, loss, accuracy
    
    
    def _compute_accuracy(self, logits, labels):
        
        with tf.name_scope('accuracy'):
            prediction=tf.nn.softmax(logits)
            labels=tf.cast(labels, tf.int64)
            accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.reshape(labels, shape=[-1])), tf.float32))
        return accuracy
        
    
    def accuracy(self, x, labels, seq_length):
        
        with tf.variable_scope('inference', reuse=True):
            logits, _=self.inference(x, seq_length)
        
        accuracy=self._compute_accuracy(logits, labels)
        
        return accuracy
    
    
    def _compute_loss(self, logits, labels):
        
        with tf.name_scope('loss'):
            loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=tf.reshape(labels, shape=[-1])
                                                                               ))
            if self.FLAGS.l2_reg==True:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                loss = loss +  self.FLAGS.lambda_ * l2_loss
        
        return loss