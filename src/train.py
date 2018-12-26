# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:38:43 2018

@author: Admin
"""

import tensorflow as tf
from utils import show_sample
from model import Model
import json
import numpy as np
from random import randint
import os

tf.app.flags.DEFINE_string('train_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/tf_records/train')), 
                           'Path for the training data.')
tf.app.flags.DEFINE_string('test_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/tf_records/test')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_string('word2idx', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/preprocessed/word2idx.txt')), 
                           'Path for the word2idx dictionary.')


tf.app.flags.DEFINE_integer('num_epoch', 5,
                            'Number of training epoch.'
                            )
tf.app.flags.DEFINE_integer('batch_size', 16,
                            'Batch size of the training set.'
                            )
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          'Learning rate of optimizer.'
                          )

tf.app.flags.DEFINE_boolean('l2_reg', True,
                            'L2 regularization.'
                            )
tf.app.flags.DEFINE_float('lambda_', 0.001,
                          'Lambda parameter for L2 regularization.'
                          )
tf.app.flags.DEFINE_integer('lstm_units', 20,
                            'Number of the LSTM hidden units.'
                            )

tf.app.flags.DEFINE_integer('embedding_size', 30,
                            'Dimension of the embedding vector for the vocabulary.'
                            )
tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of output classes.'
                            )

tf.app.flags.DEFINE_integer('eval_after', 100,
                            'Evaluate after number of batches.'
                            )

tf.app.flags.DEFINE_integer('num_train_samples', 9662,
                            'Number of all training sentences.'
                            )

tf.app.flags.DEFINE_integer('num_test_samples', 1000,
                            'Number of all training sentences.'
                            )

FLAGS = tf.app.flags.FLAGS


   

def main(_):
    
    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)
         
    num_batches=int(FLAGS.num_train_samples/FLAGS.batch_size)   
    
    training_graph=tf.Graph()
    
    with training_graph.as_default():
      
        training_set=_get_training_data(FLAGS)  
        test_set=_get_test_data(FLAGS)  
        
        iterator_train = training_set.make_initializable_iterator()
        iterator_test= test_set.make_initializable_iterator()

        line_encoded_train, label_encoded_train, labels_true, seq_length_train= iterator_train.get_next()
        line_encoded_test, label_encoded_test, _, seq_length_test=iterator_test.get_next()
        
        model=Model(FLAGS,word2idx)
        
        train_op, loss_op, acc_op_train=model.optimize(line_encoded_train, label_encoded_train, seq_length_train)
        acc_op_val=model.accuracy(line_encoded_test, label_encoded_test, seq_length_test)
        infer=model.inference(line_encoded_test, seq_length_test)
    
        with tf.Session(graph=training_graph) as sess:
            
            sess.run(tf.global_variables_initializer())
    
            for epoch in range(FLAGS.num_epoch):
                
                sess.run(iterator_train.initializer)

                traininig_loss=0
                training_acc=0
                
                for num_batch in range(num_batches):
                    
                    sess.run(iterator_test.initializer)
                    _, loss_, acc_=sess.run((train_op,loss_op,acc_op_train))
                    
                    traininig_loss+=loss_
                    training_acc+=acc_
                    
                    if num_batch!=0 and num_batch%FLAGS.eval_after==0:
                       
                        acc_val=sess.run(acc_op_val)
                    
                        print('epoch_nr: %i, batch_nr: %i/%i, train_loss: %.3f, train_acc: %.3f, val_acc: %.3f \n'%
                              (epoch, num_batch, num_batches,(traininig_loss/FLAGS.eval_after),(training_acc/FLAGS.eval_after),acc_val))
                        
                        traininig_loss=0
                        training_acc=0
                        
            show_sample(FLAGS, infer,sess, line_encoded_test)




if __name__ == "__main__":
    
    tf.app.run()