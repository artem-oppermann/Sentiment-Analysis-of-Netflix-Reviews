# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:30:06 2018

@author: Artem Oppermann
"""
import tensorflow as tf

class BaseModel:
    
    def __init__(self,FLAGS,word2idx):
        
        self.FLAGS=FLAGS
        self.vocab_size=len(word2idx)
    
    
    def inference(self, x, seq_length):

        with tf.variable_scope('embedding'):
            We=tf.get_variable('embedding_matrix', shape=(self.vocab_size, self.FLAGS.embedding_size), dtype=tf.float32)
            lstm_input=tf.nn.embedding_lookup(We, x)
        
        lstm_cell_fw=tf.contrib.rnn.LSTMCell(self.FLAGS.lstm_units)
        lstm_cell_bw=tf.contrib.rnn.LSTMCell(self.FLAGS.lstm_units)
        
        ((output_fw, output_bw) , 
         (encoder_fw_final_state, encoder_bw_final_state))=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                                           cell_bw=lstm_cell_bw,
                                                                                           inputs=lstm_input,
                                                                                           sequence_length=tf.reshape(seq_length, [-1]),
                                                                                           dtype=tf.float32,
                                                                                           time_major=False,
                                                                                           )
        fw_h=tf.cast(encoder_fw_final_state.h, tf.float32)
        bw_h=tf.cast(encoder_fw_final_state.h, tf.float32)
        encoder_final_state_h = tf.concat((fw_h, bw_h), 1)
        
        with tf.variable_scope('output'):
            Wo=tf.get_variable('output_matrix', shape=(2*self.FLAGS.lstm_units,self.FLAGS.num_classes))
            bo=tf.get_variable('output_bias', shape=(self.FLAGS.num_classes), initializer=tf.zeros_initializer)
        
        logits=tf.matmul(encoder_final_state_h,Wo)+bo
                        
        return logits, tf.argmax(tf.nn.softmax(logits),1)
    

    