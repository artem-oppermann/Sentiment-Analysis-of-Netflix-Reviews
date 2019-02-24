# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:30:06 2018

@author: Artem Oppermann
"""
import tensorflow as tf

class BaseModel:
    
    def __init__(self,FLAGS, vocab_size):
        """
        Base class for the deep learning LSTM model for sentiment analysis. 
        Is enherited by the training and inference classes.
        
        :param FLAGS: tf.Flags
        :param vocab_size: number of words in the dataset
        """
        
        self.FLAGS=FLAGS
        self.vocab_size=vocab_size
    
    def __word_embeddings(self, x):    
        """High dimensional vector embeddings for the words
        
        :param x: unique index of the word that should be embedded
        
        :return embedded_words: High dimensional embedding of :param x, with shape [self.FLAGS.embedding_size]
        """
        
        with tf.variable_scope('word_embeddings'):
            We=tf.get_variable('embedding_matrix', shape=(self.vocab_size, self.FLAGS.embedding_size), dtype=tf.float32)
            embedded_words=tf.nn.embedding_lookup(We, x)
            
        return embedded_words

    def __cell(self, dropout_keep_prob):
        """
        Builds a LSTM cell with a dropout wrapper
        
        :return: LSTM cell with a dropout wrapper
        """
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.FLAGS.lstm_units, state_is_tuple=True)
        
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob,
                                                     output_keep_prob=dropout_keep_prob)
        return dropout_cell
    
    def __rnn_layer_unidirectional(self, x, seq_length, dropout_keep_prob, scope_name="rnn_layer"):
        """
        Builds a unidirectional LSTM layer
        :param x: Input with shape [batch_size, max_length]
        :param seq_len: Sequence length tensor with shape [batch_size,1]
        
        :return: outputs with shape [batch_size, max_seq_len, hidden_size]
        """
        with tf.variable_scope(scope_name, default_name='rnn_layer'):
            
            # Build LSTM cell
            lstm_cell = self.__cell(dropout_keep_prob)
            
            # Remove the unnecessary dimension 
            seq_length=tf.squeeze(seq_length)

            # Dynamically unroll LSTM cells
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_length)
            
            outputs_mean = tf.reduce_mean(outputs, reduction_indices=[1])
                     
        return outputs_mean
    
    def __rnn_layer_bidirectional(self, x, seq_length, dropout_keep_prob, scope_name="rnn_layer"):
        """
         Builds a bidirectional LSTM layer
        :param x: Input with shape [batch_size, max_length]
        :param seq_len: Sequence length tensor with shape [batch_size,1]
        :return: outputs with shape [batch_size, max_seq_len, hidden_size]
        """
        
        with tf.variable_scope(scope_name, default_name='rnn_layer'):
            
            lstm_cell_fw = self.__cell(dropout_keep_prob)
            lstm_cell_bw = self.__cell(dropout_keep_prob)
            
            ((output_fw, output_bw) , 
             (encoder_fw_final_state, encoder_bw_final_state))=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                                           cell_bw=lstm_cell_bw,
                                                                                           inputs=x,
                                                                                           sequence_length=tf.squeeze(seq_length),
                                                                                           dtype=tf.float32,
                                                                                           time_major=False,
                                                                                           )
            fw_h=tf.cast(encoder_fw_final_state.h, tf.float32)
            bw_h=tf.cast(encoder_fw_final_state.h, tf.float32)
            encoder_final_state_h = tf.concat((fw_h, bw_h), 1)
            
            return encoder_final_state_h
    

    def compute_prediction(self, x, seq_length, dropout_keep_prob, reuse_scope):
        """Wrapper method for the inference. Is used both during training and testing time.
        This is why its necessary to decide when a scope should be reused
        
        :param x: input sequences of shape [batch_size, max_length]
        :param seq_length: length of input sequences of shape [batch_size]
        :param dropout_keep_prob: probability to shut neurons down in the drop out wrapper for the LSTM
        :param reuse_scope: boolean, should the scope be reused?
            
        :return: logits and probabilities
        """
        
        with tf.variable_scope('inference', reuse=reuse_scope):
            
            logits, probabilities=self.inference(x, seq_length, dropout_keep_prob)
        
        return logits, probabilities  
        
    
    def inference(self, x, seq_length, dropout_keep_prob):
        """Perfoms the inference operation which is the computation of logits
        and probabilities for a certain sentiment
        
        :param x: input sequences of shape [batch_size, max_length]
        :param seq_length: length of input sequences of shape [batch_size]
        :param dropout_keep_prob: probability to shut neurons down in the drop out wrapper for the LSTM
            
        :return: logits and probabilities
        """
        
        embedded_words=self.__word_embeddings(x)
        
        # Build LSTM layers
        if self.FLAGS.architecture=="bidirectional":
            outputs = self.__rnn_layer_bidirectional(embedded_words, seq_length, dropout_keep_prob, "lstm_layer")
        elif self.FLAGS.architecture=="unidirectional":
            outputs = self.__rnn_layer_unidirectional(embedded_words, seq_length, dropout_keep_prob, "lstm_layer")
                     
        # Final output layer
        with tf.variable_scope('output_layer'):
            
            n_hidden=self.FLAGS.lstm_units
            
            if self.FLAGS.architecture=="bidirectional":
                n_hidden=self.FLAGS.lstm_units*2
            
            W1=tf.get_variable('W1', shape=(n_hidden,self.FLAGS.num_classes)) 
            
            logits=tf.matmul(outputs,W1, name='logits')
        
        return logits, tf.nn.softmax(logits, name='probabilities')

