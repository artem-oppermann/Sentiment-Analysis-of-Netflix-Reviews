import tensorflow as tf


class Model:
    
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
        
        
            
            
            