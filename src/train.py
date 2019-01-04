"""
Trains a LSTM network to perform Sentiment Analysis

Created on Wed Dec 26 18:38:43 2018

@author: Artem Oppermann
"""
import json
import os
import tensorflow as tf

from data.dataset import get_training_data, get_test_data
from models.train_model import TrainModel
from data.utils import show_sample


tf.app.flags.DEFINE_string('train_path', os.path.abspath(os.path.join(os.path.dirname( "__file__" ), '..', 'data/tf_records/training_file_0.tfrecord')), 
                           'Path for the training data.')
tf.app.flags.DEFINE_string('test_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/tf_records/test_file_0.tfrecord')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_string('word2idx', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/preprocessed/word2idx.txt')), 
                           'Path for the word2idx dictionary.')

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/model.ckpt')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epoch.'
                            )
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Batch size of the training set.'
                            )
tf.app.flags.DEFINE_float('learning_rate', 0.0005,
                          'Learning rate of optimizer.'
                          )

tf.app.flags.DEFINE_string('architecture', 'unidirectional',
                          'Type of LSTM-Architecture, choose between "unidirectional" or "bidirectional"'
                          )

tf.app.flags.DEFINE_integer('lstm_units', 100,
                            'Number of the LSTM hidden units.'
                            )

tf.flags.DEFINE_float('dropout_keep_prob', 0.5,
                      '0<dropout_keep_prob<=1. Dropout keep-probability')

tf.app.flags.DEFINE_integer('embedding_size', 100,
                            'Dimension of the embedding vector for the vocabulary.'
                            )
tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of output classes.'
                            )

tf.app.flags.DEFINE_integer('n_train_samples', 8529,
                            'Number of all training sentences.'
                            )

tf.app.flags.DEFINE_integer('n_test_samples', 2133,
                            'Number of all training sentences.'
                            )

FLAGS = tf.app.flags.FLAGS



def main(_):
    
    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)

    training_graph=tf.Graph()
    
    with training_graph.as_default():
        
        train_model=TrainModel(FLAGS, len(word2idx))
        
        training_dataset=get_training_data(FLAGS)
        test_dataset=get_test_data(FLAGS)
        
        iterator_train = training_dataset.make_initializable_iterator()
        iterator_test=test_dataset.make_initializable_iterator()
        
        x_train, y_train, _, seq_length_train = iterator_train.get_next()
        x_test, y_test, _, seq_length_test =iterator_test.get_next()
    
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        logits, probs, outputs_=train_model.compute_prediction(x_train, seq_length_train, dropout_keep_prob, reuse_scope=False)
        loss=train_model.compute_loss(logits, y_train)
        train_op=train_model.train(loss)
        accuracy_train = train_model.compute_accuracy(probs, y_train)
        
        logits_test, probs_test, _=train_model.compute_prediction(x_test, seq_length_test, dropout_keep_prob, reuse_scope=True) 
        accuracy_test = train_model.compute_accuracy(probs_test, y_test)

        saver=tf.train.Saver()
        
    with tf.Session(graph=training_graph) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        n_batches=int(FLAGS.n_train_samples/FLAGS.batch_size)   

        for epoch in range(FLAGS.num_epoch):
            
            sess.run(iterator_train.initializer)
            sess.run(iterator_test.initializer)
            
            traininig_loss=0
            training_acc=0
            
            test_acc=0
              
            feed_dict={dropout_keep_prob:0.5}
            
           
            for n_batch in range(0, n_batches):
              
                _, l, acc, logits_, probs_=sess.run((train_op, loss, accuracy_train, logits, probs), feed_dict)
                
                #out=sess.run(outputs_, feed_dict)
                #print(out)
                #print(out.shape)
                #sd
                
                traininig_loss+=l
                training_acc+=acc
            
            for n in range(0, FLAGS.n_test_samples):
                
                feed_dict={dropout_keep_prob:1.0}
                
                acc=sess.run(accuracy_test, feed_dict)
                test_acc+=acc
                    
            loss_avg=traininig_loss/n_batches
            acc_avg_train=training_acc/n_batches
            
            acc_avg_test=test_acc/FLAGS.n_test_samples
  
            print('epoch_nr: %i, train_loss: %.3f, train_acc: %.3f, test_acc: %.3f'%(epoch, loss_avg, acc_avg_train, acc_avg_test))

            traininig_loss=0
            training_acc=0
            
            #show_sample(FLAGS, train_model, sess, x_test, seq_length_test, dropout_keep_prob)
            
            if acc_avg_test>0.70:
                saver.save(sess, FLAGS.checkpoints_path)

                

if __name__ == "__main__":
    
    tf.app.run() 
    