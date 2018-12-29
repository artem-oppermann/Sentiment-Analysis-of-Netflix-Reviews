# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 15:49:07 2018

@author: Artem Oppermann
"""

import tensorflow as tf
import os
from models.inference_model import InferenceModel
import json

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'checkpoints/')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_string('export_path_base', os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'model-export/')), 
                           'Directory where to export the model.')

tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model.')  

tf.app.flags.DEFINE_string('word2idx', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/preprocessed/word2idx.txt')), 
                           'Path for the word2idx dictionary.')

tf.app.flags.DEFINE_string('architecture', 'unidirectional',
                          'Type of LSTM-Architecture, choose between "unidirectional" or "bidirectional"'
                          )

tf.app.flags.DEFINE_integer('lstm_units', 100,
                            'Number of the LSTM hidden units.'
                            )

tf.app.flags.DEFINE_integer('embedding_size', 100,
                            'Dimension of the embedding vector for the vocabulary.'
                            )
tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of output classes.'
                            )

FLAGS = tf.app.flags.FLAGS

def run_inference():
    
    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)
    
    inference_graph=tf.Graph()
    
    with inference_graph.as_default():
        
        infer_model=InferenceModel(FLAGS, len(word2idx))

        x=tf.placeholder(tf.int32, shape=[None, 78])
        seq_length_test=tf.placeholder(tf.int32, shape=[None, None])
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        _, probs_test=infer_model.compute_prediction(x, seq_length_test, dropout_keep_prob, reuse_scope=False) 
        
        saver = tf.train.Saver()
    
    with tf.Session(graph=inference_graph) as sess:
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_path)
        saver.restore(sess, ckpt.model_checkpoint_path)      
        
         
        # Save the model
        export_path = os.path.join(tf.compat.as_bytes(FLAGS.export_path_base),
                                   tf.compat.as_bytes('model_v_%s'%str(FLAGS.model_version)))
        
        print('Exporting trained model to %s'%export_path)
        
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        
        # create tensors info
        predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(x)
        predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(probs_test)
            
        # build prediction signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': predict_tensor_inputs_info},
                outputs={'scores': predict_tensor_scores_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
            
        # save the model
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_probability': prediction_signature
            })

        builder.save()
        
        
if __name__ == "__main__":
    run_inference()










