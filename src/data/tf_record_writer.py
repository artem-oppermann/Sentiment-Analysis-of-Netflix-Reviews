import os
import tensorflow as tf
import sys
from scipy.misc import imread, imsave, imresize
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import json
from nltk import pos_tag, word_tokenize
from pathlib import Path

root_path = Path(__file__).parents[2]


WORD2IDX_PATH=os.path.abspath(os.path.join(root_path, 'data/preprocessed/word2idx.txt'))

TRAIN_DATA=os.path.abspath(os.path.join(root_path, 'data/preprocessed/train.txt'))  
TEST_DATA=os.path.abspath(os.path.join(root_path, 'data/preprocessed/test.txt')) 

OUTPUT_DIR=os.path.abspath(os.path.join(root_path, 'data/tf_records'))


def _get_tf_filename(output_dir,name_tf_file,num_tf_file):
    return '%s/%s_%i.tfrecord' % (output_dir, name_tf_file, num_tf_file)


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature2(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



def _add_to_tf_records(line, tf_writer, word2idx):
    
    splitted_line=line.split('|')
    line=splitted_line[0]
    label=int(splitted_line[1].rstrip('\n'))

    if label==1:
        label='positiv'
        label_encoded=[0,1]
    elif label==0:
        label='negativ'
        label_encoded=[1,0]
          
    idx_sequence=[]
    tokens=word_tokenize(line)
 
   
    for token in tokens:
        try:
            idx=word2idx[token]
        except KeyError:
            print('token: %s could not be found in the dictionary.'%token)
            continue
        
        idx_sequence.append(idx)
    
    idx_sequence=np.array(idx_sequence)
    seq_length=len(idx_sequence)
    idx_sequence=idx_sequence.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={'text_line/encoded': bytes_feature(idx_sequence),
                                                                   'text_line/seq_length': int64_feature(seq_length),
                                                                   'label/label':bytes_feature(tf.compat.as_bytes(label)),
                                                                   'label/encoded':int64_feature(label_encoded)
                                                                   }))
    
    tf_writer.write(example.SerializeToString())

def run(output_dir, name_tf_file, data):

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
        
    with open(WORD2IDX_PATH) as json_file:  
        word2idx = json.load(json_file)
    

    num_tf_file=0

    tfrecords_filename=_get_tf_filename(output_dir,name_tf_file,num_tf_file)
      
    with tf.python_io.TFRecordWriter(tfrecords_filename) as tf_writer:

        for line in open(data):      
            _add_to_tf_records(line, tf_writer, word2idx)
            

    print('\nFinished converting the text file')
    
    


if __name__ == "__main__":
    
    run(OUTPUT_DIR,'training_file', TRAIN_DATA)
    run(OUTPUT_DIR,'test_file', TEST_DATA)

