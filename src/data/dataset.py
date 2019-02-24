import tensorflow as tf
import numpy as np
import os

def get_training_data(FLAGS):

    filenames=[FLAGS.train_path]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=25000)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(FLAGS.batch_size,padded_shapes=([None],[None],[None], [None]))
    dataset = dataset.prefetch(buffer_size=4)

    return dataset


def get_test_data(FLAGS):

    filenames=[FLAGS.test_path]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(FLAGS.n_test_samples,padded_shapes=([None],[None],[None], [None]))
    dataset = dataset.prefetch(buffer_size=4)

    return dataset



def parse(serialized):

    context={'label/encoded':tf.FixedLenFeature([2], tf.int64),
             'text_line/encoded':tf.FixedLenFeature([], tf.string),
             'label/label':tf.FixedLenFeature([1], tf.string),
             'text_line/seq_length': tf.FixedLenFeature([1], tf.int64),                 
             }


    sequence_parsed=tf.parse_single_example(serialized,
                                           features=context,
                                           )

    line_encoded_raw=sequence_parsed['text_line/encoded']
    line_encoded = tf.decode_raw(line_encoded_raw, tf.int32)

    label_encoded  = sequence_parsed['label/encoded']
    label  = sequence_parsed['label/label']

    seq_length  = sequence_parsed['text_line/seq_length']


    return line_encoded, label_encoded, label, seq_length