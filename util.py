import tensorflow as tf
import numpy as np
import json

def _get_training_data(FLAGS):
    
    filenames=[FLAGS.tf_records_train_path]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=25000)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(FLAGS.batch_size,padded_shapes=([None],[None],[None], [None]))
    dataset = dataset.prefetch(buffer_size=4)
    
    return dataset
    

def _get_test_data(FLAGS):
    
    filenames=[FLAGS.tf_records_test_path]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(1000,padded_shapes=([None],[None],[None], [None]))
    dataset = dataset.prefetch(buffer_size=4)
    
    return dataset



def parse(serialized):
    
    context={'label/encoded':tf.FixedLenFeature([1], tf.int64),
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



def show_sample(FLAGS, infer,sess, line_encoded_test):
    
    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)
        
    idx2word = {v: k for k, v in word2idx.items()}    
    
    idx = np.random.choice(FLAGS.num_test_samples, size=20, replace=False)

    lines=sess.run(line_encoded_test)
    _,predictions=sess.run(infer)
    
    lines=np.array(lines)[idx]
    predictions=np.array(predictions)[idx]

    print('\n\nTEST SAMPLES: \n')
    for line, p in zip(lines, predictions):
        translate_idx2word(idx2word,line, p)
    
def translate_idx2word(idx2word,line, p):
    
    line=[idx2word[idx] for idx in line]
    line=' '.join(line).replace('PAD', '')
    
    if p==0:
        sentiment='negativ'
    elif p==1:
        sentiment='positiv'
    
    print('REVIEW: %s \nSENTIMENT: %s \n' %(line,sentiment))



