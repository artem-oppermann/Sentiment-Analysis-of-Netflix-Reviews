import tensorflow as tf
import numpy as np
import json



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



