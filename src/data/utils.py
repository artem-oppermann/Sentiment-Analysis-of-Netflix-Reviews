import tensorflow as tf
import numpy as np
import json



def show_sample(FLAGS, logits_test, probs_test, sess, x_test, seq_length_test, dropout_keep_prob):
    
    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)
        
    idx2word = {v: k for k, v in word2idx.items()}    
    
    idx = np.random.choice(100, size=5, replace=False)

    x=tf.identity(x_test)
    
    lines=sess.run(x)
    
    logits,probabilities =sess.run((logits_test, probs_test))
    
    lines=np.array(lines)[idx]
    probabilities=np.array(probabilities)[idx]

    print('\n\nTEST SAMPLES: \n')
    for line, p, l in zip(lines, probabilities, logits):
        translate_idx2word(idx2word,line, p, l)
    
def translate_idx2word(idx2word,line, p, l):
    
    line=[idx2word[idx] for idx in line]
    line=' '.join(line).replace('PAD', '')
    
    if pred==0:
        sentiment='negativ'
    elif pred==1:
        sentiment='positiv'
    
    neg=p[0]
    pos=p[1]
    #print('REVIEW: %s \nSENTIMENT: %s \n' %(line,sentiment))
    print('REVIEW: %s \n positiv opinion: %f \n negative opinion: %f ' %(line, pos, neg))



