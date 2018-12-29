import tensorflow as tf
import numpy as np
import json



def show_sample(FLAGS, infer,sess, line_encoded_test):
    
    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)
        
    idx2word = {v: k for k, v in word2idx.items()}    
    
    idx = np.random.choice(FLAGS.num_test_samples, size=5, replace=False)

    lines=sess.run(line_encoded_test)
    logits,probs,predictions =sess.run(infer)
    
    lines=np.array(lines)[idx]
    predictions=np.array(predictions)[idx]

    print('\n\nTEST SAMPLES: \n')
    for line, pred, p, l in zip(lines, predictions, probs, logits):
        translate_idx2word(idx2word,line, pred, p, l)
    
def translate_idx2word(idx2word,line, pred, p, l):
    
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



