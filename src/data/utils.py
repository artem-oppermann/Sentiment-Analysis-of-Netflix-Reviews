import tensorflow as tf
import numpy as np
import json



def show_sample(FLAGS, sess, logits_test, probs_test, dropout_keep_prob, x):
    
    with open(FLAGS.word2idx) as json_file:  
        word2idx = json.load(json_file)
        
    idx2word = {v: k for k, v in word2idx.items()}    
    
    idx = np.random.choice(FLAGS.n_test_samples, size=5, replace=False)
    
    lines=sess.run(x)
    
    feed_dict={dropout_keep_prob:1.0}
    
    logits,probabilities =sess.run((logits_test, probs_test), feed_dict)
       
    lines=np.array(lines)[idx]
    probabilities=np.array(probabilities)[idx]
    logits=np.array(logits)[idx]

    print('\n\nTest Samples: \n')
    for line, probs in zip(lines, probabilities):
        translate_idx2word(idx2word, line, probs)
    
    
def translate_idx2word(idx2word, line, probs):
    
    line=[idx2word[idx] for idx in line]
    line=' '.join(line).replace('PAD', '')
    
    line=line.strip(" ")
    prediction=np.argmax(probs)
    
    '''
    if pred==0:
        sentiment='negativ'
    elif pred==1:
        sentiment='positiv'
    '''
    
    neg=probs[0]
    pos=probs[1]
    
    #print('REVIEW: %s \nSENTIMENT: %s \n' %(line,sentiment))
    print('  Review: "%s" \n' %(line))
    print('  pos. sentiment: %.2f %%' % pos)
    print('  neg. sentiment: %.2f %%' % neg)
    print('\n')
