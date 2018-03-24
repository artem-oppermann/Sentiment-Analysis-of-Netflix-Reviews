import string
from nltk import pos_tag, word_tokenize
import numpy as np
import re
import tensorflow as tf
from sklearn.utils import shuffle
import json


POS_FILES_PATH='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/rt-polarity.pos'               
NEG_FILES_PATH='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/rt-polarity.neg'

POS_FILES_CLENED_PATH='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/pos_review_cleaned.txt'       
NEG_FILES_CLENED_PATH='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/neg_review_cleaned.txt' 

FILES_CLEANED_LABELED='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/reviews_labeled.txt'       
FINAL_FILE='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/reviews_shuffled.txt'       

TRAIN_DATA='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/train.txt' 
TEST_DATA='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/test.txt' 

WORD2IDX_PATH='C:/Users/Admin/Desktop/deep_learning _local_datasets/rt-polaritydata/word2idx.txt' 



def remove_punctuation(s):
    return s.translate(string.punctuation)

def get_tags(s):
    tuples = pos_tag(word_tokenize(s))
    return [y for x, y in tuples]

def encode_training_data(x_train,word2idx):
    
    x_train_encoded=[]
    
    for sequence in x_train:
        
        x=np.array([word2idx[sequence[i]] for i in range(0,len(sequence))])
        x_train_encoded.append(x)
    
    return x_train_encoded


def clean_data():
    

    files_path=[POS_FILES_PATH,NEG_FILES_PATH]
    cleaned_files_path=[POS_FILES_CLENED_PATH, NEG_FILES_CLENED_PATH]
    
    for file, cleaned_file in zip(files_path, cleaned_files_path):
        
        with open(cleaned_file, 'w') as writer:
            for line in open(file):
                line=line.rstrip()
                if line:
                    line=re.sub('[^A-Za-z0-9]+', ' ', line)
                    line=line.lower()
                    if len(line)>1:
                        writer.write(line+'\n')
                        
                        
def write_word2idx():
    
    cleaned_files_path=[POS_FILES_CLENED_PATH, NEG_FILES_CLENED_PATH]

    word2idx={'PAD':0}
    num_words=1
    
    for file in cleaned_files_path:
        for line in open(file):
            tokens=word_tokenize(line)
            if len(tokens)>1:
                for token in tokens:     
                    if token not in word2idx:
                        word2idx[token]=num_words
                        num_words+=1             
                    


    
    with open(WORD2IDX_PATH, 'w') as outfile:  
        json.dump(word2idx, outfile)
                   

def word2idx():
    
    with open(WORD2IDX_PATH) as json_file:  
        word2idx = json.load(json_file)
        
    cleaned_files_path=[POS_FILES_CLENED_PATH, NEG_FILES_CLENED_PATH]
    
    for file in cleaned_files_path:
        for line in open(file):
            tokens=word_tokenize(line)
            if len(tokens)>1:
                for token in tokens:     
                    print(word2idx[token])   



def add_label():
    
    cleaned_files_path=[NEG_FILES_CLENED_PATH, POS_FILES_CLENED_PATH]
    labels=[0,1]

    with open(FILES_CLEANED_LABELED, 'w') as writer:
        for file, label in zip(cleaned_files_path, labels):
            for line in open(file):
                line=line.rstrip('\n') + '| ' + str(label)+'\n'
                writer.write(line)

                
def shuffle_file():
    
    import random
    with open(FILES_CLEANED_LABELED,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(FINAL_FILE,'w') as target:
        for _, line in data:
            target.write( line )
            

def count_labels():
    
    files=[TRAIN_DATA, TEST_DATA]
    
    for file in files:
        
        num_positives=0
        num_negatives=0
        
        for line in open(file):
            
            splitted_line=line.split('|')
            label_encoded=int(splitted_line[1].rstrip('\n'))
            
            if label_encoded==0:
                num_negatives+=1
            elif label_encoded==1:
                num_positives+=1
        
        print('File %s contains %s positives and %s negatives'%(file, num_positives, num_negatives))

                        
if __name__ == "__main__":
    
    clean_data()
    write_word2idx()
    add_label()
    shuffle_file()
    count_labels()
    word2idx()









