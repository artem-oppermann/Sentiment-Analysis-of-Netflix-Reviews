import string
from nltk import pos_tag, word_tokenize
import numpy as np
import re
import tensorflow as tf
from sklearn.utils import shuffle
import json
import os
import os.path as path
from pathlib import Path
import random

root_path = Path(__file__).parents[2]

POS_FILES_PATH=os.path.abspath(os.path.join(root_path, 'data/raw/rt-polarity.pos'))  
NEG_FILES_PATH=os.path.abspath(os.path.join(root_path, 'data/raw/rt-polarity.neg'))   

POS_FILES_CLENED_PATH=os.path.abspath(os.path.join(root_path, 'data/preprocessed/pos_review_cleaned.txt'))
NEG_FILES_CLENED_PATH=os.path.abspath(os.path.join(root_path, 'data/preprocessed/neg_review_cleaned.txt'))

FILES_CLEANED_LABELED=os.path.abspath(os.path.join(root_path, 'data/preprocessed/reviews_labeled.txt'))
FINAL_FILE=os.path.abspath(os.path.join(root_path, 'data/preprocessed/reviews_shuffled.txt'))     

TRAIN_DATA=os.path.abspath(os.path.join(root_path, 'data/preprocessed/train.txt')) 
TEST_DATA=os.path.abspath(os.path.join(root_path, 'data/preprocessed/test.txt'))

WORD2IDX_PATH=os.path.abspath(os.path.join(root_path, 'data/preprocessed/word2idx.txt'))


TEST_SIZE=0.2


def clean_data():
    '''Remove numbers and other special characters from the sentences in the raw files
    and write cleaned sentences to new files.'''
    

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
    '''Assign a number to each unique word. Create a dictionary where each key is a unique word with an integer 
    as value and write this dictionary to a file. '''
    
    cleaned_files_path=[POS_FILES_CLENED_PATH, NEG_FILES_CLENED_PATH]

    word2idx={'PAD':0}
    num_words=1
    
    for file in cleaned_files_path:
        for line in open(file):
            tokens=word_tokenize(line)
            for token in tokens:     
                if token not in word2idx:
                    word2idx[token]=num_words
                    num_words+=1             
                    
    with open(WORD2IDX_PATH, 'w') as outfile:  
        json.dump(word2idx, outfile)  
        

def add_label():
    '''Add a label of either 0 (genative) or 1 (positive) to each review and write it to a new .txt-file'''
    
    cleaned_files_path=[NEG_FILES_CLENED_PATH, POS_FILES_CLENED_PATH]
    labels=[0,1]

    with open(FILES_CLEANED_LABELED, 'w') as writer:
        for file, label in zip(cleaned_files_path, labels):
            for line in open(file):
                line=line.rstrip('\n') + '| ' + str(label)+'\n'
                writer.write(line)    
                
                        
def shuffle_file():
    '''Shuffle the data in the file '''
    with open(FILES_CLEANED_LABELED,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(FINAL_FILE,'w') as target:
        for _, line in data:
            target.write( line )                   
                        
                                            

          

def count_labels():
    '''Count the number of positive and negative reviews '''
    
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

def print_word2idx():
    '''Print the word2idx dictionary '''
    with open(WORD2IDX_PATH) as json_file:  
        word2idx = json.load(json_file)
        
    cleaned_files_path=[POS_FILES_CLENED_PATH, NEG_FILES_CLENED_PATH]
    
    for file in cleaned_files_path:
        for line in open(file):
            tokens=word_tokenize(line)
            if len(tokens)>1:
                for token in tokens: 
                    word2idx[token]  

def file_len(fname):
    '''Count the number of lines in the file '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1  

def train_test_split():  
    '''Split the data into training and testing set '''
                
    n_lines=file_len(FINAL_FILE)    
    n_train=int(n_lines*(1.0-TEST_SIZE))
        
    train_writer= open(TRAIN_DATA, 'w')
    test_writer= open(TEST_DATA, 'w')
    
    with open(FINAL_FILE) as f:
        for i, l in enumerate(f):
            if i<n_train:
                train_writer.write(l)
            else:
                test_writer.write(l)
          
        
if __name__ == "__main__":
    
    clean_data()
    write_word2idx()
    add_label()
    shuffle_file()
    #print_word2idx()
    train_test_split()
    count_labels()









