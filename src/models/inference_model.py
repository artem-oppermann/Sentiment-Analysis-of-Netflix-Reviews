# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:37:49 2018

@author: Artem Oppermann
"""

from models.base_model import BaseModel

class InferenceModel(BaseModel):
    
    def __init__(self, FLAGS, vocab_size):
        
        super(InferenceModel,self).__init__(FLAGS, vocab_size)
