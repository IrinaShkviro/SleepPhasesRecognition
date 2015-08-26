 # -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:04:17 2015

@author: Ren
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:02:02 2015

@author: Ren
"""

import os
import sys
import timeit
import gc

import numpy

import theano
import theano.tensor as T

from ichi_seq_data_reader import ICHISeqDataReader

"""
class HMM(object):
    def _init_(self, n_visible, n_hidden, input, label):
        
"""       
def train():
    #get data divided on sequences with respect to labels
    #visible_seqs = ...
        
    train_data = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    valid_data = ['p09b','p023','p035','p038']
    test_data = ['p09a','p033']
        
    train_reader = ICHISeqDataReader(train_data)
    train_visible_seqs = train_reader.read_all_seqs_on_labels()
        
    valid_reader = ICHISeqDataReader(valid_data)
    valid_visible_seqs = valid_reader.read_all_seqs_on_labels()
    
    test_reader = ICHISeqDataReader(test_data)
    test_visible_seqs = test_reader.read_all_seqs_on_labels()
        
if __name__ == '__main__':
    train()
