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
from HMM_for_one_label import HMM_for_one_label

class HMM(object):
    def _init_(
        self,
        n_visibles,
        n_hiddens,
        visible_seqs,
        n_epochs,
        train_data
    ):
        """ This class is made to train HMMs for each label definetly.
        :param n_visibles: array where each element point on count of 
                visible elements for each HMM
        :param n_hiddens: array where each element point on count of 
                hidden elements for each HMM
        :param visible_seqs: array with input for each HMM
        :param n_epochs: array which point on training epochs count for each HMM
        """

        HMMs=[]
        for label in xrange(7):
            current_HMM = HMM_for_one_label(n_visible=n_visibles[label],
                                          n_hidden=n_hiddens[label],
                                          input=visible_seqs[label],
                                          label=label,
                                          n_epoch=n_epochs[label],
                                          patient_list=train_data)
            current_HMM.train()
            HMMs.append(current_HMM)
       
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
