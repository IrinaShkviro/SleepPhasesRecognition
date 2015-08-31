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

count_of_hmm = 7

class HMM(object):
    def __init__(
        self,
        n_visibles,
        n_hiddens,
        train_seqs,
        n_epochs,
        train_data_names,
        valid_seqs,
        test_seqs
    ):
        """ This class is made to train HMMs for each label definetly.
        :param n_visibles: array where each element point on count of 
                visible elements for each HMM
        :param n_hiddens: array where each element point on count of 
                hidden elements for each HMM
        :param visible_seqs: array with input for each HMM
        :param n_epochs: array which point on training epochs count for each HMM
        """

        self.HMMs=[]
        for label in xrange(count_of_hmm):
            current_HMM = HMM_for_one_label(n_visible=n_visibles[label],
                                          n_hidden=n_hiddens[label],
                                          train_data=train_seqs[label],
                                          n_epoch=n_epochs[label],
                                          patient_list=train_data_names,
                                          valid_data = valid_seqs[label],
                                          test_data = test_seqs[label]
                                          )
            current_HMM.train()
            self.HMMs.append(current_HMM)
            
    def recognition(self, visible_seq):
        probabilities = [self.HMMs[i].probability_for_seq(visible_seq) 
                            for i in xrange(count_of_hmm)]
        return numpy.argmax(probabilities)
       
def train():
    train_data_names = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    valid_data = ['p09b','p023','p035','p038']
    test_data = ['p09a','p033']
        
    train_reader = ICHISeqDataReader(train_data_names)
    #get data divided on sequences with respect to labels
    train_visible_seqs = train_reader.read_all_and_divide()
        
    valid_reader = ICHISeqDataReader(valid_data)
    valid_visible_seqs = valid_reader.read_all_and_divide()
    
    test_reader = ICHISeqDataReader(test_data)
    test_visible_seqs = test_reader.read_all_and_divide()
    
    rank = 1
    base = pow(10, rank) + 1
    n_visible_labels = pow(base, 3)
    n_visibles = [n_visible_labels] * 7
    n_hiddens = [200] * 7
    n_epochs = [1] * 7
    
    trained_HMM = HMM(n_visibles=n_visibles,
                      n_hiddens=n_hiddens,
                      train_seqs=train_visible_seqs,
                      n_epochs=n_epochs,
                      train_data_names=train_data_names,
                      valid_seqs = valid_visible_seqs,
                      test_seqs = test_visible_seqs)
        
if __name__ == '__main__':
    train()
