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
from MyVisualizer import visualize_hmm_for_one_label
from preprocess import generate_random_probabilities, generate_probabilities_for_matrix
from ichi_seq_data_reader import ICHISeqDataReader

debug_file = open('D:\Irka\Projects\NeuralNetwok\DeepLearning\debug_info.txt', 'w+')

class HMM_second(object):
    def __init__(self, n_visible, train_set, train_patient_list):

        self.n_hidden = 7
        self.n_visible = n_visible
        
        # split the datasets
        (train_set_x, train_set_y) = train_set

        train_set_hidden = train_set_y.eval()
        train_set_visible = train_set_x.eval()
        
        pi_values = numpy.zeros((self.n_hidden,))
        a_values = numpy.zeros((self.n_hidden, self.n_hidden)) 
        b_values = numpy.zeros((self.n_hidden, self.n_visible))
        array_from_hidden = numpy.zeros((self.n_hidden,))
        
        for patient_number in xrange(len(train_patient_list)):
            pi_values[train_set_hidden[patient_number][0]] += 1
            hiddens_patient = train_set_hidden[patient_number]
            visibles_patient = train_set_visible[patient_number]
            n_patient_samples = len(hiddens_patient)
                    
            for sample in xrange(n_patient_samples - 1):
                a_values[hiddens_patient[sample], hiddens_patient[sample+1]] += 1
                b_values[hiddens_patient[sample], visibles_patient[sample]] += 1
            b_values[hiddens_patient[n_patient_samples - 1],
                     visibles_patient[n_patient_samples - 1]] += 1
                     
            for hidden in xrange(self.n_hidden):
                array_from_hidden[hidden] += len(hiddens_patient[numpy.where(hiddens_patient==hidden)])
            array_from_hidden[hiddens_patient[n_patient_samples-1]] -= 1
            
        for hidden in xrange(self.n_hidden):
            a_values[hidden] = a_values[hidden]/array_from_hidden[hidden]
            b_values[hidden] = b_values[hidden]/array_from_hidden[hidden]
                
        self.Pi=theano.shared(
            value=pi_values/float(len(train_patient_list)),
            name='Pi'
        )
        
        self.A=theano.shared(
            value=a_values,
            name='A'
        )
        
        #B is matrix which consider probabilities observe visible element 
        #from hidden state
        self.B=theano.shared(
            value=b_values,
            name='B'
        )
        debug_file.write('end of init')
        
def viterbi_algo(HMM, visible_seq):
    debug_file.write('start_viterbi_algo')
    n_time = len(visible_seq)
    n_hidden = HMM.n_hidden
    debug_file.write(str(n_time) + 'n_time')
    qual = theano.shared(
        value = numpy.zeros((n_time, n_hidden)),
        name = 'quality'
    )
    arg = numpy.zeros((n_time, n_hidden))
    T.set_subtensor(qual[0], HMM.Pi * HMM.B[:][visible_seq[0]])
    arg[0, :] = [0] * n_hidden
        
    for time in xrange(n_time - 1):
        for j in xrange(n_hidden):
            T.set_subtensor(qual[time+1, j], max(qual[time, :].eval() *\
                HMM.A[:, j].eval()) *\
                HMM.B[j, visible_seq[time+1]])
            arg[time+1, j] = numpy.argmax(qual[time, :].eval() *\
                HMM.A[:, j].eval())
                
    path = numpy.zeros((n_time,))
    path[n_time - 1] = numpy.argmax(qual[n_time - 1, :])
    for rev_time in reversed(xrange(n_time - 1)):
        path[rev_time] = arg[rev_time + 1, path[rev_time + 1]]
    return path
        
def errors(HMM, hidden_seq, visible_seq):
    """Return 1 if y!=y_predicted (error) and 0 if right

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
                  correct label
    """
    debug_file.write('in error function')
    viterbi_path = viterbi_algo(HMM, visible_seq)

    # check if y has same dimension of y_pred
    if hidden_seq.ndim != viterbi_path.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('hidden_seq', hidden_seq.type, 'viterbi_path', viterbi_path.type)
        )

    # check if y is of the correct datatype
    if hidden_seq.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.neq(viterbi_path, hidden_seq)
    else:
        raise NotImplementedError()
            
def validation(HMM, patient_list, valid_set):
    debug_file.write('in validation')
    n_patients = len(patient_list)
    debug_file.write(str(n_patients) + 'n_patients')
    (valid_set_x, valid_set_y) = valid_set
 
    valid_set_hidden = valid_set_y.eval()
    valid_set_visible = valid_set_x.eval()

    for patient_number in xrange(n_patients):
        debug_file.write(str(patient_number) + 'patient_number')
        debug_file.write('want to get error')
        error_array=errors(HMM=HMM,
                           hidden_seq=valid_set_hidden[patient_number],
                           visible_seq=valid_set_visible[patient_number])
        debug_file.write(str(T.mean(error_array)) + 'mean error value')

def train():
    #train_data_names = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    train_data_names = ['p10a','p011','p013']
    valid_data = ['p09b','p023','p035','p038']
    test_data = ['p09a','p033']
        
    train_reader = ICHISeqDataReader(train_data_names)
    #get data divided on sequences with respect to labels
    train_set_x, train_set_y = train_reader.read_all_for_second_hmm()
        
    valid_reader = ICHISeqDataReader(valid_data)
    valid_set_x, valid_set_y = valid_reader.read_all_for_second_hmm()
    
    test_reader = ICHISeqDataReader(test_data)
    test_set_x, test_set_y = test_reader.read_all_for_second_hmm()
    
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)]
    
    debug_file.write('data is got')
    
    rank = 1
    base = pow(10, rank) + 1
    n_visible_labels = pow(base, 3)

    trained_HMM = HMM_second(n_visible=n_visible_labels,
                      train_set=(train_set_x, train_set_y),
                      train_patient_list = train_data_names)
    gc.collect()                
    debug_file.write('Hmm created')
    debug_file.write('Start validation')
    validation(HMM = trained_HMM,
               patient_list = valid_data,
               valid_set = (valid_set_x, valid_set_y))
   
if __name__ == '__main__':
    try:
        train()
    except BaseException:
        debug_file.close()