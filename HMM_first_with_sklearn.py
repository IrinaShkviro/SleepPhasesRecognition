# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 08:25:48 2015

@author: irka
"""

import os
import sys
import timeit
import gc

import numpy

import theano
import theano.tensor as T

from sklearn import hmm

from MyVisualizer import visualize_hmm_for_one_label
from preprocess import generate_random_probabilities, generate_probabilities_for_matrix
from ichi_seq_data_reader import ICHISeqDataReader

class GeneralHMM(object):
    """
    A general HMM model is a container with several Gaussian HMMs.
    We create and fit each of them and then estimate probability of appearance
    of the observations in each model, so we predict label as label of those
    HMM which probabity is the biggest.
    """

    def __init__(
        self,
        n_hiddens,
        n_hmms=7
    ):
        #create hmm models for each label
        self.n_hmms = n_hmms
        self.hmm_models = []
        for i in xrange(n_hmms):
            self.hmm_models.append(hmm.GaussianHMM(
                n_components=n_hiddens[i]
            ))
        self.valid_error_array=[]
            
    def train(self, train_names, valid_names, window_size, rank, start_base):
        train_reader = ICHISeqDataReader(train_names)
        n_train_patients = len(train_names)
        #train hmms on data of each pattient
        for train_patient in xrange(n_train_patients):
            #get data divided on sequences with respect to labels
            train_set = train_reader.read_one_with_window(
                window_size = window_size,
                divide = True
            )
            for i in xrange(self.n_hmms):
                #get (avg, disp) labels for x-values
                x_labels = create_labels(
                    matrix = train_set[i].eval(),
                    rank=rank,
                    start_base=start_base
                )
                self.hmm_models[i].fit([numpy.array(x_labels).reshape(-1, 1)])
                        
            error_cur_epoch = self.validate_model(
                valid_names = valid_names,
                window_size = window_size,
                rank = rank,
                start_base = start_base
            )
            self.valid_error_array.append([])
            self.valid_error_array[-1].append(train_patient)
            self.valid_error_array[-1].append(error_cur_epoch)
            
            gc.collect()
            
    def validate_model(self, valid_names, window_size, rank, start_base):
        valid_reader = ICHISeqDataReader(valid_names)
        all_valid_x = []
        all_valid_y = []
        for i in xrange (len(valid_names)):
            valid_x, valid_y = valid_reader.read_one_with_window(
                window_size = window_size,
                divide = False
            )
            valid_x = create_labels(
                matrix = valid_x.eval(),
                rank=rank,
                start_base=start_base
            )
            all_valid_x = numpy.concatenate((all_valid_x, valid_x))
            all_valid_y = numpy.concatenate((all_valid_y, valid_y.eval()))
        print(len(all_valid_x), 'x')
        print(len(all_valid_y), 'y')
        #compute mean error value for patients in validation set
        error = mean_error(
            gen_hmm = self,
            obs_seq = all_valid_x,
            actual_states = all_valid_y
        )
        return error
            
    #compute label for one observation (with respect to window size)
    def define_label(self, obs):
        probabilities = []
        for i in xrange(self.n_hmms):
            probabilities.append(self.hmm_models[i].score(numpy.array([obs]).reshape((-1, 1))))
        return numpy.argmax(probabilities)
        
    def define_labels_seq(self, obs_seq):
        return [self.define_label(obs_seq[i]) for i in xrange(len(obs_seq))]
    
def mean_error(gen_hmm, obs_seq, actual_states):
    predicted_states = gen_hmm.define_labels_seq(obs_seq)
    error_array=errors(
        predicted_states=numpy.array(predicted_states),
        actual_states=numpy.array(actual_states)
    )
    return error_array.eval().mean()        
                
def errors(predicted_states, actual_states):
    """Return 1 if y!=y_predicted (error) and 0 if right

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
                  correct label
    """
    # check if y has same dimension of y_pred
    if predicted_states.ndim != actual_states.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('actual_states', actual_states.type, 'predicted_states', predicted_states.type)
        )

    # the T.neq operator returns a vector of 0s and 1s, where 1
    # represents a mistake in prediction
    return T.neq(predicted_states, actual_states)
                        
def test_hmm(gen_hmm, test_names, window_size, rank, start_base):
    test_reader = ICHISeqDataReader(test_names)
    n_test_patients = len(test_names)

    for i in xrange(n_test_patients):
        #get data divided on sequences with respect to labels
        test_x, test_y = test_reader.read_one_with_window(
            window_size = window_size,
            divide = False
        )
        test_x = create_labels(
            matrix = test_x.eval(),
            rank=rank,
            window_size = window_size,
            start_base=start_base
        )
        
        #compute mean error value for one patient in test set
        patient_error = mean_error(
            gen_hmm = gen_hmm,
            obs_seq = test_x,
            actual_states = test_y.eval()
        )
        
        print(patient_error, ' error for patient ' + str(test_names[i]))

        gc.collect()  

def train_test_model():
    """train_data_names = ['p10a','p011','p013','p014','p020','p022','p040',
                        'p045','p048','p09b','p023','p035','p038', 'p09a','p033']
    valid_data_names = ['p09b','p023','p035','p038', 'p09a','p033']
    test_data_names = ['p002']"""
    train_data_names = ['p10a', 'p011']
    valid_data_names = ['p09b']
    test_data_names = ['p002']
    
    rank = 1
    start_base=10
    window_size = 1
    hmms_count = 7
    n_hiddens=[5]*hmms_count

    gen_hmm = GeneralHMM(
        n_hiddens = n_hiddens,
        n_hmms = hmms_count
    )
    
    #training
    gen_hmm.train(
        train_names = train_data_names,
        valid_names = valid_data_names,
        window_size = window_size,
        rank = rank,
        start_base = start_base
    )
    
    gc.collect()
    print('General HMM with several Gaussian hmms created')
    print('Start testing')
    
    test_hmm(
        gen_hmm = gen_hmm,
        test_names = test_data_names,
        window_size = window_size,
        rank = rank,
        start_base = start_base
    )
    
def create_labels(matrix, rank, start_base=10):
    """
    Normalize matrix and get average and dispersion
    Matrix considers window, so we don't consider it here
    """
    #normalization
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    matrix = ((matrix-mins)*((1-(-1.))/(maxs-mins)))/2
    #get average and dispersion
    avg_disp_matrix = numpy.array([[matrix[i].mean(),
                         matrix[i].max()-
                         matrix[i].min()]
        for i in xrange(matrix.shape[0])])
    base = pow(start_base, rank) + 1
    arounded_matrix = numpy.rint(avg_disp_matrix.flatten()*pow(start_base, rank)).reshape((matrix.shape[0], 2))
    data_labels = []
    #n_in=2
    for row in arounded_matrix:
        data_labels.append(int(row[0]*base + row[1]))
    return data_labels
                   
if __name__ == '__main__':
    train_test_model()