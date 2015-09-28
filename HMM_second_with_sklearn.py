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

from sklearn import hmm

from MyVisualizer import visualize_hmm_for_one_label
from preprocess import generate_random_probabilities, generate_probabilities_for_matrix
from ichi_seq_data_reader import ICHISeqDataReader

def update_params_on_patient(pi_values, a_values, b_values, array_from_hidden,
                             hiddens_patient, visibles_patient, n_hidden):
    pi_values[hiddens_patient[0]] += 1
    n_patient_samples = len(hiddens_patient)
    for index in xrange(n_patient_samples-1):
        a_values[hiddens_patient[index], hiddens_patient[index+1]] += 1
        b_values[hiddens_patient[index], visibles_patient[index]] += 1            
    b_values[hiddens_patient[n_patient_samples-1],
             visibles_patient[n_patient_samples-1]] += 1
    for hidden in xrange(n_hidden):
        array_from_hidden[hidden] += len(hiddens_patient[numpy.where(hiddens_patient==hidden)])
    array_from_hidden[hiddens_patient[n_patient_samples-1]] -= 1
    return (pi_values, a_values, b_values, array_from_hidden)

def finish_training(pi_values, a_values, b_values, array_from_hidden, n_hidden,
                    n_patients):
    for hidden in xrange(n_hidden):
        a_values[hidden] = a_values[hidden]/array_from_hidden[hidden]
        b_values[hidden] = b_values[hidden]/array_from_hidden[hidden]
    pi_values = pi_values/float(n_patients)
    return (pi_values, a_values, b_values)
        
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

def get_error_on_patient(model, visible_set, hidden_set, algo):
    predicted_states = model.predict(
            obs=visible_set,
            algorithm=algo
    )
    error_array=errors(predicted_states=predicted_states,
                       actual_states=hidden_set)
    return error_array.eval().mean()
                        
def train_separately():
    train_data_names = ['p10a','p011','p013','p014','p020','p022','p040',
                        'p045','p048','p09b','p023','p035','p038', 'p09a','p033']
    valid_data = ['p09b','p023','p035','p038', 'p09a','p033']

    n_train_patients=len(train_data_names)
    n_valid_patients=len(valid_data)
    
    rank = 1
    start_base=10
    base = pow(start_base, rank) + 1
    window_size = 1
    n_visible=pow(base, 6)
    n_hidden=7
        
    train_reader = ICHISeqDataReader(train_data_names)
    valid_reader = ICHISeqDataReader(valid_data)
    
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden)) 
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))

    for train_patient in xrange(n_train_patients):
        #get data divided on sequences with respect to labels
        train_set_x, train_set_y = train_reader.read_doc_with_av_disp(
            rank=rank,
            start_base=start_base,
            window_size=window_size
        )
        
        pi_values, a_values, b_values, array_from_hidden = update_params_on_patient(
            pi_values=pi_values,
            a_values=a_values,
            b_values=b_values,
            array_from_hidden=array_from_hidden,
            hiddens_patient=train_set_y.eval(),
            visibles_patient=train_set_x.eval(),
            n_hidden=n_hidden
        )
        
        gc.collect()
        
    pi_values, a_values, b_values = finish_training(
        pi_values=pi_values,
        a_values=a_values,
        b_values=b_values,
        array_from_hidden=array_from_hidden,
        n_hidden=n_hidden,
        n_patients=n_train_patients
    )
    
    hmm_model = hmm.MultinomialHMM(
        n_components=n_hidden,
        startprob=pi_values,
        transmat=a_values
    )
    hmm_model.n_symbols=n_visible
    hmm_model.emissionprob_=b_values 
    gc.collect()
    print('MultinomialHMM created')
    algo='viterbi'

    for valid_patient in xrange(n_valid_patients):
        #get data divided on sequences with respect to labels
        valid_set_x, valid_set_y = valid_reader.read_doc_with_av_disp(
            rank=rank,
            start_base=start_base,
            window_size=window_size
        )
        
        patient_error = get_error_on_patient(
            model=hmm_model,
            visible_set=valid_set_x.eval(),
            hidden_set=valid_set_y.eval(),
            algo=algo
        )
        
        print(patient_error, ' error for patient ' + str(valid_patient))

        gc.collect()  
                   
if __name__ == '__main__':
    train_separately()