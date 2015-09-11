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

def change_data_for_one_patient(hiddens_patient, visibles_patient,
                                window_size, base_for_labels):
    n_patient_samples = len(hiddens_patient)
    half_window_size = int(window_size/2)
    new_hidden=hiddens_patient[half_window_size:n_patient_samples-half_window_size]
    new_visible=[]
    for index in xrange(n_patient_samples-window_size):
        sample=index+half_window_size
        #create labels with respect to window size
        visible_label=visibles_patient[sample]*\
            pow(base_for_labels, half_window_size)                    
        for i in xrange(half_window_size):
            visible_label += visibles_patient[sample-i-1]*\
                pow(base_for_labels, window_size+i-half_window_size)
            visible_label += visibles_patient[sample+i+1]*\
                pow(base_for_labels, half_window_size-i-1) 
        new_visible.append(visible_label)                       
    last_visible_label=visibles_patient[n_patient_samples-half_window_size-1]*\
        pow(base_for_labels, half_window_size)                    
    for i in xrange(half_window_size):
        last_visible_label += visibles_patient[n_patient_samples-half_window_size-2-i]*\
            pow(base_for_labels, half_window_size+i+1)
        last_visible_label += visibles_patient[n_patient_samples-half_window_size+i]*\
            pow(base_for_labels, half_window_size-i-1)
    new_visible.append(last_visible_label)
    return (new_visible, new_hidden)    

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
    return T.mean(error_array).eval()
                        
def train_separately():
    train_data_names = ['p10a','p011','p013','p014','p020','p022','p040',
                        'p045','p048','p09b','p023','p035','p038', 'p09a','p033']
    valid_data = ['p09b','p023','p035','p038', 'p09a','p033']

    n_train_patients=len(train_data_names)
    n_valid_patients=len(valid_data)
    
    rank = 1
    start_base=5
    base = pow(start_base, rank) + 1
    n_visible_labels = pow(base, 3)
    window_size = 1
    n_visible=pow(n_visible_labels, window_size)
    n_hidden=7
        
    train_reader = ICHISeqDataReader(train_data_names)
    valid_reader = ICHISeqDataReader(valid_data)
    
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden)) 
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))

    for train_patient in xrange(n_train_patients):
        #get data divided on sequences with respect to labels
        train_set_x, train_set_y = train_reader.read_doc_for_second_hmm(
            rank=rank,
            start_base=start_base
        )
        
        new_train_visible, new_train_hidden = change_data_for_one_patient(
            hiddens_patient=train_set_y.eval(),
            visibles_patient=train_set_x.eval(),
            window_size=window_size,
            base_for_labels=n_visible_labels
        )
        
        pi_values, a_values, b_values, array_from_hidden = update_params_on_patient(
            pi_values=pi_values,
            a_values=a_values,
            b_values=b_values,
            array_from_hidden=array_from_hidden,
            hiddens_patient=new_train_hidden,
            visibles_patient=new_train_visible,
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
        valid_set_x, valid_set_y = valid_reader.read_doc_for_second_hmm(
            rank=rank,
            start_base=start_base
        )
        
        new_valid_visible, new_valid_hidden = change_data_for_one_patient(
            hiddens_patient=valid_set_y.eval(),
            visibles_patient=valid_set_x.eval(),
            window_size=window_size,
            base_for_labels=n_visible_labels
        )
        
        patient_error = get_error_on_patient(
            model=hmm_model,
            visible_set=new_valid_visible,
            hidden_set=new_valid_hidden,
            algo=algo
        )
        
        print(patient_error, ' error for patient ' + str(valid_patient))
        gc.collect()  
                   
if __name__ == '__main__':
    train_separately()



def change_data_for_ws(dataset, window_size, base_for_labels, n_patients):
    (set_x, set_y) = dataset
    
    set_hidden = set_y.eval()
    set_visible = set_x.eval()
    
    new_hidden=[]
    new_visible=[]
    
    for patient_number in xrange(n_patients):
        visibles_patient, hiddens_patient = change_data_for_one_patient(
            hiddens_patient=set_hidden[patient_number],
            visibles_patient=set_visible[patient_number],
            window_size=window_size,
            base_for_labels=base_for_labels
        )
        new_hidden.append(hiddens_patient)
        new_visible.append(visibles_patient)
                    
    return (new_visible, new_hidden)

def create_hmm_for_all_data(n_hidden, n_visible, train_set, n_patients, window_size=1):
    # split the datasets
    (train_set_visible, train_set_hidden) = train_set
        
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden)) 
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))
        
    for patient_number in xrange(n_patients):
        pi_values, a_values, b_values, array_from_hidden = update_params_on_patient(
            pi_values=pi_values,
            a_values=a_values,
            b_values=b_values,
            array_from_hidden=array_from_hidden,
            hiddens_patient=train_set_hidden[patient_number],
            visibles_patient=train_set_visible[patient_number],
            n_hidden=n_hidden
        )
            
    pi_values, a_values, b_values = finish_training(
        pi_values=pi_values,
        a_values=a_values,
        b_values=b_values,
        array_from_hidden=array_from_hidden,
        n_hidden=n_hidden,
        n_patients=n_patients
    )
        
    model = hmm.MultinomialHMM(
        n_components=n_hidden,
        startprob=pi_values,
        transmat=a_values
    )
    model.n_symbols=n_visible
    model.emissionprob_=b_values 
         
    print('MultinomialHMM created')
    return model   

def get_error_on_model(model, n_patients, test_set, window_size=1):
    print('in validate_model \n')
    print(str(n_patients) + ' n_patients \n')
    
    (set_visible, set_hidden) = test_set
    
    for patient in xrange(n_patients):
        print(str(patient) + ' patient number \n')
        patient_error = get_error_on_patient(
            model=model,
            visible_set=set_visible[patient],
            hidden_set=set_hidden[patient]
        )
        print(str(patient_error) + ' mean error value \n')   
        
def train_all_data():
    #train_data_names = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    train_data_names = ['p10a']
    valid_data = ['p09b','p023','p035','p038']
    #valid_data=['p10a']
    test_data = ['p09a','p033']

    n_train_patients=len(train_data_names)
    n_valid_patients=len(valid_data)
    n_test_patients=len(test_data)
    
    rank = 1
    start_base=5
    base = pow(start_base, rank) + 1
        
    train_reader = ICHISeqDataReader(train_data_names)
    #get data divided on sequences with respect to labels
    train_set_x, train_set_y = train_reader.read_all_for_second_hmm(
        rank=rank,
        start_base=start_base
    )
            
    valid_reader = ICHISeqDataReader(valid_data)
    valid_set_x, valid_set_y = valid_reader.read_all_for_second_hmm(
        rank=rank,
        start_base=start_base
    )
    
    test_reader = ICHISeqDataReader(test_data)
    test_set_x, test_set_y = test_reader.read_all_for_second_hmm(
        rank=rank,
        start_base=start_base
    )
    
    print('data is got')
    
    n_visible_labels = pow(base, 3)
    n_hidden=7
    window_size = 1
    
    new_train_set_x, new_train_set_y = change_data_for_ws(
            dataset = (train_set_x, train_set_y),
            window_size=window_size,
            base_for_labels=n_visible_labels,
            n_patients=n_train_patients
    )
        
    new_valid_set_x, new_valid_set_y = change_data_for_ws(
            dataset = (valid_set_x, valid_set_y),
            window_size=window_size,
            base_for_labels=n_visible_labels,
            n_patients=n_valid_patients
    )
        
    new_test_set_x, new_test_set_y = change_data_for_ws(
            dataset = (test_set_x, test_set_y),
            window_size=window_size,
            base_for_labels=n_visible_labels,
            n_patients=n_test_patients
    )
        
    trained_HMM = create_hmm_for_all_data(
            n_hidden=n_hidden,
            n_visible=pow(n_visible_labels, window_size),
            train_set=(new_train_set_x, new_train_set_y),
            n_patients=n_train_patients,
            window_size=window_size
    )
            
    gc.collect()  
    print('Hmm created')
    get_error_on_model(model = trained_HMM,
                   n_patients = n_valid_patients,
                   test_set = (new_valid_set_x, new_valid_set_y),
                   window_size=1)