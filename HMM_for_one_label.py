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

class HMM_for_one_label(object):
    def __init__(self, n_visible, n_hidden, input, n_epoch, patient_list):
        
        #Pi is matrix which consider probabilities of each hidden state 
        #in start time
        self.Pi=theano.shared(
            value=numpy.zeros(
                (n_hidden,),
                dtype=theano.config.floatX
            ),
            name='Pi',
            borrow=True
        )
        
        self.label=input[1]
        self.A=theano.shared(
            value=numpy.zeros(
                (n_hidden, n_hidden),
                dtype=theano.config.floatX
            ),
            name='A',
            borrow=True
        )
        
        #B is matrix which consider probabilities observe visible element 
        #from hidden state
        self.B=theano.shared(
            value=numpy.zeros(
                (n_hidden, n_visible),
                dtype=theano.config.floatX
            ),
            name='B',
            borrow=True
        )
        
        self.visible_seq=input[0]
        self.max_time = len(self.visible_seq)
        print(self.visible_seq, 'input')
        self.n_visible = n_visible
        self.n_hidden=n_hidden
        self.epochs=n_epoch
        self.patient_list=patient_list
        
        #alpha[t][j] is probability observe visible sequence visible_seq[0:t]
        # and condition with number j in time t
        self.alpha = theano.shared(
            value=numpy.zeros(
                (self.max_time, self.n_hidden),
                dtype=theano.config.floatX
            ),
            name='alpha',
            borrow=True
        )
        
        #betta[t][j] is probability observe visible sequence visible_seq[t+1:max_time]
        # and condition with number j in time t
        self.betta = theano.shared(
            value=numpy.zeros(
                (self.max_time, self.n_hidden),
                dtype=theano.config.floatX
            ),
            name='betta',
            borrow=True
        )
        
        # gamma is matrix (time*n_hidden)
        self.gamma=theano.shared(
            value=numpy.zeros(
                (self.max_time, self.n_hidden),
                dtype=theano.config.floatX
            ),
            name='gamma',
            borrow=True
        )
        
        self.params=[self.Pi, self.A, self.B]
        
        #train_probabilty is mean value of probability of appearance train data
        #   in this HMM
        self.train_probability = 0
        self.bestParams=self.params
        
    def probability_for_cur_model(self):
        return T.sum(self.alpha[-1])
     
    def train(self):
        for epoch in xrange(self.epochs):
            self.train_one_epoch()
            
    def train_one_epoch(self):
        """
        for patient in xrange(len(visible_seqs)):
            self.update_params_for_one_patient(visible_seqs[patient])
            self.get_new_params(visible_seqs[patient])
        """
        self.update_params_for_one_patient()
        self.get_new_params()
        self.get_best_model()
        
    def update_params_for_one_patient(self):
        #generate probabilities observe visible_seq[0] in initial time
        for j in xrange(self.n_hidden):
            print(self.visible_seq[0], 'self.visible_seq[0]')
            print(self.Pi[j], 'self.Pi[j]')
            print(self.B[j, self.visible_seq[0]], 'self.B[j, self.visible_seq[0]]')
            print(self.alpha[0, j], 'self.alpha[0, j]')
            T.set_subtensor(self.alpha[0, j], self.Pi[j] * self.B[j, self.visible_seq[0]])
            
        for t in xrange(self.max_time-1):
            for j in xrange(self.n_hidden):
                T.set_subtensor(self.alpha[t+1,j],
                                T.dot(self.alpha[t,:], self.A[:,j])*\
                                self.B[j, self.visible_seq[t+1]])
              
        T.set_subtensor(self.betta[self.max_time - 1], [1] * self.n_hidden)
        for t in reversed(xrange(self.max_time - 1)):
            for i in xrange(self.n_hidden):
                cur_value = 0
                for j in xrange(self.n_hidden):
                    cur_value += self.A[i,j]*self.B[j,self.visible_seq[t+1]]*self.betta[t+1,j]
                T.set_subtensor(self.betta[t,i], cur_value)
                
        for t in xrange(self.max_time):
            numerator=0
            for i in xrange(self.n_hidden):
                for j in xrange(self.n_hidden):
                    numerator+=self.alpha[t,i]*self.A[i,j]*self.B[j, self.visible_seq[t]]*self.betta[t+1,j]
            for i in xrange(self.n_hidden):
                for j in xrange(self.n_hidden):
                    T.set_subtensor(self.ksi[t,i,j], self.alpha[t,i]*self.A[i,j]*self.B[j, self.visible_seq[t]]*self.betta[t+1,j]/numerator)

        T.set_subtensor(self.gamma, T.sum(self.ksi, axis=0))
                
    def get_new_params(self):
        #new Pi is first row in gamma
        Pi=[self.gamma[0]]
        A=[]
        B=[]
        
        for i in xrange(self.n_hidden):
            for j in xrange(self.n_hidden):
                A[i,j]=T.sum(self.ksi[:,i,j])/T.sum(self.gamma[:,i])
                
        for i in xrange(self.n_hidden):
            for k in xrange(self.n_visible):
                numerator=0
                denominator=0
                for t in xrange(self.max_time):
                    denominator+=self.gamma[t,i]
                    if (self.visible_seq[t]==k):
                        numerator+=self.gamma[t,i]
                B[i,k]=numerator/denominator
        
        self.Pi = Pi
        self.A = A
        self.B = B
        self.params = [Pi, A, B]
        
    def get_best_model(self):
        """
        prob=[]
        for patient in xrange(len(visible_seqs)):
            self.update_params_for_one_patient(visible_seqs[patient])
            prob.append(self.probability())
        if (self.train_probabilty<numpy.mean(prob)):
            self.train_probabilty=numpy.mean(prob)
            self.bestParams=self.params
        """
        train_probability = self.probability_for_cur_model()
        if (self.train_probabilty<train_probability):
            self.train_probabilty=train_probability
            self.bestParams=self.params
        self.Pi = self.bestParams[0]
        self.A = self.bestParams[1]
        self.B = self.bestParams[2]
                
