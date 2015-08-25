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
    def _init_(self, n_visible, n_hidden, input, label):
        
        self.Pi=theano.shared(
            value=numpy.zeros(
                (n_hidden,),
                dtype=theano.config.floatX
            ),
            name='Pi',
            borrow=True
        )
        
        self.label=label
        self.A=theano.shared(
            value=numpy.zeros(
                (n_hidden, n_hidden),
                dtype=theano.config.floatX
            ),
            name='A',
            borrow=True
        )
        
        self.B=theano.shared(
            value=numpy.zeros(
                (n_hidden, n_visible),
                dtype=theano.config.floatX
            ),
            name='B',
            borrow=True
        )
        
        self.input=input
        self.n_hidden=n_hidden
        self.alpha=[]
        self.betta=[]
        # gamma is matrix (time*n_hidden)
        self.gamma=[]
        self.params=[self.Pi, self.A, self.B]
        self.values=[numpy.inf, 0]
        self.bestParams=self.params
        
    def probability(self):
        return T.sum(self.alpha[-1])
        
    def get_new_params(self, visible_seq):
        max_time=len(visible_seq)
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
                for t in xrange(max_time):
                    denominator+=self.gamma[t,i]
                    if (visible_seq[t]==k):
                        numerator+=self.gamma[t,i]
                B[k,i]=numerator/denominator
        return [Pi, A, B]
        
    def train_one_patient(self, visible_seq):
        max_time=len(visible_seq)
        self.alpha[1,:]=self.Pi*self.B[visible_seq[0],:]
        for t in xrange(max_time-1):
            for j in xrange(self.n_hidden):
                self.alpha[t+1,j]=numpy.dot(self.alpha[t,:], self.A[:,j])*self.B[visible_seq[t+1],j]
                
        self.betta[max_time, :]=[1 for i in xrange(self.n_hidden)]
        for t in reversed(xrange(max_time)):
            for i in xrange(self.n_hidden):
                self.betta[t,i]=0
                for j in xrange(self.n_hidden):
                    self.betta[t,i]+=self.A[i,j]*self.B[visible_seq[t+1,j]]*self.betta[t+1,j]
                    
        for t in xrange(max_time):
            numerator=0
            for i in xrange(self.n_hidden):
                for j in xrange(self.n_hidden):
                    numerator+=self.alpha[t,i]*self.A[i,j]*self.B[visible_seq[t+1,j]*self.betta[t+1,j]]
            for i in xrange(self.n_hidden):
                for j in xrange(self.n_hidden):
                    self.ksi[t,i,j]=self.alpha[t,i]*self.A[i,j]*self.B[visible_seq[t+1,j]*self.betta[t+1,j]]/numerator

        self.gamma = T.sum(self.ksi, axis=0)
        
    def train_one_epoch(self, visible_seqs, patients):
        for patient in patients:
            self.train_one_patient(visible_seqs[patient])
            self.params=self.get_new_params(visible_seqs[patient])
        self.get_best_model(visible_seqs, patients)
        
    def get_best_model(self, visible_seqs, patients):
        prob=[]
        for patient in patients:
            self.train_one_patient(visible_seqs[patient])
            prob.append(self.probability())
            if (self.values[0]>numpy.mean(prob)):
                self.values[0]=numpy.mean(prob)
                self.bestParams=self.params
