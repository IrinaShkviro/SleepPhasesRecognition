# -*- coding: utf-8 -*-
"""
Created on Fri May 08 10:44:27 2015

@author: irka
"""
__docformat__ = 'restructedtext en'

import os
import numpy
import matplotlib.pyplot as plt

class LogisticRegression(object):
    def __init__(self):
        print "visualizer"
        
def visualize_costs(train_cost, train_error, valid_error, test_error, 
                    window_size, learning_rate,
                    train_data, valid_data, test_data):
        print "Visualizer visualize_costs"
        
        base_folder='regression_plots'
        
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        os.chdir(base_folder)
        
        output_folder=('[%s], [%s], [%s]')%(",".join(train_data), ",".join(valid_data), ",".join(test_data))
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)
        print('Set output')
                        
        train_cost=numpy.asarray(train_cost)
        train_error=numpy.asarray(train_error)
        valid_error=numpy.asarray(valid_error)
        test_error=numpy.asarray(test_error)
        print('converted to arrays')
                
        # print errors
        plt.figure(1)
        plt.plot(train_error[:, 0],train_error[:,1],label='train_error')
        plt.plot(valid_error[:, 0],valid_error[:,1],label='valid_error')
        plt.plot(test_error[:, 0],test_error[:,1],label='test_error')
        print('plots created, start decor')        
        
        # decorative part       
        plt.xlabel('epochs')
        plt.ylabel('error(%)')
        plt.title(
            ('WS: %i  LR: %f')
            % (window_size, learning_rate)
        )
        plt.legend(loc='upper left')
        plot_name = ('error LR %f WS %i.png')%(learning_rate, window_size)
        plt.savefig(plot_name, dpi=200)
        plt.close()
        print('errors visualized')
        
        # print cost
        plt.figure(2)
        plt.plot(train_cost[:, 0],train_cost[:,1],label='train_cost')

        # decorative part      
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.title(
            ('Window size: %i  Learning rate: %f')
            % (window_size, learning_rate)
        )
        plt.legend(loc='upper left')
        plot_name = ('cost LR %f WS %i.png')%(learning_rate, window_size)
        plt.savefig(plot_name, dpi=200)                    
        plt.clf()
        plt.close()
        print('cost visualized')
        
        os.chdir('../')
        os.chdir('../')

def visualize_da(train_error, valid_error, test_error, 
                 window_size, learning_rate, corruption_level, train_data, valid_data, test_data):
        print "Visualizer visualize_costs"
                        
        train_error=numpy.asarray(train_error)
        valid_error=numpy.asarray(valid_error)
        test_error=numpy.asarray(test_error)
        
        
        # print errors
        plt.figure(1)
        plt.plot(train_error[:, 0],train_error[:,1],label='train_error')
        plt.plot(valid_error[:, 0],valid_error[:,1],label='valid_error')
        plt.plot(test_error[:, 0],test_error[:,1],label='test_error')
        
        # decorative part       
        plt.xlabel('epochs')
        plt.ylabel('error(%)')
        plt.title(
            ('Window size: %i  Learning rate: %f  Corruption level %f \n \
            Train: %s  Valid: %s  Test: %s')
            % (window_size, learning_rate, corruption_level, train_data, valid_data, test_data)
        )
        plt.legend(loc='upper left') 
        plt.legend(loc='upper left')
        plot_name = ('errorLR%fWS%iCL%f.png')%(learning_rate, window_size, corruption_level)                
        plt.savefig(plot_name, dpi=200)
        plt.close()
        
def test_visualizer():
    print('test')
    train_cost = []
    for i in xrange(10):
        train_cost.append([])
        train_cost[-1].append(i)
        train_cost[-1].append(i*i)

    valid_cost = []
    for i in xrange(10):
        valid_cost.append([])
        valid_cost[-1].append(i)
        valid_cost[-1].append(i*i*i)
        
    test_cost = []
    for i in xrange(10):
        test_cost.append([])
        test_cost[-1].append(i)
        test_cost[-1].append(2*i)
        
    train_cost=numpy.asarray(train_cost)
    valid_cost=numpy.asarray(valid_cost)
    test_cost=numpy.asarray(test_cost)
       
    print(train_cost)
    
    f = open('train_array.txt', 'w')
    train_cost.tofile(f)
    f.close()    
    # print costs
    plt.figure(1)
    plt.plot(train_cost[:, 0],train_cost[:,1],label='x^2')
    plt.plot(valid_cost[:, 0],valid_cost[:,1],label='x^3')
    plt.plot(test_cost[:, 0],test_cost[:,1],label='2x')
        
    # decorative part       
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Some simple functions')
    plt.legend(loc='upper left')
         
    plt.savefig('test.png', dpi=200)               
    plt.clf()
    plt.close()
        
if __name__ == '__main__':
    test_visualizer()