"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.
Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.
Mathematically, this can be written as:
.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}
The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).
.. math::
  y_{pred} = argmax_i P(Y=i|x,W,b)
This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.
References:
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2
"""
__docformat__ = 'restructedtext en'

import os
import sys
import time
import gc

import numpy

import theano
import theano.tensor as T

from MyVisualizer import visualize_logistic
from ichi_seq_data_reader import ICHISeqDataReader
from cg import train_logistic_cg
from sgd import train_logistic_sgd

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        # start-snippet-1
        
        # initialize theta = (W,b) with 0s; W gets the shape (n_in, n_out),
        # while b is a vector of n_out elements, making theta a vector of
        # n_in*n_out + n_out elements
        self.n_in=n_in
        self.n_out=n_out
        self.theta = theano.shared(
            value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )
        # W is represented by the fisr n_in*n_out elements of theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        # b is the rest (last n_out elements)
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]
        
        self.input = input

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.flatten(T.nnet.softmax(T.dot(self.input.reshape((1, n_in)),
                                                 self.W) + self.b))

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x)
        # end-snippet-1
        
        self.train_cost_array=[]
        self.train_error_array=[]
        self.valid_error_array=[]
        self.test_error_array=[]
        self.validation_scores=[numpy.inf, 0]
        self.epoch=0
        
    def print_log_reg_types(self):
        print(self.W.type(), 'W')
        print(self.b.type(), 'b')
        print(self.p_y_given_x.type(), 'p_y_given_x')
        print(self.y_pred.type(), 'y_pred')
        

    def negative_log_likelihood(self, y):
        """Return the negative log-likelihood of the prediction
        of this model under a given target distribution.
        :type y: theano.tensor.TensorType
        :param y: corresponds to a number that gives for each example the
                  correct label
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return -T.log(self.p_y_given_x)[y]
        # end-snippet-2

    def errors(self, y):
        """Return 1 if y!=y_predicted (error) and 0 if right
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(self.y_pred, y)
        else:
            raise NotImplementedError()
            
    def predict(self):
        """
        Return predicted y
        """
        return self.y_pred
        
    def distribution(self):
        return self.p_y_given_x
        
def test_params(learning_rate,
                n_epochs,
                window_size,
                datasets,
                output_folder,
                base_folder):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model
    This is demonstrated on ICHI.
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    start_time = time.clock()

    n_in = window_size*3  # number of input units
    n_out = 7  # number of output units
    x = T.matrix('x')

    classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out)

    updated_classifier = train_logistic_sgd(learning_rate=learning_rate,
                                   window_size=window_size,
                                   n_epochs=n_epochs,
                                   datasets=datasets,
                                   classifier=classifier)
    
    '''                                 
    updated_classifier = train_logistic_cg(datasets=datasets,
                                     window_size=window_size,
                                     n_epochs=n_epochs)
    '''

    visualize_logistic(train_cost=updated_classifier.train_cost_array,
                    train_error=updated_classifier.train_error_array, 
                    valid_error=updated_classifier.valid_error_array,
                    test_error=updated_classifier.test_error_array,
                    window_size=window_size,
                    learning_rate=learning_rate,
                    output_folder=output_folder,
                    base_folder=base_folder)
    
    end_time = time.clock()
    print(
        (
           'Optimization complete with best validation score of %f %%,'
           'with test performance %f %%'
        )
        % (updated_classifier.validation_scores[0],
           updated_classifier.validation_scores[1])
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        updated_classifier.epoch, 1. * updated_classifier.epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                         os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))
    
def test_all_params():
    learning_rates = [0.0001]
    window_sizes = [13]
    
    train_data = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    valid_data = ['p09b','p023','p035','p038']
    test_data = ['p09a','p033']
    
    train_reader = ICHISeqDataReader(train_data)
    train_set_x, train_set_y = train_reader.read_all()
    
    valid_reader = ICHISeqDataReader(valid_data)
    valid_set_x, valid_set_y = valid_reader.read_all()

    test_reader = ICHISeqDataReader(test_data)
    test_set_x, test_set_y = test_reader.read_all()
    
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
            
    output_folder=('[%s], [%s], [%s]')%(",".join(train_data), ",".join(valid_data), ",".join(test_data))

    for lr in learning_rates:
        for ws in window_sizes:
            test_params(learning_rate=lr,
                        n_epochs=1000,
                        window_size = ws,
                        datasets=datasets,
                        output_folder=output_folder,
                        base_folder='regression_plots')

if __name__ == '__main__':
    test_all_params()