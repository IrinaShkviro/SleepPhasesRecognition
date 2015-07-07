"""
This tutorial introduces logistic regression using Theano and conjugate
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


This tutorial presents a conjugate gradient optimization method that is
suitable for smaller datasets.


References:

   - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2


"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from MyVisualizer import visualize_costs_cg
from ichi_seq_data_reader import ICHISeqDataReader

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
                      architecture ( one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoint lies

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the target lies

        """

        # initialize theta = (W,b) with 0s; W gets the shape (n_in, n_out),
        # while b is a vector of n_out elements, making theta a vector of
        # n_in*n_out + n_out elements
        self.theta = theano.shared(
            value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )

        # keep track of model input
        self.input = input.reshape((1, n_in))

        # W is represented by the fisr n_in*n_out elements of theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        # b is the rest (last n_out elements)
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.flatten(T.nnet.softmax(T.dot(self.input, self.W) + self.b))

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x)

    def negative_log_likelihood(self, y):
        """Return the negative log-likelihood of the prediction of this model
        under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        return -T.log(self.p_y_given_x)[y]
       
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                  the correct label
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


def test_params(datasets, output_folder, window_size, n_epochs=50):
    """Demonstrate conjugate gradient optimization of a log-linear model

    This is demonstrated on ICHI.

    :type n_epochs: int
    :param n_epochs: number of epochs to run the optimizer

    """
    #############
    # LOAD DATA #
    #############
    
    # split the datasets
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]

        # compute number of examples given in datasets
    n_train_samples =  train_set_x.get_value(borrow=True).shape[0] - window_size + 1    
    n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1    
    n_test_samples = test_set_x.get_value(borrow=True).shape[0] - window_size + 1
    
    n_in = window_size*3  # number of input units
    n_out = 7  # number of output units

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  

    # generate symbolic variables for input
    x = T.matrix('x')  # data, presented as window with x, y, x for each sample
    y = T.iscalar('y')  # labels, presented as int label

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compile a theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: test_set_x[index:index + window_size],
            y: test_set_y[index + window_size - 1]
        },
        name="test"
    )

    validate_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: valid_set_x[index: index + window_size],
            y: valid_set_y[index + window_size - 1]
        },
        name="validate"
    )

    #  compile a theano function that returns the cost
    conj_cost = theano.function(
        inputs=[index],
        outputs=[cost, classifier.errors(y), classifier.predict(), y],
        givens={
            x: train_set_x[index: index + window_size],
            y: train_set_y[index + window_size - 1]
        },
        name="conj_cost"
    )

    # compile a theano function that returns the gradient with respect to theta
    conj_grad = theano.function(
        [index],
        T.grad(cost, classifier.theta),
        givens={
            x: train_set_x[index: index + window_size],
            y: train_set_y[index + window_size - 1]
        },
        name="conj_grad"
    )
    
    train_cost_array = []
    train_error_array = []
    train_confusion_matrix = numpy.zeros((7, 7))
    iter = 0

    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        cur_train_cost = []
        cur_train_error =[]
        for i in xrange(n_train_samples):
            sample_cost, sample_error, cur_pred, cur_actual = conj_cost(i)
            cur_train_cost.append(sample_cost)
            cur_train_error.append(sample_error)
            train_confusion_matrix[cur_actual][cur_pred] += 1
        
        this_train_loss = float(numpy.mean(cur_train_cost))  
        train_cost_array.append([])
        train_cost_array[-1].append(iter)
        train_cost_array[-1].append(this_train_loss)
       
        train_error_array.append([])
        train_error_array[-1].append(iter)
        train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
                
        iter += 1
        
        return this_train_loss

    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        grad = conj_grad(0)
        for i in xrange(1, n_train_samples):
            grad += conj_grad(i)
        return grad / n_train_samples

    validation_scores = [numpy.inf, 0]
    valid_error_array = []
    test_error_array = []

    # creates the validation function
    def callback(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        #compute the validation loss
        validation_losses = [validate_model(i)
                             for i in xrange(n_valid_samples)]
        this_validation_loss = numpy.mean(validation_losses) * 100.,
        print('validation error %f %%' % (this_validation_loss))
        valid_error_array.append([])
        valid_error_array[-1].append(iter)
        valid_error_array[-1].append(this_validation_loss)

        # check if it is better then best validation score got until now
        if this_validation_loss < validation_scores[0]:
            # if so, replace the old one, and compute the score on the
            # testing dataset
            validation_scores[0] = this_validation_loss
            test_losses = [test_model(i)
                           for i in xrange(n_test_samples)]
            validation_scores[1] = numpy.mean(test_losses)
            test_error_array.append([])
            test_error_array[-1].append(iter)
            test_error_array[-1].append(validation_scores[1])

    ###############
    # TRAIN MODEL #
    ###############

    # using scipy conjugate gradient optimizer
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = timeit.default_timer()
    best_theta = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((n_in + 1) * n_out, dtype=x.dtype),
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=n_epochs
    )
    visualize_costs_cg(train_cost_array, train_error_array, 
                    valid_error_array, test_error_array,
                    window_size, output_folder)
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, with '
            'test performance %f %%'
        )
        % (validation_scores[0] * 100., validation_scores[1] * 100.)
    )

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


def test_all_params():
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
    
    for ws in window_sizes:
        test_params(datasets=datasets, output_folder=output_folder,
                    window_size = ws, n_epochs=1000)

if __name__ == '__main__':
    test_all_params()
