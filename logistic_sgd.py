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
        
        self.input = input.reshape((1, n_in))

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.flatten(T.nnet.softmax(T.dot(self.input,
                                                 self.W) + self.b))

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
        
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
        
def zero_in_array(array):
    return [[0 for col in range(7)] for row in range(7)]
    
def train_sgd(learning_rate, window_size, n_epochs,
              datasets, classifier=None):
                  
    # split the datasets
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]

    # compute number of examples given in datasets
    n_train_samples =  train_set_x.get_value(borrow=True).shape[0] - window_size + 1    
    n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1    
    n_test_samples = test_set_x.get_value(borrow=True).shape[0] - window_size + 1

    # allocate symbolic variables for the data
    index = T.lscalar()

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as window with x, y, x for each sample
    y = T.iscalar('y')  # labels, presented as int label

    if classifier is None:
        # construct the logistic regression class
        # Each ICHI input has size window_size*3
        classifier = LogisticRegression(input=x, n_in=window_size*3, n_out=7)

    cost = classifier.negative_log_likelihood(y)
    # compute the gradient of cost with respect to theta = (W,b)
    g_theta = T.grad(cost=cost, wrt=classifier.theta)
    
    predict = classifier.predict()
    
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a row
    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), predict, y],
        givens={
            x: test_set_x[index: index + window_size],
            y: test_set_y[index + window_size - 1]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), predict, y],
        givens={
            x: valid_set_x[index: index + window_size],
            y: valid_set_y[index + window_size - 1]
        }
    )

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    #updates = [(classifier.theta, T.set_subtensor(classifier.W, classifier.W - learning_rate * g_W)),
    #           (classifier.theta, T.set_subtensor(classifier.b, classifier.b - learning_rate * g_b))]

    updates = [(classifier.theta, classifier.theta - learning_rate * g_theta)]
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=[cost, classifier.errors(y), predict, y],
        updates=updates,
        givens={
            x: train_set_x[index: index + window_size],
            y: train_set_y[index + window_size - 1]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = n_train_samples*2  # look as this many examples regardless
    patience_increase = 25  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = patience / 4

    done_looping = False

    iter = 0
    cur_train_cost =[]
    cur_train_error = []
    train_confusion_matrix = numpy.zeros((7, 7))
    valid_confusion_matrix = numpy.zeros((7, 7))
    print(n_train_samples, 'train_samples')
    
    while (classifier.epoch < n_epochs) and (not done_looping):
        train_confusion_matrix = zero_in_array(train_confusion_matrix)
        for index in xrange(n_train_samples):            
            sample_cost, sample_error, cur_pred, cur_actual = train_model(index)
            # iteration number
            iter = classifier.epoch * n_train_samples + index
                
            cur_train_cost.append(sample_cost)
            cur_train_error.append(sample_error)
            train_confusion_matrix[cur_actual][cur_pred] += 1
        
            if (iter + 1) % validation_frequency == 0:
                valid_confusion_matrix = zero_in_array(valid_confusion_matrix)
                # compute zero-one loss on validation set
                validation_losses = []
                for i in xrange(n_valid_samples):
                    validation_loss, cur_pred, cur_actual = validate_model(i)
                    validation_losses.append(validation_loss)
                    valid_confusion_matrix[cur_actual][cur_pred] += 1
    
                this_validation_loss = float(numpy.mean(validation_losses))*100                 
                classifier.valid_error_array.append([])
                classifier.valid_error_array[-1].append(float(iter)/n_train_samples)
                classifier.valid_error_array[-1].append(this_validation_loss)
                        
                print(
                    'epoch %i, iter %i/%i, validation error %f %%' %
                    (
                        classifier.epoch,
                        index + 1,
                        n_train_samples,
                        this_validation_loss
                    )
                )
       
                # if we got the best validation score until now
                if this_validation_loss < classifier.validation_scores[0]:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < classifier.validation_scores[0] *  \
                        improvement_threshold:
                        patience = max(patience, iter * patience_increase)
        
                    classifier.validation_scores[0] = this_validation_loss
                    # test it on the test set
                         
                    test_result = [test_model(i)
                                   for i in xrange(n_test_samples)]
                    test_result = numpy.asarray(test_result)
                    test_losses = test_result[:,0]
                    test_score = float(numpy.mean(test_losses))*100
                    classifier.validation_scores[1] = test_score
                            
                    classifier.test_error_array.append([])
                    classifier.test_error_array[-1].append(float(iter)/n_train_samples)
                    classifier.test_error_array[-1].append(test_score)
        
                    print(
                        (
                            '     epoch %i, iter %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            classifier.epoch,
                            index + 1,
                            n_train_samples,
                            test_score
                        )
                    )
            if patience*4 <= iter:
                done_looping = True
                print('Done looping')
                break
                           
        classifier.train_cost_array.append([])
        classifier.train_cost_array[-1].append(float(iter)/n_train_samples)
        classifier.train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
        cur_train_cost =[]
       
        classifier.train_error_array.append([])
        classifier.train_error_array[-1].append(float(iter)/n_train_samples)
        classifier.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
        cur_train_error =[]
                
        classifier.epoch = classifier.epoch + 1
        gc.collect()
            
    test_confusion_matrix = zero_in_array(numpy.zeros((7, 7)))
    test_losses = []
    for i in xrange(n_test_samples):
        test_loss, cur_pred, cur_actual = test_model(i)
        test_losses.append(test_loss)
        test_confusion_matrix[cur_actual][cur_pred] += 1
    
    test_score = numpy.mean(test_losses)*100
    classifier.test_error_array.append([])
    classifier.test_error_array[-1].append(float(iter)/n_train_samples)
    classifier.test_error_array[-1].append(test_score)
    
    print(train_confusion_matrix, 'train_confusion_matrix')
    print(valid_confusion_matrix, 'valid_confusion_matrix')
    print(test_confusion_matrix, 'test_confusion_matrix')

    return classifier
    
def train_cg(datasets, window_size, n_epochs, classifier=None):
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

    if classifier is None:
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
    
    train_confusion_matrix = numpy.zeros((7, 7))

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
        classifier.train_cost_array.append([])
        classifier.train_cost_array[-1].append(classifier.epoch)
        classifier.train_cost_array[-1].append(this_train_loss)
       
        classifier.train_error_array.append([])
        classifier.train_error_array[-1].append(classifier.epoch)
        classifier.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
                
        classifier.epoch += 1
        
        return this_train_loss

    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        grad = conj_grad(0)
        for i in xrange(1, n_train_samples):
            grad += conj_grad(i)
        return grad / n_train_samples

    # creates the validation function
    def callback(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        #compute the validation loss
        validation_losses = [validate_model(i)
                             for i in xrange(n_valid_samples)]
        this_validation_loss = float(numpy.mean(validation_losses) * 100.,)
        print('validation error %f %%' % (this_validation_loss))
        classifier.valid_error_array.append([])
        classifier.valid_error_array[-1].append(classifier.epoch)
        classifier.valid_error_array[-1].append(this_validation_loss)

        # check if it is better then best validation score got until now
        if this_validation_loss < classifier.validation_scores[0]:
            # if so, replace the old one, and compute the score on the
            # testing dataset
            classifier.validation_scores[0] = this_validation_loss
            test_losses = [test_model(i)
                           for i in xrange(n_test_samples)]
            classifier.validation_scores[1] = float(numpy.mean(test_losses))
            classifier.test_error_array.append([])
            classifier.test_error_array[-1].append(classifier.epoch)
            classifier.test_error_array[-1].append(classifier.validation_scores[1])

    ###############
    # TRAIN MODEL #
    ###############

    # using scipy conjugate gradient optimizer
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    best_theta = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((n_in + 1) * n_out, dtype=x.dtype),
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=n_epochs
    )
    return classifier

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

    
    updated_classifier = train_sgd(learning_rate=learning_rate,
                                   window_size=window_size,
                                   n_epochs=n_epochs,
                                   datasets=datasets)
    
                                     
    updated_classifier = train_cg(datasets=datasets,
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
    '''
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