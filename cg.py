import os
import sys
import timeit
import gc

import numpy

import theano
import theano.tensor as T
import scipy.optimize

def train_logistic_cg(datasets, window_size, n_epochs, classifier):
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
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  

    # generate symbolic variables for input
    x = classifier.input  # data, presented as window with x, y, x for each sample
    y = T.iscalar('y')  # labels, presented as int label


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
    print ("Optimizing using scipy.optimize.fmin_cg...")
    best_theta = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((classifier.n_in + 1) * classifier.n_out, dtype=x.dtype),
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=n_epochs
    )
    return classifier
    
def train_da_cg(da, train_set, window_size, corruption_level, training_epochs):

    n_train_samples = train_set.get_value(borrow=True).shape[0] - window_size + 1
    
     # allocate symbolic variables for the data
    index = T.lscalar()    # index
    x = da.input  # the data is presented as 3D vector

    cost = da.get_cost(
        corruption_level=corruption_level
    )
           
    #  compile a theano function that returns the cost
    conj_cost = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set[index: index + window_size]
        },
        name="conj_cost"
    )

    # compile a theano function that returns the gradient with respect to theta
    conj_grad = theano.function(
        [index],
        T.grad(cost, da.theta),
        givens={
            x: train_set[index: index + window_size]
        },
        name="conj_grad"
    )
    
    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        da.theta.set_value(theta_value, borrow=True)
        train_losses = [conj_cost(i)
                        for i in xrange(n_train_samples)]
                            
        this_train_loss = float(numpy.mean(train_losses))  
        da.train_cost_array.append([])
        da.train_cost_array[-1].append(da.epoch)
        da.train_cost_array[-1].append(this_train_loss)
        da.epoch += 1
        return this_train_loss
        
    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(theta_value):
        da.theta.set_value(theta_value, borrow=True)
        grad = conj_grad(0)
        for i in xrange(1, n_train_samples):
            grad += conj_grad(i)
        return grad / n_train_samples

    ############
    # TRAINING #
    ############

    # using scipy conjugate gradient optimizer
    print ("Optimizing using scipy.optimize.fmin_cg...")
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((da.n_visible + 1) * da.n_hidden, dtype=x.dtype),
        fprime=train_fn_grad,
        disp=0,
        maxiter=training_epochs
    )
    return da
    
def pretraining_functions_sda_cg(sda, train_set_x, window_size, corruption_levels):
    ''' Generates a list of functions, each of them implementing one
    step in trainnig the dA corresponding to the layer with same index.
    The function will require as input the index, and to train
    a dA you just need to iterate, calling the corresponding function on
    all indexes.
    :type train_set_x: theano.tensor.TensorType
    :param train_set_x: Shared variable that contains all datapoints used        for training the dA
    :type window_size: int
    :param window_size: size of a window
    '''

    # index
    index = T.lscalar('index')
    theta_value = T.vector('theta')
#    corruption_level = T.scalar('corruption')  # % of corruption to use
    n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
        
    # creates a function that computes the average cost on the training set
    def train_fn_vis(da_index, conj_cost):
        train_losses = [conj_cost(i) for i in xrange(n_train_samples)]
                                            
        this_train_loss = float(numpy.mean(train_losses))  
        sda.dA_layers[da_index].train_cost_array.append([])
        sda.dA_layers[da_index].train_cost_array[-1].append(sda.dA_layers[da_index].epoch)
        sda.dA_layers[da_index].train_cost_array[-1].append(this_train_loss)
        sda.dA_layers[da_index].epoch += 1
        return theano.shared(this_train_loss)
            
    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad_vis(conj_grad):
        grad = conj_grad(0)
        for i in xrange(1, n_train_samples):
            grad += conj_grad(i)
        return theano.shared(grad / n_train_samples)

    pretrain_fns = []
    pretrain_updates = []
    for da_index in xrange(sda.n_layers):
        cur_dA=sda.dA_layers[da_index]
        # get the cost and the updates list
        cost = cur_dA.get_cost(corruption_levels[da_index])
        
        # compile a theano function that returns the cost
        sample_cost = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                sda.x: train_set_x[index: index + window_size]
            },
            on_unused_input='warn'
        )
        
        # compile a theano function that returns the gradient with respect to theta
        sample_grad = theano.function(
            inputs=[index],
            outputs=T.grad(cost, cur_dA.theta),
            givens={
                sda.x: train_set_x[index: index + window_size]
            },
            on_unused_input='warn'
        )
        
        def train_fn(theta_value):
            sda.dA_layers[da_index].theta.set_value(theta_value, borrow=True)
            train_losses = [sample_cost(i)
                            for i in xrange(n_train_samples)]
            this_train_loss = float(numpy.mean(train_losses))  
            sda.dA_layers[da_index].train_cost_array.append([])
            sda.dA_layers[da_index].train_cost_array[-1].append(sda.dA_layers[da_index].epoch)
            sda.dA_layers[da_index].train_cost_array[-1].append(this_train_loss)
            sda.dA_layers[da_index].epoch += 1

            return numpy.mean(train_losses)
            
            
        def train_fn_grad(theta_value):
            sda.dA_layers[da_index].theta.set_value(theta_value, borrow=True)
            grad = sample_grad(0)
            for i in xrange(1, n_train_samples):
                grad += sample_grad(i)
            return grad / n_train_samples

        '''    
        train_result = train_fn_vis(da_index, sample_cost)
        
        train_fn = theano.function(
            inputs=[theta_value],
            outputs=train_result,
            updates=[(cur_dA.theta, theta_value)]
        )
            
        train_grad_result = train_fn_grad_vis(sample_grad)
            
        train_fn_grad = theano.function(
            inputs=[theta_value],
            outputs=train_grad_result,
            updates=[(cur_dA.theta, theta_value)]
        )
        '''                                                   
        # append `fn` to the list of functions
        pretrain_fns.append(train_fn)
        pretrain_updates.append(train_fn_grad)

    return pretrain_fns, pretrain_updates
   
def finetune_functions_sda_cg(sda, datasets, window_size):
    '''Generates a function `train` that implements one step of
    finetuning, a function `validate` that computes the error on
    a batch from the validation set, and a function `test` that
    computes the error on a batch from the testing set
    :type datasets: list of pairs of theano.tensor.TensorType
    :param datasets: It is a list that contain all the datasets;
        the has to contain three pairs, `train`,
        `valid`, `test` in this order, where each pair
        is formed of two Theano variables, one for the
        datapoints, the other for the labels
    :type window_size: int
    :param window_size: size of window
    '''

    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]

    # compute number of examples for validation and testing
    n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1       
    n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1       
    n_test_samples = test_set_x.get_value(borrow=True).shape[0] - window_size + 1

    index = T.lscalar('index')  # index to a sample

    validate_model = theano.function(
        [index],
        outputs=sda.logLayer.errors(sda.y),
        givens={
            sda.x: valid_set_x[index: index + window_size],
            sda.y: valid_set_y[index + window_size - 1]
        },
        name='valid'
    )
       
    test_model = theano.function(
        [index],
        outputs=sda.logLayer.errors(sda.y),
        givens={
            sda.x: test_set_x[index: index + window_size],
            sda.y: test_set_y[index + window_size - 1]
        },
        name='test'
    )
    
    # compute the cost for second phase of training,
    # defined as the negative log likelihood
    finetune_cost = sda.logLayer.negative_log_likelihood(sda.y)
    finetune_error = sda.logLayer.errors(sda.y)
            
    #  compile a theano function that returns the cost of a minibatch
    conj_cost = theano.function(
        [index],
        outputs=[finetune_cost, finetune_error],
        givens={
            sda.x: train_set_x[index: index + window_size],
            sda.y: train_set_y[index + window_size - 1]
        },
        name="conj_cost"
    )
        
    # compile a theano function that returns the gradient with respect to theta
    conj_grad = theano.function(
        [index],
        outputs=T.grad(finetune_cost, sda.logLayer.theta),
        givens={
            sda.x: train_set_x[index: index + window_size],
            sda.y: train_set_y[index + window_size - 1]
        },
        name="conj_grad"
    )
    
    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        sda.logLayer.theta.set_value(theta_value, borrow=True)
        cur_train_cost = []
        cur_train_error =[]
        for i in xrange(n_train_samples):
            sample_cost, sample_error = conj_cost(i)
            cur_train_cost.append(sample_cost)
            cur_train_error.append(sample_error)
            
        this_train_loss = float(numpy.mean(cur_train_cost))  
        sda.logLayer.train_cost_array.append([])
        sda.logLayer.train_cost_array[-1].append(sda.logLayer.epoch)
        sda.logLayer.train_cost_array[-1].append(this_train_loss)
           
        sda.logLayer.train_error_array.append([])
        sda.logLayer.train_error_array[-1].append(sda.logLayer.epoch)
        sda.logLayer.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
                    
        sda.logLayer.epoch += 1
            
        return this_train_loss

    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(theta_value):
        sda.logLayer.theta.set_value(theta_value, borrow=True)
        grad = conj_grad(0)
        for i in xrange(1, n_train_samples):
            grad += conj_grad(i)
        return grad / n_train_samples
        
    # creates the validation function
    def callback(theta_value):
        sda.logLayer.theta.set_value(theta_value, borrow=True)
        #compute the validation loss
        validation_losses = [validate_model(i)
                             for i in xrange(n_valid_samples)]
        this_validation_loss = numpy.mean(validation_losses)
        print('validation error %f %%' % (this_validation_loss * 100.,))
        sda.logLayer.valid_error_array.append([])
        sda.logLayer.valid_error_array[-1].append(sda.logLayer.epoch)
        sda.logLayer.valid_error_array[-1].append(this_validation_loss)
    
        # check if it is better then best validation score got until now
        if this_validation_loss < sda.logLayer.validation_scores[0]:
            # if so, replace the old one, and compute the score on the
            # testing dataset
            sda.logLayer.validation_scores[0] = this_validation_loss
            test_losses = [test_model(i)
                           for i in xrange(n_test_samples)]
            sda.logLayer.validation_scores[1] = numpy.mean(test_losses) * 100.
            sda.logLayer.test_error_array.append([])
            sda.logLayer.test_error_array[-1].append(sda.logLayer.epoch)
            sda.logLayer.test_error_array[-1].append(sda.logLayer.validation_scores[1])
    return train_fn, train_fn_grad, callback
    
def pretrain_sda_cg(sda, train_set_x, window_size, pretraining_epochs, corruption_levels):
    ## Pre-train layer-wise
    print '... getting the pretraining functions'
    import scipy.optimize
    pretraining_fns, pretraining_updates = pretraining_functions_sda_cg(
        sda=sda,
        train_set_x=train_set_x,
        window_size=window_size,
        corruption_levels=corruption_levels
    )
    print '... pre-training the model'
    # using scipy conjugate gradient optimizer
    print ("Optimizing using scipy.optimize.fmin_cg...")
    for i in xrange(sda.n_layers):
        best_w_b = scipy.optimize.fmin_cg(
            f=pretraining_fns[i],
            x0=numpy.zeros((sda.dA_layers[i].n_visible + 1) * sda.dA_layers[i].n_hidden,
                           dtype=sda.dA_layers[i].input.dtype),
            fprime=pretraining_updates[i],
            maxiter=pretraining_epochs
        )                            
    return sda
    
def finetune_sda_cg(sda, datasets, window_size, training_epochs):
    # get the training, validation and testing functions for the model
    print '... getting the finetuning functions'
    import scipy.optimize
    train_fn, train_fn_grad, callback = finetune_functions_sda_cg(
        sda=sda,
        datasets=datasets,
        window_size=window_size
    )

    print '... finetunning the model'
    print ("Optimizing using scipy.optimize.fmin_cg...")
    
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros(sda.logLayer.n_in * sda.logLayer.n_out + sda.logLayer.n_out, dtype=theano.config.floatX),
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=training_epochs
    )
    return sda
