import gc
import timeit

import numpy

import theano
import theano.tensor as T

def zero_in_array(array):
    return [[0 for col in range(7)] for row in range(7)]

def train_logistic_sgd(learning_rate, window_size, n_epochs,
              datasets, classifier):
                  
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
    x = classifier.input  # data, presented as window with x, y, x for each sample
    y = T.iscalar('y')  # labels, presented as int label

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
    
def train_da_sgd(learning_rate, window_size, training_epochs, corruption_level,
              train_set, da):
    
    n_train_samples = train_set.get_value(borrow=True).shape[0] - window_size + 1
    
    # allocate symbolic variables for the data
    index = T.lscalar()    # index
    x = da.input

    cost = da.get_cost(
        corruption_level=corruption_level
    )
    
    # compute the gradients of the cost of the `dA` with respect
    # to its parameters
    gparams = T.grad(cost, da.params)
    # generate the list of updates
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(da.params, gparams)
    ]
        
    train_da = theano.function(
        [index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set[index: index + window_size]
        }
    )
    
    ############
    # TRAINING #
    ############
    print(n_train_samples, 'train_samples')

    # go through training epochs
    while da.epoch < training_epochs:
        # go through trainng set
        cur_train_cost = []
        for index in xrange(n_train_samples):
            cur_train_cost.append(train_da(index))
        
        train_cost = float(numpy.mean(cur_train_cost))
        
        da.train_cost_array.append([])
        da.train_cost_array[-1].append(da.epoch)
        da.train_cost_array[-1].append(train_cost)
        cur_train_cost =[]
        
        da.epoch = da.epoch + 1

        print 'Training epoch %d, cost ' % da.epoch, train_cost

    return da
    
def pretraining_functions_sda_sgd(sda, train_set_x, window_size):
    ''' Generates a list of functions, each of them implementing one
    step in trainnig the dA corresponding to the layer with same index.
    The function will require as input the minibatch index, and to train
    a dA you just need to iterate, calling the corresponding function on
    all minibatch indexes.

    :type train_set_x: theano.tensor.TensorType
    :param train_set_x: Shared variable that contains all datapoints used
                        for training the dA

    :type window_size: int
    :param window_size: size of a window

    :type learning_rate: float
    :param learning_rate: learning rate used during training for any of
                              the dA layers
    '''

    index = T.lscalar('index')
    corruption_level = T.scalar('corruption')  # % of corruption to use
    learning_rate = T.scalar('lr')  # learning rate to use

    pretrain_fns = []
    for cur_dA in sda.dA_layers:
        # get the cost and the updates list
        cost, updates = cur_dA.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate
        )

        # compile the theano function
        fn = theano.function(
            inputs=[
                index,
                theano.Param(corruption_level, default=0.2),
                theano.Param(learning_rate, default=0.1)
            ],
            outputs=cost,
            updates=updates,
            givens={
                sda.x: train_set_x[index: index + window_size]
            }
        )
        # append `fn` to the list of functions
        pretrain_fns.append(fn)

    return pretrain_fns
    
def build_finetune_functions(sda, datasets, window_size, learning_rate):
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

    :type learning_rate: float
    :param learning_rate: learning rate used during finetune stage
    '''

    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]

    # compute number of examples for validation and testing
    n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1       
    n_test_samples = test_set_x.get_value(borrow=True).shape[0] - window_size + 1

    index = T.lscalar('index')  # index to a sample

    # compute the gradients with respect to the model parameters
    gparams = T.grad(sda.finetune_cost, sda.params)

    # compute list of fine-tuning updates
    updates = [
        (param, param - gparam * learning_rate)
        for param, gparam in zip(sda.params, gparams)
    ]

    train_fn = theano.function(
        inputs=[index],
        outputs=[sda.finetune_cost,
                 sda.errors,
                 sda.predict,
                 sda.y],
        updates=updates,
        givens={
            sda.x: train_set_x[index: index + window_size],
            sda.y: train_set_y[index + window_size - 1]
        },
        name='train'
    )

    test_score_i = theano.function(
        [index],
        outputs=sda.errors,
        givens={
            sda.x: test_set_x[index: index + window_size],
            sda.y: test_set_y[index + window_size - 1]
        },
        name='test'
    )

    valid_score_i = theano.function(
        [index],
        outputs=sda.errors,
        givens={
            sda.x: valid_set_x[index: index + window_size],
            sda.y: valid_set_y[index + window_size - 1]
        },
        name='valid'
    )

    # Create a function that scans the entire validation set
    def valid_score():
        return [valid_score_i(i) for i in xrange(n_valid_samples)]

    # Create a function that scans the entire test set
    def test_score():
        return [test_score_i(i) for i in xrange(n_test_samples)]

    return train_fn, valid_score, test_score
    
def pretrain_sda_sgd(sda, train_set_x, window_size, pretraining_epochs,
                  pretrain_lr, corruption_levels):
    # compute number of examples given in training set
    n_train_samples =  train_set_x.get_value(borrow=True).shape[0] - window_size + 1    

    print '... getting the pretraining functions'
    pretraining_fns = pretraining_functions_sda_sgd(sda=sda,
                                                    train_set_x=train_set_x,
                                                    window_size=window_size)

    print '... pre-training the model'
    ## Pre-train layer-wise
    for i in xrange(sda.n_layers):
        cur_dA = sda.dA_layers[i]
        cur_dA.train_cost_array = []
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            cur_train_cost = []
            for index in xrange(n_train_samples):
                cur_train_cost.append(pretraining_fns[i](index=index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
                         
            cur_dA.train_cost_array.append([])
            cur_dA.train_cost_array[-1].append(epoch)
            cur_dA.train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
            
            print 'Pre-training layer %i, epoch %d, cost %f' % (i, epoch, float(numpy.mean(cur_train_cost)))
    return sda
    
def finetune_sda_sgd(sda, datasets, window_size, finetune_lr, training_epochs):
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing functions for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = build_finetune_functions(
        sda=sda,
        datasets=datasets,
        window_size=window_size,
        learning_rate=finetune_lr
    )
    
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]

    # compute number of examples for training
    n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1

    print '... finetunning the model'
    # early-stopping parameters
    patience = n_train_samples*2  # look as this many examples regardless
    patience_increase = 25  # wait this much longer when a new best is                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant   
    validation_frequency = patience / 2
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    done_looping = False
    best_iter = 0
    cur_train_cost =[]
    cur_train_error = []
    train_confusion_matrix = numpy.zeros((7, 7))
    print(n_train_samples, 'train_samples')
    
    while (sda.logLayer.epoch < training_epochs) and (not done_looping):
        train_confusion_matrix = zero_in_array(train_confusion_matrix)
        for index in xrange(n_train_samples):          
            sample_cost, sample_error, cur_pred, cur_actual = train_fn(index)
            # iteration number
            iter = sda.logLayer.epoch * n_train_samples + index
                
            cur_train_cost.append(sample_cost)
            cur_train_error.append(sample_error)
            train_confusion_matrix[cur_actual][cur_pred] += 1

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = float(numpy.mean(validation_losses))*100                 
                sda.logLayer.valid_error_array.append([])
                sda.logLayer.valid_error_array[-1].append(float(iter)/n_train_samples)
                sda.logLayer.valid_error_array[-1].append(this_validation_loss)
                        
                print(
                    'epoch %i, iter %i/%i, validation error %f %%' %
                    (
                        sda.logLayer.epoch,
                        index + 1,
                        n_train_samples,
                        this_validation_loss
                    )
                )
                
                # if we got the best validation score until now
                if this_validation_loss < sda.logLayer.validation_scores[0]:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < sda.logLayer.validation_scores[0] * \
                        improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    sda.logLayer.validation_scores[0] = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = float(numpy.mean(test_losses))*100                           
                    sda.logLayer.test_error_array.append([])
                    sda.logLayer.test_error_array[-1].append(float(iter)/n_train_samples)
                    sda.logLayer.test_error_array[-1].append(test_score)
        
                    print(
                        (
                            '     epoch %i, iter %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            sda.logLayer.epoch,
                            index + 1,
                            n_train_samples,
                            test_score
                        )
                    )
            if patience*4 <= iter:
                done_looping = True
                print('Done looping')
                break
                           
        sda.logLayer.train_cost_array.append([])
        sda.logLayer.train_cost_array[-1].append(float(iter)/n_train_samples)
        sda.logLayer.train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
        cur_train_cost =[]
       
        sda.logLayer.train_error_array.append([])
        sda.logLayer.train_error_array[-1].append(float(iter)/n_train_samples)
        sda.logLayer.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
        cur_train_error =[]
                
        sda.logLayer.epoch = sda.logLayer.epoch + 1
        gc.collect()
            
    test_losses = test_model()
    test_score = float(numpy.mean(test_losses))*100                           
    sda.logLayer.test_error_array.append([])
    sda.logLayer.test_error_array[-1].append(float(iter)/n_train_samples)
    sda.logLayer.test_error_array[-1].append(test_score)
    
    print(train_confusion_matrix, 'train_confusion_matrix')
    print(best_iter, 'best_iter')
    return sda