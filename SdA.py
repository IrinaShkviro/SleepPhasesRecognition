"""
 This code introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x.
"""
import os
import sys
import timeit
import gc

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, zero_in_array
from mlp import HiddenLayer
from dA import dA
from MyVisualizer import visualize_costs
from ichi_seq_data_reader import ICHISeqDataReader

# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        n_ins,
        hidden_layers_sizes,
        corruption_levels=[0.1, 0.1],
        theano_rng=None,
        n_outs=7
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        self.predict = self.logLayer.predict()

    def pretraining_functions(self, train_set_x, window_size):
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

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use

        pretrain_fns = []
        for cur_dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = cur_dA.get_cost_updates(corruption_level,
                                                learning_rate)
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
                    self.x: train_set_x[index: index + window_size]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, window_size, learning_rate):
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

        :type batch_size: int
        :param batch_size: size of a minibatch

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
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=[self.finetune_cost, self.errors, self.predict, self.y],
            updates=updates,
            givens={
                self.x: train_set_x[index: index + window_size],
                self.y: train_set_y[index + window_size - 1]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            outputs=self.errors,
            givens={
                self.x: test_set_x[index: index + window_size],
                self.y: test_set_y[index + window_size - 1]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            outputs=self.errors,
            givens={
                self.x: valid_set_x[index: index + window_size],
                self.y: valid_set_y[index + window_size - 1]
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


def test_SdA(datasets,output_folder, window_size,
             pretrain_lr=0.001, pretraining_epochs=15,
             finetune_lr=0.1, training_epochs=1000):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pretraining

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type datasets: array
    :param datasets: [train_set, valid_set, test_set]
    
    :type output_folder: string
    :param output_folder: folder for costand error graphics with results

    """

    # split the datasets
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]

    # compute number of examples given in training set
    n_train_samples =  train_set_x.get_value(borrow=True).shape[0] - window_size + 1    
 
    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=3*window_size,
        hidden_layers_sizes=[window_size*2, window_size*2],
        n_outs=7
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                window_size=window_size)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_samples):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost %f' % (i, epoch, float(numpy.mean(c)))

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        window_size=window_size,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'
    # early-stopping parameters
    patience = n_train_samples*2  # look as this many examples regardless
    patience_increase = 25  # wait this much longer when a new best is                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant   
    validation_frequency = patience / 4
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    iter = 0
    train_cost_array = []
    train_error_array = []
    valid_error_array = []
    test_error_array = []
    cur_train_cost =[]
    cur_train_error = []
    train_confusion_matrix = numpy.zeros((7, 7))
    print(n_train_samples, 'train_samples')
    
    while (epoch < training_epochs) and (not done_looping):
        train_confusion_matrix = zero_in_array(train_confusion_matrix)
        for index in xrange(n_train_samples):          
            sample_cost, sample_error, cur_pred, cur_actual = train_fn(index)
            # iteration number
            iter = epoch * n_train_samples + index
                
            cur_train_cost.append(sample_cost)
            cur_train_error.append(sample_error)
            train_confusion_matrix[cur_actual][cur_pred] += 1

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = float(numpy.mean(validation_losses))*100                 
                valid_error_array.append([])
                valid_error_array[-1].append(float(iter)/n_train_samples)
                valid_error_array[-1].append(this_validation_loss)
                        
                print(
                    'epoch %i, iter %i/%i, validation error %f %%' %
                    (
                        epoch,
                        index + 1,
                        n_train_samples,
                        this_validation_loss
                    )
                )
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                        improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = float(numpy.mean(test_losses))*100                           
                    test_error_array.append([])
                    test_error_array[-1].append(float(iter)/n_train_samples)
                    test_error_array[-1].append(test_score)
        
                    print(
                        (
                            '     epoch %i, iter %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            index + 1,
                            n_train_samples,
                            test_score
                        )
                    )
            if patience*4 <= iter:
                done_looping = True
                print('Done looping')
                break
                           
        train_cost_array.append([])
        train_cost_array[-1].append(float(iter)/n_train_samples)
        train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
        cur_train_cost =[]
       
        train_error_array.append([])
        train_error_array[-1].append(float(iter)/n_train_samples)
        train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
        cur_train_error =[]
                
        epoch = epoch + 1
        gc.collect()
            
    test_losses = test_model()
    test_score = float(numpy.mean(test_losses))*100                           
    test_error_array.append([])
    test_error_array[-1].append(float(iter)/n_train_samples)
    test_error_array[-1].append(test_score)
    
    visualize_costs(train_cost_array, train_error_array, 
                    valid_error_array, test_error_array,
                    window_size, learning_rate,
                    output_folder)
                    
    print(train_confusion_matrix, 'train_confusion_matrix')    

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss, best_iter + 1, test_score)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

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
    
    for lr in learning_rates:
        for ws in window_sizes:
            test_params(learning_rate=lr, n_epochs=1000, window_size = ws,
                                  train_set_x=train_set_x, train_set_y=train_set_y,
                                  valid_set_x=valid_set_x, valid_set_y=valid_set_y,
                                  test_set_x=test_set_x, test_set_y=test_set_y,
                                  train_data=train_data, valid_data=valid_data,
                                  test_data=test_data)

if __name__ == '__main__':
    test_SdA()