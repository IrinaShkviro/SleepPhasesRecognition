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

from logistic_sgd_cg import LogisticRegression
from mlp_cg import HiddenLayer
from dA_cg import dA
from MyVisualizer import visualize_pretraining, visualize_finetuning
from ichi_seq_data_reader import ICHISeqDataReader

theano.config.exception_verbosity='high'

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
        self.y = T.iscalar('y')  # the labels are presented as 1D vector of
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
                          theta=sigmoid_layer.theta)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

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
        The function will require as input the index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type window_size: int
        :param window_size: size of a window
        '''

        # index
        index = T.lscalar('index')
        theta_value = T.vector('theta')
        x = T.matrix('x')  # the data is presented as 3D vector
        corruption_level = T.scalar('corruption')  # % of corruption to use
        n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
        
        # creates a function that computes the average cost on the training set
        def train_fn_vis(cur_dA, conj_cost):
            train_losses = [conj_cost(i) for i in xrange(n_train_samples)]
                                            
            this_train_loss = float(numpy.mean(train_losses))  
            cur_dA.train_cost_array.append([])
            cur_dA.train_cost_array[-1].append(cur_dA.epoch)
            cur_dA.train_cost_array[-1].append(this_train_loss)
            cur_dA.epoch += 1
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
        for cur_dA in self.dA_layers:
            # get the cost and the updates list
            cost = cur_dA.get_cost(corruption_level)

            # compile a theano function that returns the cost
            conj_cost = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                ],
                outputs=cost,
                givens={
                    self.x: train_set_x[index: index + window_size]
                },
                on_unused_input='warn'
            )
        
            # compile a theano function that returns the gradient with respect to theta
            conj_grad = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                ],
                outputs=T.grad(cost, cur_dA.theta),
                givens={
                    self.x: train_set_x[index: index + window_size]
                },
                on_unused_input='warn'
            )
            
            cur_dA.train_cost_array = []
            cur_dA.epoch = 0
            
            train_result = train_fn_vis(cur_dA, conj_cost)
            
            train_fn = theano.function(
                inputs=[theta_value],
                outputs=train_result,
                updates=[(cur_dA.theta, theta_value)]
            )
            
            train_grad_result = train_fn_grad_vis(conj_grad)
            
            train_fn_grad = theano.function(
                inputs=[theta_value],
                outputs=train_grad_result,
                updates=[(cur_dA.theta, theta_value)]
            )
                                                           
            # append `fn` to the list of functions
            pretrain_fns.append(train_fn)
            pretrain_updates.append(train_fn_grad)

        return pretrain_fns, pretrain_updates

    def build_finetune_functions(self, datasets, window_size):
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
            outputs=self.errors,
            givens={
                self.x: valid_set_x[index: index + window_size],
                self.y: valid_set_y[index + window_size - 1]
            },
            name='valid'
        )
        
        test_model = theano.function(
            [index],
            outputs=self.errors,
            givens={
                self.x: test_set_x[index: index + window_size],
                self.y: test_set_y[index + window_size - 1]
            },
            name='test'
        )
            
        #  compile a theano function that returns the cost of a minibatch
        conj_cost = theano.function(
            [index],
            outputs=[self.finetune_cost, self.errors],
            givens={
                self.x: train_set_x[index: index + window_size],
                self.y: train_set_y[index + window_size - 1]
            },
            name="conj_cost"
        )
        
        # compile a theano function that returns the gradient with respect to theta
        conj_grad = theano.function(
            [index],
            outputs=T.grad(self.finetune_cost, self.logLayer.theta),
            givens={
                self.x: train_set_x[index: index + window_size],
                self.y: train_set_y[index + window_size - 1]
            },
            name="conj_grad"
        )
        
        self.logLayer.train_cost_array = []
        self.logLayer.train_error_array = []
        self.logLayer.epoch = 0
       
        # creates a function that computes the average cost on the training set
        def train_fn(theta_value):
            self.logLayer.theta.set_value(theta_value, borrow=True)
            cur_train_cost = []
            cur_train_error =[]
            for i in xrange(n_train_samples):
                sample_cost, sample_error = conj_cost(i)
                cur_train_cost.append(sample_cost)
                cur_train_error.append(sample_error)
            
            this_train_loss = float(numpy.mean(cur_train_cost))  
            self.logLayer.train_cost_array.append([])
            self.logLayer.train_cost_array[-1].append(self.logLayer.epoch)
            self.logLayer.train_cost_array[-1].append(this_train_loss)
           
            self.logLayer.train_error_array.append([])
            self.logLayer.train_error_array[-1].append(self.logLayer.epoch)
            self.logLayer.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
                    
            self.logLayer.epoch += 1
            
            return this_train_loss

        # creates a function that computes the average gradient of cost with
        # respect to theta
        def train_fn_grad(theta_value):
            self.logLayer.theta.set_value(theta_value, borrow=True)
            grad = conj_grad(0)
            for i in xrange(1, n_train_samples):
                grad += conj_grad(i)
            return grad / n_train_samples

        self.logLayer.validation_scores = [numpy.inf, 0]
        self.logLayer.valid_error_array = []
        self.logLayer.test_error_array = []

        # creates the validation function
        def callback(theta_value):
            self.logLayer.theta.set_value(theta_value, borrow=True)
            #compute the validation loss
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_samples)]
            this_validation_loss = numpy.mean(validation_losses)
            print('validation error %f %%' % (this_validation_loss * 100.,))
    
            # check if it is better then best validation score got until now
            if this_validation_loss < self.logLayer.validation_scores[0]:
                # if so, replace the old one, and compute the score on the
                # testing dataset
                self.logLayer.validation_scores[0] = this_validation_loss
                test_losses = [test_model(i)
                               for i in xrange(n_test_samples)]
                self.logLayer.validation_scores[1] = numpy.mean(test_losses)

        return train_fn, train_fn_grad, callback


def test_SdA(datasets,
             output_folder, base_folder,
             window_size,
             pretraining_epochs=15,
             training_epochs=1000):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on ICHI.

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

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
    n_in = window_size*3  # number of input units
    n_out = 7  # number of output units
    x = T.matrix('x')
    
    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_in,
        hidden_layers_sizes=[window_size*2, window_size*2],
        n_outs=n_out
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    '''
    print '... getting the pretraining functions'
    pretraining_fns, pretraining_updates = sda.pretraining_functions(train_set_x=train_set_x,
                                                window_size=window_size)
    '''
    print '... pre-training the model'
    # using scipy conjugate gradient optimizer
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")

    start_time = timeit.default_timer()
    index = T.lscalar('index')
    n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
    ## Pre-train layer-wise
    corruption_levels = [.1, .2]
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

        best_w_b = scipy.optimize.fmin_cg(
            f=train_fn,
            x0=numpy.zeros((sda.dA_layers[da_index].n_visible + 1) * sda.dA_layers[da_index].n_hidden,
                           dtype=x.dtype),
            fprime=train_fn_grad,
            disp=0,
            maxiter=pretraining_epochs
        )
        visualize_pretraining(train_cost=sda.dA_layers[da_index].train_cost_array,
                              window_size=window_size,
                              learning_rate=0,
                              corruption_level=corruption_levels[da_index],
                              n_hidden=sda.dA_layers[da_index].n_hidden,
                              da_layer=da_index,
                              datasets_folder=output_folder,
                              base_folder=base_folder)

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing functions for the model
    print '... getting the finetuning functions'
    train_fn, train_fn_grad, callback = sda.build_finetune_functions(
        datasets=datasets,
        window_size=window_size
    )

    print '... finetunning the model'
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = timeit.default_timer()
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((n_in + 1) * n_out, dtype=sda.x.dtype),
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=training_epochs
    )
    
    print(
        (
            'Optimization complete with best validation score of %f %%, with '
            'test performance %f %%'
        )
        % (sda.validation_scores[0] * 100., sda.validation_scores[1] * 100.)
    )
    
    visualize_finetuning(train_cost=sda.logLayer.train_cost_array,
                         train_error=sda.logLayer.train_error_array,
                         valid_error=sda.logLayer.valid_error_array,
                         test_error=sda.logLayer.test_error_array,
                         window_size=window_size,
                         learning_rate=0,
                         datasets_folder=output_folder,
                         base_folder=base_folder)
    
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
        test_SdA(datasets=datasets,
                 output_folder=output_folder,
                 base_folder='SdA_cg_plots',
                 window_size=ws,
                 pretraining_epochs=100,
                 training_epochs=1000)

if __name__ == '__main__':
    test_all_params()
