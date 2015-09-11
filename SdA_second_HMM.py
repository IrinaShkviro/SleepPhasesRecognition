# -*- coding: utf-8 -*-
"""
Created on Sept 2015

@author: Irka
"""

import os
import sys
import timeit
import gc

import numpy
from sklearn import hmm

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic import LogisticRegression
from mlp import HiddenLayer
from dA import dA
from MyVisualizer import visualize_pretraining, visualize_finetuning
from ichi_seq_data_reader import ICHISeqDataReader
from cg import pretrain_sda_cg, finetune_sda_cg
from sgd import pretrain_sda_sgd, finetune_sda_sgd
from HMM_second_with_sklearn import change_data_for_one_patient, \
    update_params_on_patient, finish_training, errors

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
        self.n_ins=n_ins
        self.n_outs=n_outs
        
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.iscalar('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
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
            self.params.append(sigmoid_layer.theta)

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
        sda_input = T.matrix('sda_input')  # the data is presented as rasterized images
        self.da_layers_output_size = hidden_layers_sizes[-1]
        self.get_da_output = theano.function(
            inputs=[sda_input],
            outputs=self.sigmoid_layers[-1].output.reshape((-1, self.da_layers_output_size)),
            givens={
                self.x: sda_input
            }
        )
        
    def set_hmm_layer(self, hmm_model):
        self.hmmLayer = hmm_model
        self.predict = self.hmmLayer.predict()
        self.errors = errors()

def test_SdA(datasets,
             output_folder, base_folder,
             window_size,
             corruption_levels,
             pretraining_epochs,
             training_epochs,
             pretrain_lr=0,
             finetune_lr=0):
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
    
    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_in,
        hidden_layers_sizes=[window_size*2, window_size],
        n_outs=n_out
    )
    # end-snippet-3 start-snippet-4
        
    #########################
    # PRETRAINING THE MODEL #
    #########################
    
    start_time = timeit.default_timer()
    
    pretrained_sda = pretrain_sda_sgd(sda=sda,
                                  train_set_x=train_set_x,
                                  window_size=window_size,
                                  pretraining_epochs=pretraining_epochs,
                                  pretrain_lr=pretrain_lr,
                                  corruption_levels=corruption_levels)
    '''

    pretrained_sda = pretrain_sda_cg(sda=sda,
                                  train_set_x=train_set_x,
                                  window_size=window_size,
                                  pretraining_epochs=pretraining_epochs,
                                  corruption_levels=corruption_levels)
    '''                       
    end_time = timeit.default_timer()
    
    for i in xrange(sda.n_layers):
        print(i, 'i pretrained')
        visualize_pretraining(train_cost=pretrained_sda.dA_layers[i].train_cost_array,
                              window_size=window_size,
                              learning_rate=0,
                              corruption_level=corruption_levels[i],
                              n_hidden=sda.dA_layers[i].n_hidden,
                              da_layer=i,
                              datasets_folder=output_folder,
                              base_folder=base_folder)

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################
                          
    #create matrices for params of HMM layer
    train_data_names = ['p10a','p011','p013','p014','p020','p022','p040',
                        'p045','p048','p09b','p023','p035','p038', 'p09a','p033']
    valid_data = ['p09b','p023','p035','p038', 'p09a','p033']

    n_train_patients=len(train_data_names)
    n_valid_patients=len(valid_data)
    
    rank = 1
    start_base=5
    base = pow(start_base, rank) + 1
    n_visible=pow(base, sda.da_layers_output_size)
    n_hidden=n_out
        
    train_reader = ICHISeqDataReader(train_data_names)
    valid_reader = ICHISeqDataReader(valid_data)
    
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden)) 
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))

    for train_patient in xrange(n_train_patients):
        #get data divided on sequences with respect to labels
        train_set_x, train_set_y = train_reader.read_next_doc()
        train_x_array = train_set_x.get_value()
        n_train_times = train_x_array.shape[0] - window_size + 1
        train_visible_after_sda = numpy.array([sda.get_da_output(
                train_x_array[time: time+window_size]).ravel()
                for time in xrange(n_train_times)]).ravel()
                    
        print(train_set_y.eval(), 'hiddens_patient')
        print(train_visible_after_sda, 'visibles_patient')
        
        new_train_visible, new_train_hidden = change_data_for_one_patient(
            hiddens_patient=train_set_y.eval(),
            visibles_patient=train_visible_after_sda,
            window_size=sda.da_layers_output_size,
            base_for_labels=base
        )
        
        pi_values, a_values, b_values, array_from_hidden = update_params_on_patient(
            pi_values=pi_values,
            a_values=a_values,
            b_values=b_values,
            array_from_hidden=array_from_hidden,
            hiddens_patient=new_train_hidden,
            visibles_patient=new_train_visible,
            n_hidden=n_hidden
        )
        
        gc.collect()
        
    pi_values, a_values, b_values = finish_training(
        pi_values=pi_values,
        a_values=a_values,
        b_values=b_values,
        array_from_hidden=array_from_hidden,
        n_hidden=n_hidden,
        n_patients=n_train_patients
    )
    
    hmm_model = hmm.MultinomialHMM(
        n_components=n_hidden,
        startprob=pi_values,
        transmat=a_values
    )
    
    hmm_model.n_symbols=n_visible
    hmm_model.emissionprob_=b_values 
    gc.collect()
    print('MultinomialHMM created')
    
    sda.set_hmm_layer(
        hmm_model=hmm_model
    )
                          
    start_time = timeit.default_timer()
    '''
    finetuned_sda = finetune_sda_sgd(sda=pretrained_sda,
                                    datasets=datasets,
                                    window_size=window_size,
                                    finetune_lr=finetune_lr,
                                    training_epochs=training_epochs)
    
    finetuned_sda = finetune_sda_cg(sda=pretrained_sda,
                                    datasets=datasets,
                                    window_size=window_size,
                                    training_epochs=training_epochs)
    
    end_time = timeit.default_timer()
    
    visualize_finetuning(train_cost=finetuned_sda.logLayer.train_cost_array,
                         train_error=finetuned_sda.logLayer.train_error_array,
                         valid_error=finetuned_sda.logLayer.valid_error_array,
                         test_error=finetuned_sda.logLayer.test_error_array,
                         window_size=window_size,
                         learning_rate=0,
                         datasets_folder=output_folder,
                         base_folder=base_folder)
    '''
def test_all_params():
    window_sizes = [1]
    
    #train_data = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    train_data = ['p10a']    
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
    corruption_levels = [.1, .2]
    pretrain_lr=.03
    finetune_lr=.03
    
    for ws in window_sizes:
        test_SdA(datasets=datasets,
                 output_folder=output_folder,
                 base_folder='SdA_sgd_cg_plots',
                 window_size=ws,
                 corruption_levels=corruption_levels,
                 pretrain_lr=pretrain_lr,
                 finetune_lr=finetune_lr,
                 pretraining_epochs=1,
                 training_epochs=1)

if __name__ == '__main__':
    test_all_params()
