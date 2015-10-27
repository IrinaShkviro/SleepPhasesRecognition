# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 17:27:33 2015

@author: irka
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
from HMM_second_with_sklearn import update_params_on_patient,\
 finish_training, get_error_on_patient, errors
from HMM_first_with_sklearn import GeneralHMM
from preprocess import preprocess_av_disp

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
        self.x = T.matrix('x')
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

def train_SdA(train_names, valid_names,
             output_folder, base_folder,
             window_size,
             corruption_levels,
             pretraining_epochs,
             start_base,
             rank,
             pretrain_lr):
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
    '''
    pretrained_sda = pretrain_sda_sgd(sda=sda,
                                  train_names=train_names,
                                  window_size=window_size,
                                  pretraining_epochs=pretraining_epochs,
                                  pretrain_lr=pretrain_lr,
                                  corruption_levels=corruption_levels)
    
    '''
    pretrained_sda = pretrain_sda_cg(sda=sda,
                                  train_names=train_names,
                                  window_size=window_size,
                                  pretraining_epochs=pretraining_epochs,
                                  corruption_levels=corruption_levels)
                         
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

    n_hiddens=[5]*n_out
    
    #create hmm container
    hmmLayer = GeneralHMM(
        n_hiddens = n_hiddens,
        n_hmms = n_out
    )
    
    #train_hmm        
    train_reader = ICHISeqDataReader(train_names)
    n_train_patients = len(train_names)
    #train hmms on data of each pattient
    for train_patient in xrange(n_train_patients):
        #get data divided on sequences with respect to labels
        train_set = train_reader.read_one_with_window(
            window_size=window_size,
            divide=True
        )
        for i in xrange(hmmLayer.n_hmms):
            cur_train_set = train_set[i].eval()
            if cur_train_set.shape[0] <= 0:
                continue
            print('train_set[i].eval(): ', train_set[i].eval().shape)
            #get (avg, disp) labels for x-values
            train_visible_after_sda = numpy.array([sda.get_da_output(
                numpy.array(cur_train_set[time]).reshape(1, -1))
                for time in xrange(cur_train_set.shape[0])])
                    
            x_labels = create_labels_after_das(
                da_output_matrix = train_visible_after_sda,
                rank=rank,
                start_base=start_base
            )
            hmmLayer.hmm_models[i].fit([numpy.array(x_labels).reshape(-1, 1)])
        
        error_cur_epoch = hmmLayer.validate_model(
            valid_names = valid_names,
            window_size = window_size,
            rank = rank,
            start_base = start_base
        )
        hmmLayer.valid_error_array.append([])
        hmmLayer.valid_error_array[-1].append(train_patient)
        hmmLayer.valid_error_array[-1].append(error_cur_epoch)
            
        gc.collect()
        
    gc.collect()
    print('MultinomialHMM created')
    
    sda.set_hmm_layer(
        hmm_model=hmmLayer
    )
    return sda
    
def test_sda(sda, test_names, rank, start_base, window_size=1, algo='viterbi'):
    test_reader = ICHISeqDataReader(test_names)    
    n_test_patients = len(test_names)
    
    for test_patient in xrange(n_test_patients):
        test_set_x, test_set_y = test_reader.read_one_with_window(
            window_size=window_size,
            divide=False
        )
        test_set_x = test_set_x.eval()
        test_set_y = test_set_y.eval()
        n_test_times = test_set_x.shape[0]
        
        test_visible_after_sda = numpy.array([sda.get_da_output(
                numpy.array(test_set_x[time]).reshape(1, -1))
                for time in xrange(n_test_times)])
                    
        new_test_visible = create_labels_after_das(
            da_output_matrix=test_visible_after_sda,
            rank=rank,
            start_base=start_base
        )
        '''
        n_patient_samples = len(test_set_y)
        half_window_size = int(window_size/2)
        new_test_hidden=test_set_y[half_window_size:n_patient_samples-half_window_size]
        '''
        predicted_states = sda.hmmLayer.define_labels_seq(new_test_visible)
        error_array=errors(predicted_states=numpy.array(predicted_states),
                       actual_states=numpy.array(test_set_y))
                       
        patient_error = error_array.eval().mean()
        
        print(patient_error, ' error for patient ' + str(test_patient))
        gc.collect()  
    
def test_all_params():
    window_size = 10
    train_data = ['p002']
    test_data = ['p002']
    output_folder=('[%s], [%s]')%(train_data, test_data)
    rank = 1
    start_base=10
    corruption_levels = [.1, .2]
    
    trained_sda = train_SdA(
            train_names=train_data,
            valid_names=test_data,
            output_folder=output_folder,
            base_folder='SdA_first_HMM_cg',
            window_size=window_size,
            corruption_levels=corruption_levels,
            pretrain_lr=0.,
            start_base=start_base,
            rank=rank,
            pretraining_epochs=15
        )
    test_sda(sda=trained_sda,
            test_names=test_data,
            rank=rank,
            start_base=start_base,
            window_size=window_size
    )
    '''
    all_train = ['002','003','005','007','08a','08b','09a','09b',
			'10a','011','012','013','014','15a','15b','016','017','018','019',
			'020','021','022','023','025','026','027','028','029',
			'030','031','032','033','034','035','036','037','038',
			'040','042','043','044','045','047','048','049','050',
			'051']
    train_data = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
                  'p10a','p011','p012','p013','p014','p15a','p15b','p016',
                  'p017','p018','p019','p020','p021','p022','p023','p025',
                  'p026','p027','p028','p029','p030','p031','p032','p033',
                  'p034','p035','p036','p037','p038','p040','p042','p043',
                  'p044','p045','p047','p048','p049','p050','p051']
    
    corruption_levels = [.1, .2]
    pretrain_lr=.03
    
    rank = 1
    start_base=10
    for test_pat_num in xrange(len(train_data)):
        test_pat = train_data.pop(test_pat_num)
        print(test_pat, 'test_pat')
        print(train_data, 'train_data')
        output_folder=('all_data, [%s]')%(test_pat)
        trained_sda = train_SdA(
            train_names=train_data,
            valid_names=[test_pat],
            output_folder=output_folder,
            base_folder='SdA_second_HMM',
            window_size=window_size,
            corruption_levels=corruption_levels,
            pretrain_lr=pretrain_lr,
            start_base=start_base,
            rank=rank,
            pretraining_epochs=1
        )
        test_sda(sda=trained_sda,
            test_names=[test_pat],
            rank=rank,
            start_base=start_base,
            window_size=window_size
        )
        train_data.insert(test_pat_num, test_pat)
        '''

def create_labels_after_das(da_output_matrix, rank, start_base=10):
    """
    Normalize sequence matrix and get average and dispersion
    """
    #normalization
    mins = da_output_matrix.min(axis=0)
    maxs = da_output_matrix.max(axis=0)
    da_output_matrix = ((da_output_matrix-mins)*((1-(-1.))/(maxs-mins)))/2
    #get average and dispersion
    avg_disp_matrix = numpy.array([[da_output_matrix[i].mean(),
                         da_output_matrix[i].max()-
                         da_output_matrix[i].min()]
        for i in xrange(da_output_matrix.shape[0])])
    base = pow(start_base, rank) + 1
    arounded_matrix = numpy.rint(avg_disp_matrix.flatten()*pow(start_base, rank)).reshape((da_output_matrix.shape[0], 2))
    data_labels = []
    #n_in=2
    for row in arounded_matrix:
        data_labels.append(int(row[0]*base + row[1]))
    return data_labels

if __name__ == '__main__':
    test_all_params()
