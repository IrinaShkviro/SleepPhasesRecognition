"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
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
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ichi_seq_data_reader import ICHISeqDataReader
from MyVisualizer import visualize_da

try:
    import PIL.Image as Image
except ImportError:
    import Image


# start-snippet-1
class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space.
    """

    def __init__(
        self,
        numpy_rng,
        window_size,
        n_hidden,
        theano_rng=None,
        input=None,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input), the number of hidden units (the dimension
        d' of the latent or hidden space) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = 3*window_size
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + self.n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + self.n_visible)),
                    size=(self.n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    (self.n_visible,),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    (n_hidden,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input.reshape((1, self.n_visible))

        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        
        y = self.get_hidden_values(tilde_x)
        self.z = self.get_reconstructed_input(y)
        
        cost = T.sqrt(T.sum(T.sqr(T.flatten(self.x - self.z, outdim=1))))
                
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
        
    def predict(self):
        """
        Return predicted vector
        """
        return self.z
    
    def actual(self):
        """
        Return actual vector
        """
        return self.x    
        
def train_dA(learning_rate, training_epochs, window_size, corruption_level, n_hidden,
                          train_set, valid_set, test_set,
                          train_data, valid_data, test_data):

    """
    This dA is tested on ICHI_Data

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type window_size: int
    :param window_size: size of window used for training

    :type dataset: string
    :param dataset: path to the picked dataset
    
    :type corruption_level: float
    :param corruption_level: corruption_level used for training the DeNosing
                          AutoEncoder


    """
    
    n_train_samples = train_set.get_value(borrow=True).shape[0] - window_size + 1

    n_valid_samples = valid_set.get_value(borrow=True).shape[0] - window_size + 1
       
    n_test_samples = test_set.get_value(borrow=True).shape[0] - window_size + 1
    
    # allocate symbolic variables for the data
    index = T.lscalar()    # index
    x = T.matrix('x')  # the data is presented as 3D vector

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        window_size=window_size,
        n_hidden=n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )
    
    predict = da.predict()
    actual = da.actual()
    
    train_da = theano.function(
        [index],
        outputs=[cost, predict, actual],
        updates=updates,
        givens={
            x: train_set[index: index + window_size]
        }
    )
    
    validate_da = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set[index: index + window_size],
        }
    )
        
    test_da = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: test_set[index: index + window_size],
        }
    )

    ############
    # TRAINING #
    ############
    patience = n_train_samples*2  # look as this many examples regardless
    patience_increase = 25  # wait this much longer when a new best is
                                      # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
    validation_frequency = patience / 4
    
    best_validation_cost = numpy.inf
    start_time = time.clock()
    
    done_looping = False
    epoch = 0
    iter = 0
   
    train_cost_array = []
    valid_cost_array = []
    test_cost_array = []
    cur_train_cost =[]
    print(n_train_samples, 'train_samples')

    # go through training epochs
    while (epoch < training_epochs) and (not done_looping):
        # go through trainng set
        for index in xrange(n_train_samples):
            sample_cost, sample_predict, sample_actual = train_da(index)
            # iteration number
            iter = epoch * n_train_samples + index
            
            cur_train_cost.append(sample_cost)
               
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_costs = [validate_da(i)
                                   for i in xrange(n_valid_samples)]
                validation_cost = float(numpy.mean(numpy.asarray(validation_costs)))*100
                   
                valid_cost_array.append([])
                valid_cost_array[-1].append(float(iter)/n_train_samples)
                valid_cost_array[-1].append(validation_cost)
                    
                print(
                    'epoch %i, iter %i/%i, validation error %f %%' %
                    (
                        epoch,
                        index + 1,
                        n_train_samples,
                        validation_cost
                    )
                )
                   
                # if we got the best validation score until now
                if validation_cost < best_validation_cost:
                    #improve patience if loss improvement is good enough
                    if validation_cost < best_validation_cost *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
    
                    best_validation_cost = validation_cost
                    # test it on the test set                      
                    test_costs = [test_da(i)
                                   for i in xrange(n_test_samples)]
                    test_cost = float(numpy.mean(numpy.asarray(test_costs)))*100
                        
                    test_cost_array.append([])
                    test_cost_array[-1].append(float(iter)/n_train_samples)
                    test_cost_array[-1].append(test_cost)
    
                    print(
                        (
                            '     epoch %i, iter %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            index + 1,
                            n_train_samples,
                            test_cost
                        )
                    )
                              
            if patience*4 <= iter:
                done_looping = True
                break
                    
        train_cost_array.append([])
        train_cost_array[-1].append(float(iter)/n_train_samples)
        train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
        cur_train_cost =[]
                
        epoch = epoch + 1
                
    test_costs = [test_da(i)
                    for i in xrange(n_test_samples)]
    test_cost = float(numpy.mean(numpy.asarray(test_costs)))*100
                        
    test_cost_array.append([])
    test_cost_array[-1].append(float(iter)/n_train_samples)
    test_cost_array[-1].append(test_cost)                   
    
    visualize_da(train_cost_array, valid_cost_array, test_cost_array,
                 window_size, learning_rate, corruption_level, n_hidden,
                 train_data, valid_data, test_data)
    
    end_time = time.clock()
    training_time = (end_time - start_time)
    
    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                         ' ran for %.2fm' % ((training_time) / 60.))

def test_da_params(corruption_level):
    learning_rates = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015]
    window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    train_data = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    valid_data = ['p09b','p023','p035','p038']
    test_data = ['p09a','p033']
    
    train_reader = ICHISeqDataReader(train_data)
    train_set, train_labels = train_reader.read_all()
    
    valid_reader = ICHISeqDataReader(valid_data)
    valid_set, valid_labels = valid_reader.read_all()

    test_reader = ICHISeqDataReader(test_data)
    test_set, test_labels = test_reader.read_all()   
    
    for lr in learning_rates:
        for ws in window_sizes:
            train_dA(learning_rate=lr, training_epochs=1, window_size = ws, 
                     corruption_level=corruption_level, n_hidden=ws*2,
                     train_set=train_set, valid_set=valid_set, test_set=test_set,
                     train_data=train_data, valid_data=valid_data, test_data=test_data)


if __name__ == '__main__':
    test_da_params(corruption_level=0.)
    test_da_params(corruption_level=0.3)