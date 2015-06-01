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

import ichi_seq_data_reader
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
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        window_size=10,
        n_hidden=30,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
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
            print("No initial W")
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
                    self.n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
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
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
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
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch           
        L = T.sqrt(T.sum(T.sqr(T.flatten(self.x - z, 1))))
         
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

def train_dA(learning_rate=0.01, training_epochs=15,
            window_size=1, output_folder='dA_plots'):

    """
    This demo is tested on ICHI_Data

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
#    datasets = load_data(dataset)
    train_reader = ICHISeqDataReader(['p002'])
    train_set_x, train_set_y = train_reader.read_all()
    n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1

    valid_reader = ICHISeqDataReader(['p08a'])
    valid_set_x, valid_set_y = valid_reader.read_all()
    n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1
    
    test_reader = ICHISeqDataReader(['p019'])
    test_set_x, test_set_y = test_reader.read_all()   
    n_test_samples = test_set_x.get_value(borrow=True).shape[0] - window_size + 1

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    
    def train_corruption_model(corruption_level=0.):
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as 3D vector

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            window_size=window_size,
            n_hidden=30
        )

        cost, updates = da.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate
        )
    
        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index: index + window_size]
            }
        )
    
        validate_da = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                x: valid_set_x[index: index + window_size],
            }
        )
        
        test_da = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                x: test_set_x[index: index + window_size],
            }
        )

        ############
        # TRAINING #
        ############
        patience = 5000  # look as this many examples regardless
        patience_increase = 25  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        validation_frequency = patience / 2
    
        best_validation_loss = numpy.inf
        start_time = time.clock()
    
        done_looping = False
    
        train_error_array = []
        valid_error_array = []
        test_error_array = []
        cur_train_error = []
        cur_valid_error = []
    
        # go through training epochs
        try:
            for epoch in xrange(training_epochs):
                if done_looping:
                    break
                # go through trainng set
                for index in xrange(n_train_samples):
                    cur_cost = train_da(index)
                    cur_train_error.append(cur_cost)
                    # iteration number
                    iter = epoch * n_train_samples + index
                    
                    if (iter + 1) % validation_frequency == 0:
                            # compute zero-one loss on validation set
                            validation_losses = [validate_da(i)
                                                 for i in xrange(n_valid_samples)]
                            this_validation_loss = float(numpy.mean(validation_losses))                  
                            cur_valid_error.append(this_validation_loss)
                            
                            print(
                                'epoch %i, minibatch %i/%i, validation error %f %%' %
                                (
                                    epoch,
                                    index + 1,
                                    n_train_samples,
                                    this_validation_loss * 100.
                                )
                            )
            
                            # if we got the best validation score until now
                            if this_validation_loss < best_validation_loss:
                                #improve patience if loss improvement is good enough
                                if this_validation_loss < best_validation_loss *  \
                                   improvement_threshold:
                                    patience = max(patience, iter * patience_increase)
            
                                best_validation_loss = this_validation_loss
                                # test it on the test set
                               
                                test_losses = [test_da(i)
                                               for i in xrange(n_test_samples)]
                                test_score = float(numpy.mean(test_losses))
                                
                                test_error_array.append([])
                                test_error_array[-1].append(float(iter)/n_train_samples)
                                test_error_array[-1].append(test_score)
            
                                print(
                                    (
                                        '     epoch %i, minibatch %i/%i, test error of'
                                        ' best model %f %%'
                                    ) %
                                    (
                                        epoch,
                                        index + 1,
                                        n_train_samples,
                                        test_score * 100.
                                    )
                                )
                                
                    if patience <= iter:
                        done_looping = True
                        break
                    
                valid_error_array.append([])
                valid_error_array[-1].append(float(iter)/n_train_samples)
                valid_error_array[-1].append(float(numpy.mean(cur_valid_error)*100))
                cur_valid_error = []
            
                train_error_array.append([])
                train_error_array[-1].append(float(iter)/n_train_samples)
                train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
                cur_train_error =[]
                
        except  Exception:
            print('catch Exception')
    
        finally:
            test_losses = [test_da(i)
                                           for i in xrange(n_test_samples)]
            test_score = numpy.mean(test_losses)
            test_error_array.append([])
            test_error_array[-1].append(float(iter)/n_train_samples)
            test_error_array[-1].append(test_score)
            visualize_da(train_error_array, valid_error_array, test_error_array,
                            window_size, learning_rate, corruption_level)
    
        end_time = time.clock()
    
        training_time = (end_time - start_time)
        
        visualize_da(train_error_array, valid_error_array, test_error_array, 
                     window_size, learning_rate)
    
        print >> sys.stderr, ('The no corruption code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((training_time) / 60.))


    train_corruption_model(0.)
    train_corruption_model(0.3)

    os.chdir('../')


if __name__ == '__main__':
    train_dA()
