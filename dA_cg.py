"""
 This tutorial introduces denoising auto-encoders (dA) using Theano and conjugate
gradient descent.

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
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ichi_seq_data_reader import ICHISeqDataReader
from MyVisualizer import visualize_da


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
        n_visible,
        n_hidden,
        theano_rng=None,
        input=None,
        theta=None,
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

        :type theta: theano.tensor.TensorType
        :param theta: Theano variable theta = (W, b). W gets the shape (n_visible, n_hidden),
                     while b is a vector of n_out elements, making theta a vector of
                     n_visible*n_hidden + n_hidden elements

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # initialize theta = (W,b) with 0s; W gets the shape (n_visible, n_hidden),
        # while b is a vector of n_out elements, making theta a vector of
        # n_visible*n_hidden + n_hidden elements
        if not theta:
            theta = theano.shared(
                value=numpy.asarray(
                    numpy_rng.uniform(
                        low=-4 * numpy.sqrt(6. / (n_hidden + n_visible + 1)),
                        high=4 * numpy.sqrt(6. / (n_hidden + n_visible + 1)),
                        size=(n_visible * n_hidden + n_hidden)
                    ),
                    dtype=theano.config.floatX
                ),
                name='theta',
                borrow=True
            )
        self.theta = theta
        
        # W is represented by the fisr n_visible*n_hidden elements of theta
        W = self.theta[0:n_visible * n_hidden].reshape((n_visible, n_hidden))
        # b is the rest (last n_hidden elements)
        bhid = self.theta[n_visible * n_hidden:n_visible * n_hidden + n_hidden]

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    (n_visible,),
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
        
        self.epoch = 0
        self.train_cost_array=[]
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
        
    def get_cost(self, corruption_level):
        """ This function computes the cost for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        
        y = self.get_hidden_values(tilde_x)
        self.z = self.get_reconstructed_input(y)
        
        #cost = -T.log(self.z)[self.x]
        cost = T.sqrt(T.sum(T.sqr(T.flatten(self.x - self.z, outdim=1))))

        return cost
        
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
        
def train_dA(training_epochs, window_size, corruption_level, n_hidden,
             dataset, output_folder, base_folder):

    """
    This dA is tested on ICHI_Data

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type window_size: int
    :param window_size: size of window used for training

    :type corruption_level: float
    :param corruption_level: corruption_level used for training the DeNosing
                          AutoEncoder

    :type n_hidden: int
    :param n_hidden: count of nodes in hidden layer

    :type datasets: array
    :param datasets: [train_set, valid_set, test_set]
    
    :type output_folder: string
    :param output_folder: folder for costand error graphics with results

    """
    
    # allocate symbolic variables for the data
    index = T.lscalar()    # index
    x = T.matrix('x')  # the data is presented as 3D vector

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    n_visible = window_size*3  # number of input units

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=n_visible,
        n_hidden=n_hidden
    )
    
    n_train_samples = dataset.get_value(borrow=True).shape[0] - window_size + 1

    cost = da.get_cost(
        corruption_level=corruption_level
    )
           
    #  compile a theano function that returns the cost
    conj_cost = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: dataset[index: index + window_size]
        },
        name="conj_cost"
    )

    # compile a theano function that returns the gradient with respect to theta
    conj_grad = theano.function(
        [index],
        T.grad(cost, da.theta),
        givens={
            x: dataset[index: index + window_size]
        },
        name="conj_grad"
    )
    
    da.train_cost_array = []
    da.epoch = 0

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
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = timeit.default_timer()
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((n_visible + 1) * n_hidden, dtype=x.dtype),
        fprime=train_fn_grad,
        disp=0,
        maxiter=training_epochs
    )
    end_time = timeit.default_timer()
    print(
            'Optimization complete'
    )

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))                 
    
    visualize_da(train_cost=da.train_cost_array,
                 window_size=window_size,
                 learning_rate=0,
                 corruption_level=corruption_level,
                 n_hidden=n_hidden,
                 output_folder=output_folder,
                 base_folder=base_folder)
    
def test_da_params(corruption_level):
    window_sizes = [13, 30, 50, 75, 100]
    
    train_data = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    
    train_reader = ICHISeqDataReader(train_data)
    train_set, train_labels = train_reader.read_all()
    
    output_folder=('[%s]')%(",".join(train_data))
    
    for ws in window_sizes:
        train_dA(training_epochs=1,
                 window_size = ws, 
                 corruption_level=corruption_level,
                 n_hidden=ws*2,
                 dataset=train_set,
                 output_folder=output_folder,
                 base_folder='dA_cg_plots')


if __name__ == '__main__':
    test_da_params(corruption_level=0.)
    test_da_params(corruption_level=0.3)
