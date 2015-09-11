import numpy
from os import walk
import theano
import theano.tensor as T

def preprocess_sequence(sequence_matrix):
    """
    Normalize sequence matrix
    Return matrix with values from -1 to 1
    """
    mins = sequence_matrix.min(axis=0)
    maxs = sequence_matrix.max(axis=0)
    sequence_matrix = (sequence_matrix-mins)*((1-(-1.))/(maxs-mins)) - 1
    return sequence_matrix
    
def preprocess_for_HMM(sequence_matrix, rank, start_base, n_in=3):
    """
    Normalize sequence matrix
    Return matrix with values from -1 to 1
    """
    mins = sequence_matrix.min(axis=0)
    maxs = sequence_matrix.max(axis=0)
    #11^3 variant for labels
    base = pow(start_base, rank) + 1
    sequence_matrix = ((sequence_matrix-mins)*((1-(-1.))/(maxs-mins)))/(2*10/start_base)
    arounded_matrix = numpy.around(sequence_matrix, rank)*pow(10, rank)
    data_labels = []
    for row in arounded_matrix:
        #create individual labels for vectors
        cur_value=0
        for degree in xrange(n_in):
            cur_value += row[degree]*pow(base, n_in-1-degree)
        data_labels.append(int(cur_value))
    return data_labels

def preprocess_for_HMM_in_sda(sequence_matrix, rank, start_base, n_in=3):
    """
    Normalize sequence matrix
    Return matrix with values from -1 to 1
    """
    print(sequence_matrix, 'seq_matrix')
    sequence_matrix = sequence_matrix.reshape((-1, n_in))
    print(sequence_matrix, 'seq_matrix')
    mins = T.min(sequence_matrix, axis=0)
    maxs = T.max(sequence_matrix, axis=0)
    #11^3 variant for labels
    base = pow(start_base, rank) + 1
    sequence_matrix = ((sequence_matrix-mins)*((1-(-1.))/(maxs-mins)))/(2*10/start_base)
    print(sequence_matrix, 'seq_matrix')
    arounded_matrix = T.round(sequence_matrix * pow(10, rank))
    return arounded_matrix
    
def generate_random_probabilities(length):
    #generate randow values in interval [0; 99]
    full_array = [numpy.random.randint(0, 100) for i in xrange(length)]
    random_sum = numpy.sum(full_array) * 1.0
    normalized_array = full_array/random_sum
    return normalized_array
    
def generate_probabilities_for_matrix(x_length, y_length):
    result_matrix=[]
    for row in xrange(x_length):
        result_matrix.append(generate_random_probabilities(y_length))
    return result_matrix
    
if __name__ == '__main__':
    result = generate_probabilities_for_matrix(3, 5)
    n_elems=pow(1331, 3)
    print(sys.getsizeof(0.))
    print(16*7*n_elems/1024/1024/1024)