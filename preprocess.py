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
    Normalize sequence matrix and divide elements on classes
    Return matrix with values from 0 to count of classes
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

def preprocess_av_disp(sequence_matrix, rank, window_size=1, n_in=3, start_base=10):
    """
    Normalize sequence matrix and get average and dispersion
    """
    #normalization
    mins = sequence_matrix.min(axis=0)
    maxs = sequence_matrix.max(axis=0)
    sequence_matrix = ((sequence_matrix-mins)*((1-(-1.))/(maxs-mins)))/2
    #get average and dispersion
    avg_disp_matrix = [[sequence_matrix[i: i + window_size].mean(axis=0),
                         sequence_matrix[i: i + window_size].max(axis=0)-
                         sequence_matrix[i: i + window_size].min(axis=0)]
        for i in xrange(sequence_matrix.shape[0]-window_size+1)]
    print(avg_disp_matrix, 'matrix av disp')
    #11^3 variant for labels
    base = pow(start_base, rank) + 1
    arounded_matrix = numpy.around(avg_disp_matrix, rank)*pow(10, rank)
    data_labels = []
    n_in=2*n_in
    for row in arounded_matrix:
        new_row = row.flat
        #create individual labels for vectors
        cur_value=0
        for degree in xrange(n_in):
            cur_value += new_row[degree]*pow(base, n_in-1-degree)
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
    array = numpy.asarray([(3, 5, 7),(8, 9, 3),(7, 5 ,0),(5, 9, 1),(11, 10, 3)], dtype=theano.config.floatX)
    window_size=5
    res=[[array[i: i + window_size].mean(axis=0), array[i: i + window_size].max(axis=0)-array[i: i + window_size].min(axis=0)]
        for i in xrange(array.shape[0]-window_size+1)] 

    print('str')    
    print(res)