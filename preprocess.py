import numpy
from os import walk
import theano
import theano.tensor as T

def preprocess_sequence(sequence_matrix):
    """
    Normalize sequence matrix
    Return matrix with values from -1 to 1
    """
    means = sequence_matrix.mean(axis=0)
    mins = sequence_matrix.min(axis=0)
    maxs = sequence_matrix.max(axis=0)
    sequence_matrix = (sequence_matrix-mins)*((1-(-1.))/(maxs-mins)) - 1
    return sequence_matrix
    
def preprocess_for_HMM(sequence_matrix):
    """
    Normalize sequence matrix
    Return matrix with values from -1 to 1
    """
    means = sequence_matrix.mean(axis=0)
    mins = sequence_matrix.min(axis=0)
    maxs = sequence_matrix.max(axis=0)
    #11^3 variant for labels
    rank = 1
    base = pow(10, rank) + 1
    sequence_matrix = ((sequence_matrix-mins)*((1-(-1.))/(maxs-mins)))/2
    arounded_matrix = numpy.around(sequence_matrix, rank)*pow(10, rank) + 1
    data_labels = []
    min_label = int(pow(base, 2) + base + 1)
    i = 0
    for row in arounded_matrix:
        i+=1
        #create individual labels for vectors
        cur_value = int(row[0]*pow(base, 2) + row[1]*base + row[2] - min_label)
        data_labels.append(cur_value)    
    return data_labels
    
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
    print('0')
