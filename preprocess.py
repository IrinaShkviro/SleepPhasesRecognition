import numpy
from os import walk

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
