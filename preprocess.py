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