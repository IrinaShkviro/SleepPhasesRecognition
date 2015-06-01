import numpy
from os import walk

def preprocess_sequence(sequence_matrix):
    means = sequence_matrix.mean(axis=0)
    mins = sequence_matrix.min(axis=0)
    maxs = sequence_matrix.max(axis=0)
    sequence_matrix = (sequence_matrix-mins)*((1-(-1.))/(maxs-mins)) - 1
    return sequence_matrix
    
def shuffle(sequence_matrix, step):
    sm = []
    
    for i in range(step):
        for j in range(i,sequence_matrix.shape[0],step):
            sm.append(sequence_matrix[j])
            
            # if(sequence_matrix[j,0]==0 or sequence_matrix[j,0]==10 or sequence_matrix[j,0]==12):
                # sm.append(sequence_matrix[j])
                
            # if(sequence_matrix[j,0]==0 or sequence_matrix[j,0]==4):
                # sm.append(sequence_matrix[j])

    return numpy.array(sm)
    
def simplify(sequence_matrix, max_label):
    for j in range(sequence_matrix.shape[0]):
        sequence_matrix[j,0] = min(sequence_matrix[j,0],max_label)
        
        # if(sequence_matrix[j,0]==10): sequence_matrix[j,0]=1
        # if(sequence_matrix[j,0]==12): sequence_matrix[j,0]=2
        
        #if(sequence_matrix[j,0]==4): sequence_matrix[j,0]=1

    return sequence_matrix