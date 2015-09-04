import numpy
import gc

import theano
import theano.tensor as T

from preprocess import preprocess_sequence, preprocess_for_HMM

class ICHISeqDataReader(object):
    def __init__(self, seqs_for_analyse):
        print "init ICHISeqDataReader"
        
        #seqs - files for each patient
        self.seqs = seqs_for_analyse
        
        #n - count of patients
        self.n = len(self.seqs)
        
        #n_in - count of marks (dimension of input data)
        self.n_in = 3
        
        self.sequence_index = 0
        # path to folder with data
        dataset = 'D:\Irka\Projects\NeuralNetwok\data\data' # "./data/7/ICHI14_data_set/data"
        self.init_sequence(dataset)
    
    # read all docs in sequence
    def read_all(self):
        # sequence_matrix = array[size of 1st doc][data.x, data.y, data.z, data.gt]
        sequence_matrix = self.get_sequence()

        # d_x1 = array[size of 1st doc][x, y, z]
        d_x1 = preprocess_sequence(sequence_matrix[:, 0:self.n_in])
        
        # d_y1 = array[size of 1st doc][labels]
        d_y1 = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)

        # data_x_ar = union for (x, y, z) coordinates in all files
        data_x = d_x1
        
        # data_y_ar = union for labels in all files
        data_y = d_y1
        
        for t in range(len(self.seqs) - 1):
            # sequence_matrix = array[size of t-th doc][data.x, data.y, data.z, data.gt]
            sequence_matrix = self.get_sequence()

            # d_x = array[size of t-th doc][x, y, z]
            d_x = preprocess_sequence(sequence_matrix[:, 0:self.n_in])
            
            # d_y = array[size of t-th doc][labels]
            d_y = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)
            
            # concatenate data in current file with data in prev files in one array
            data_x = numpy.vstack((data_x, d_x))
            data_y = numpy.concatenate((data_y, d_y))
                            
            gc.collect()
        
        set_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=True)
        set_y = T.cast(theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=True), 'int32')
        
        return (set_x, set_y) 
    
    # read one doc in sequence
    def read_next_doc(self):    
       
        # sequence_matrix = array[size of doc][data.x, data.y, data.z, data.gt]
        sequence_matrix = self.get_sequence()
        
        # d_x = array[size of doc][x, y, z]
        d_x = preprocess_sequence(sequence_matrix[:, 0:self.n_in])
        
        # d_y = array[size of doc][labels]
        d_y = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)

        data_x = d_x
        data_y = d_y
           
        gc.collect()
        
        set_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=True)
        set_y = T.cast(theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=True), 'int32')
        
        return (set_x, set_y) 
        
    def init_sequence(self, dataset):
        self.sequence_files = []
        
        for f in self.seqs:
            # sequence_file - full path to each document
            sequence_file = dataset+"/"+str(f)+".npy"
            print sequence_file
            self.sequence_files.append(sequence_file)
            
    # define current file for reading
    def get_sequence(self):
        
        if self.sequence_index>=len(self.sequence_files):
            self.sequence_index = 0
            
        sequence_file = self.sequence_files[self.sequence_index]
        self.sequence_index = self.sequence_index+1
        #print sequence_file
        return self.read_sequence(sequence_file)
        
    #read sequence_file and return array of data (x, y, z, gt - label)
    def read_sequence(self, sequence_file):
        # load files with data as records
        data = numpy.load(sequence_file).view(numpy.recarray)
    
        data.gt[numpy.where(data.gt==7)] = 4
        
        # convert records with data to array with x, y, z coordinates and gt as label of class
        sequence_matrix = numpy.asarray(zip(data.x,data.y,data.z, data.gt))
  
        return sequence_matrix
        
    # read all docs in sequence
    def read_all_for_second_hmm(self, rank, start_base):
        # sequence_matrix = array[size of 1st doc][data.x, data.y, data.z, data.gt]
        sequence_matrix = self.get_sequence()

        # d_x1 = array[size of 1st doc][x, y, z]
        d_x1 = preprocess_for_HMM(sequence_matrix[:, 0:self.n_in], rank, start_base)
        
        # d_y1 = array[size of 1st doc][labels]
        d_y1 = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)

        # data_x_ar = union for (x, y, z) coordinates in all files
        data_x = []        
        data_x.append(d_x1)
        
        # data_y_ar = union for labels in all files
        data_y = []
        data_y.append(d_y1)
        
        for t in range(len(self.seqs) - 1):
            # sequence_matrix = array[size of t-th doc][data.x, data.y, data.z, data.gt]
            sequence_matrix = self.get_sequence()

            # d_x = array[size of t-th doc][x, y, z]
            d_x = preprocess_for_HMM(sequence_matrix[:, 0:self.n_in], rank, start_base)
            
            # d_y = array[size of t-th doc][labels]
            d_y = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)
            
            # concatenate data in current file with data in prev files in one array
            data_x.append(d_x)
            data_y.append(d_y)
                            
            gc.collect()
        
        set_x = theano.shared(data_x)
        set_y = theano.shared(data_y)
        
        return (set_x, set_y) 

    # read all docs in sequence
    def read_doc_for_second_hmm(self, rank, start_base):
        # sequence_matrix = array[size of 1st doc][data.x, data.y, data.z, data.gt]
        sequence_matrix = self.get_sequence()

        d_x = preprocess_for_HMM(sequence_matrix[:, 0:self.n_in], rank, start_base)
        
        d_y = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)
        
        set_x = theano.shared(d_x)
        set_y = theano.shared(d_y)
        
        return (set_x, set_y)         
        # read all docs in sequence
    def read_all_seqs_on_labels(self, rank, start_base):
        all_visible_seqs = []
        for label in xrange(7):
            # visible_seqs = array[count of labels][size of each label in doc][data.x, data.y, data.z, data.gt]
            visible_seqs = self.get_sequence_on_labels()
            #visible_seqs[label] is ndarray
            
            if visible_seqs[label]!=[]:
                # d_x1 = array[size of 1st doc][x, y, z]
                d_x1 = preprocess_for_HMM(visible_seqs[label][:, 0:self.n_in], rank, start_base)
                
                # d_y1 = array[size of 1st doc][labels]
                d_y1 = visible_seqs[label][:, self.n_in:self.n_in+1].reshape(-1)
        
                # data_x_ar = union for (x, y, z) coordinates in all files
                data_x = d_x1
                
                # data_y_ar = union for labels in all files
                data_y = d_y1
            else:
                data_x=[]
                data_y=[]
            
            for t in range(len(self.seqs) - 1):
                # sequence_matrix = array[size of t-th doc][data.x, data.y, data.z, data.gt]
                visible_seqs = self.get_sequence_on_labels()
                    

                if visible_seqs[label]!=[]:
                    # d_x = array[size of t-th doc]
                    # consider new labels for data
                    d_x = preprocess_for_HMM(visible_seqs[label][:, 0:self.n_in], rank, start_base)
                        
                    # d_y = array[size of t-th doc][labels]
                    d_y = visible_seqs[label][:, self.n_in:self.n_in+1].reshape(-1)
                        
                    # concatenate data in current file with data in prev files in one array
                    data_x = numpy.concatenate((data_x, d_x))
                    data_y = numpy.concatenate((data_y, d_y))
                                    
                gc.collect()
                    
            set_x = theano.shared(numpy.asarray(data_x,
                                                       dtype=theano.config.floatX),
                                         borrow=True)
            set_y = T.cast(theano.shared(numpy.asarray(data_y,
                                                       dtype=theano.config.floatX),
                                         borrow=True), 'int32')
            all_visible_seqs.append((set_x, set_y))
            
        return all_visible_seqs
        
       # define current file for reading
    def get_sequence_on_labels(self):
        
        if self.sequence_index>=len(self.sequence_files):
            self.sequence_index = 0
            
        sequence_file = self.sequence_files[self.sequence_index]
        self.sequence_index = self.sequence_index+1
        print(sequence_file, 'file')
        return self.read_sequence_on_labels(sequence_file)
        
    #read sequence_file and return array of data (x, y, z, gt - label)
    def read_sequence_on_labels(self, sequence_file):
        # load files with data as records
        data = numpy.load(sequence_file).view(numpy.recarray)
    
        data.gt[numpy.where(data.gt==7)] = 4
        
        # convert records with data to array with x, y, z coordinates and gt as label of class
        sequence_matrix = numpy.array(zip(data.x,data.y,data.z, data.gt))
        visible_seqs=[]
        
        for label in xrange(7):
            visible_seqs.append([])
        
        for row in sequence_matrix:
            label = row[3]
            #visible_seqs[label].append(row)
            if visible_seqs[label]==[]:
                visible_seqs[label] = row
            else:           
                visible_seqs[label] = numpy.vstack((visible_seqs[label], row))
  
        return visible_seqs

    # read all docs in sequence
    def read_all_and_divide(self, rank, start_base):
        # sequence_matrix = array[size of 1st doc][data.x, data.y, data.z, data.gt]
        sequence_matrix = self.get_sequence()

        # d_x1 = array[size of 1st doc][x, y, z]
        d_x1 = preprocess_for_HMM(sequence_matrix[:, 0:self.n_in], rank, start_base)
        
        # d_y1 = array[size of 1st doc][labels]
        d_y1 = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)

        # data_x_ar = union for (x, y, z) coordinates in all files
        data_x = d_x1
        
        # data_y_ar = union for labels in all files
        data_y = d_y1
        
        for t in range(len(self.seqs) - 1):
            # sequence_matrix = array[size of t-th doc][data.x, data.y, data.z, data.gt]
            sequence_matrix = self.get_sequence()

            # d_x = array[size of t-th doc][x, y, z]
            d_x = preprocess_for_HMM(sequence_matrix[:, 0:self.n_in], rank, start_base)
            
            # d_y = array[size of t-th doc][labels]
            d_y = sequence_matrix[:, self.n_in:self.n_in+1].reshape(-1)
            
            # concatenate data in current file with data in prev files in one array
            data_x = numpy.concatenate((data_x, d_x))
            data_y = numpy.concatenate((data_y, d_y))
                            
            gc.collect()
            
        all_data = zip(data_x, data_y)
        all_visible_seqs = []
        
        for label in xrange(7):
            data_x_for_cur_label=[]
            for row in all_data:
                if row[1] == label:
                    data_x_for_cur_label.append(row[0])
            #data_for_cur_label = all_data[numpy.where(all_data[:,1] == label)]
                        
            set_x = theano.shared(numpy.asarray(data_x_for_cur_label,
                                                       dtype=theano.config.floatX),
                                         borrow=True)
            
            all_visible_seqs.append((set_x, label))
        
        return all_visible_seqs
