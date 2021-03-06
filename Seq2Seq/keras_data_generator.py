import numpy as np
import keras
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inps, tars, batch_size=64, max_len=20, shuffle=True):
        'Initialization'
        self.max_len = max_len
        self.batch_size = batch_size
        self.tars = tars
        self.inps = inps
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.inps) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_inps_temp = [self.inps[k] for k in indexes]
        
        list_tars_temp = [self.tars[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_inps_temp, list_tars_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.inps))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def pad(self, batch):
        'pad or cut to max_len'
        res = []
        
        for sample in batch:
            if len(sample) < self.max_len:
                res.append(sample + [0] * (self.max_len - len(sample)))
            else:
                res.append(sample[:self.max_len])
                
        return res
        
    def __data_generation(self, list_inps_temp, list_tars_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.max_len, 10))
        Y = np.empty((self.batch_size, self.max_len, 10))

        # Generate data
        for i, x in enumerate(self.pad(list_inps_temp)):
            X[i,] = to_categorical(x, 10)


        for i, y in enumerate(self.pad(list_tars_temp)):
            Y[i,] = to_categorical(y, 10)
        
        
        return X, Y