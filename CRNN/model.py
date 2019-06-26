# keras imports
import keras
import numpy as np
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Conv1D, GRU, Reshape

# std imports
import time
import os
import gc

class CRNN:
    def __init__(self, input_size, number_filter, number_dense, rate_drop_dense, validation_split_ratio):
        
        self.input_size = input_size
        self.number_filter = number_filter
        self.number_dense_units = number_dense
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio
        
    def train_model(self, X, Y, model_save_directory='./'):
        
        cut = int(len(X) * self.validation_split_ratio)
        train_data, val_data = np.expand_dims(X[cut:], axis=-1), np.expand_dims(X[:cut], axis=-1)
        train_labels = to_categorical(Y[cut:], 12)
        val_labels = to_categorical(Y[:cut], 12)

        if train_data is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        input_ = Input(shape=(self.input_size, 1), dtype='float32')
        conv1 = Conv1D(filters=32, kernel_size=9, strides=1, padding="same")
        conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding="same")
        
        x = conv1(input_)
        x = conv2(x)
#         x = Flatten()(x)
#         x = Reshape((, 12, ))(x)
        x = GRU(units=128, return_sequences=True)(x)
        x = GRU(units=128, return_sequences=True)(x)
        x = GRU(units=128, return_sequences=False)(x)
        x = Dense(128)(x)
        x = Dropout(self.rate_drop_dense)(x)
        x = Dense(128)(x)
        preds = Dense(self.number_dense_units)(x)
        
        rmsprop = RMSprop()

        model = Model(inputs=input_, outputs=preds)
        model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=30)

        STAMP = 'crnn_%d_%d_%.2f' % (self.number_filter, self.number_dense_units, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit(train_data, train_labels,
                  validation_data=(val_data, val_labels),
                  epochs=200, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return bst_model_path
    
    def update_model(self, saved_model_path, X, Y):

        cut = int(len(X) * self.validation_split_ratio)

        train_data, val_data = np.expand_dims(X[cut:], axis=-1), np.expand_dims(X[:cut], axis=-1)
        train_labels, val_labels = to_categorical(Y[cut:], 12), to_categorical(Y[:cut], 12)

        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))

        model.fit(train_data, train_labels,
                  validation_data=(val_data, val_labels),
                  epochs=50, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path
