# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model

# std imports
import time
import os
import gc
import numpy as np

class SiameseBiLSTM:
    def __init__(self, x1_length, x2_length, number_lstm, number_dense, rate_drop_lstm, 
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        
        self.x1_length = x1_length
        self.x2_length = x2_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio
        
    def train_model(self, X1, X2, Y, model_save_directory='./'):
        
        cut = int(len(X1) * self.validation_split_ratio)

        train_data_x1, val_data_x1 = np.expand_dims(X1[cut:], axis=-1), np.expand_dims(X1[:cut], axis=-1)
        train_data_x2, val_data_x2 = np.expand_dims(X2[cut:], axis=-1), np.expand_dims(X2[:cut], axis=-1)
        train_labels, val_labels = np.expand_dims(Y[cut:], axis=-1), np.expand_dims(Y[:cut], axis=-1)

        if train_data_x1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        # Creating LSTM Encoder
        lstm_layer = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))
        
        # Creating LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.x1_length,1), dtype='float32')
        x1 = lstm_layer(sequence_1_input)

        # Creating LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.x2_length,1), dtype='float32')
        x2 = lstm_layer(sequence_2_input)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=20)

        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2], train_labels,
                  validation_data=([val_data_x1, val_data_x2], val_labels),
                  epochs=200, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return bst_model_path
    
    def update_model(self, saved_model_path, X1, X2, Y):

        cut = int(len(X1) * self.validation_split_ratio)

        train_data_x1, val_data_x1 = np.expand_dims(X1[cut:], axis=-1), np.expand_dims(X1[:cut], axis=-1)
        train_data_x2, val_data_x2 = np.expand_dims(X2[cut:], axis=-1), np.expand_dims(X2[:cut], axis=-1)
        train_labels, val_labels = np.expand_dims(Y[cut:], axis=-1), np.expand_dims(Y[:cut], axis=-1)

        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2], train_labels,
                  validation_data=([val_data_x1, val_data_x2], val_labels),
                  epochs=50, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path