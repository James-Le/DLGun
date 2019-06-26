import numpy as np
from model import CRNN

train_X = np.loadtxt("data/train_sequence.csv", delimiter=",")
train_Y = np.loadtxt("data/train_label.csv", delimiter=",")

# hyper-parameters
input_size = 12
number_filter = 12
output_size = 12
rate_drop_dense = 0
validation_split_ratio = 0.1

cnn = CRNN(input_size, number_filter, output_size, rate_drop_dense, validation_split_ratio)

best_model_path = cnn.train_model(train_X, train_Y, model_save_directory='./')

