import numpy as np
from model import SiameseBiLSTM

train_x1 = np.loadtxt("data/train_ques.csv", delimiter=",")
train_x2 = np.loadtxt("data/train_para.csv", delimiter=",")
train_Y = np.loadtxt("data/train_label.csv", delimiter=",")

# hyper-parameters
x1_length = 3
x2_length = 10
number_lstm = 50
rate_drop_lstm = 0
number_dense = 50
hidden_activation = "relu"
rate_drop_dense = 0 
validation_split_ratio = 0.1

siamese = SiameseBiLSTM(x1_length, x2_length, number_lstm, number_dense, rate_drop_lstm, rate_drop_dense, hidden_activation, validation_split_ratio)

best_model_path = siamese.train_model(train_x1, train_x2, train_Y, model_save_directory='./')