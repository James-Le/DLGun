import numpy as np
from model import SiameseBiLSTM
from keras.models import load_model

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

# generate prediction on the test set
test_x1 = np.loadtxt("data/test_ques.csv", delimiter=",")
test_x2 = np.loadtxt("data/test_para.csv", delimiter=",")

test_x1 = np.expand_dims(test_x1, axis=-1)
test_x2 = np.expand_dims(test_x2, axis=-1)

model = load_model(best_model_path)
preds = model.predict([test_x1, test_x2], verbose=1).ravel()
np.savetxt("data/test_pred.csv", preds, fmt='%d', delimiter=',')