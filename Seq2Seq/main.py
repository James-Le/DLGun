from models import *
import numpy as np
from keras_data_generator import DataGenerator

# hyper-parameters
max_len = 20
batch_size = 64
validation_split_ratio = 0.1

# load data
train_X, train_Y = [], []

with open("data/train_inp.txt") as file:
    for line in file:
        if line:
            train_X.append([int(x) if x else 0 for x in line.strip().split(",")])
        else:
            train_X.append([0])
        
with open("data/train_tar.txt") as file:
    for line in file:
        if line:
            train_Y.append([int(y) if y else 0 for y in line.strip().split(",")])
        else:
            train_Y.append([0])

cut = int(validation_split_ratio * len(train_X))
        
train_generator = DataGenerator(train_X[-cut:], train_Y[-cut:], batch_size, max_len)
val_generator = DataGenerator(train_X[:-cut], train_Y[:-cut], batch_size, max_len)

model = AttentionSeq2Seq(input_dim=10, input_length=20, hidden_dim=100, output_length=20, output_dim=10, depth=3)

model.compile(loss='mse', optimizer='rmsprop')

model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    verbose=1, epochs=30)


