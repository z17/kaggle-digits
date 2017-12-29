import random

from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils

import numpy as np

hidden_size = 10
num_classes = 2
epochs = 5

train_data_size = 5000
test_data_size = 1000

inp = Input(shape=(2,))
hidden_1 = Dense(hidden_size, activation='relu')(inp)
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1)
out = Dense(num_classes, activation='softmax')(hidden_2)

x_train = []
y_train = []

for x in range(0, train_data_size):
    a = bool(random.getrandbits(1))
    b = bool(random.getrandbits(1))
    x_train.append([a, b])
    y_train.append(a ^ b)

x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], 2).astype('int32')
y_train = np.asarray(y_train)
y_train = np_utils.to_categorical(y_train, num_classes)

model = Model(input=inp, output=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs= epochs, verbose=1, validation_split=0.1, batch_size=128)

data = []
for i in range(0, test_data_size):
    a = bool(random.getrandbits(1))
    b = bool(random.getrandbits(1))
    data.append([a, b])

data = np.asarray(data).astype('int32')
d = model.predict(data, 128, 1)

a = d.tolist()
fails = 0
for i, x in enumerate(a):
    if x[0] > x[1]:
        res = 0
    else:
        res = 1
    true_res = data[i][0] ^ data[i][1]
    if true_res != res:
        fails += 1

print("FAILS COUNT: " + str(fails) + " / " + str(test_data_size))
