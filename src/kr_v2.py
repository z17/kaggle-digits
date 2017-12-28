from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils
import pandas as pd
import functions
from sklearn.model_selection import train_test_split

batch_size = 128
num_epochs = 20
hidden_size = 512

height, width, depth = 28, 28, 1
num_classes = 10

train_data = pd.read_csv("../data/train.csv")
train_images = train_data.iloc[:, 1:].values
train_values = train_data.iloc[0:, 0].values.astype('int32')
train_images = train_images.reshape(train_images.shape[0], height * width)
train_images = train_images.astype('float32')
train_images /= 255

X_train, X_test, y_train, y_test = train_test_split(train_images, train_values,
                                                    test_size=0.10, random_state=42)

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

inp = Input(shape=(height * width,))
hidden_1 = Dense(hidden_size, activation='relu')(inp)
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1)
out = Dense(num_classes, activation='softmax')(hidden_2)

model = Model(input=inp, output=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1, validation_split=0.1)

history = model.evaluate(X_test, Y_test, verbose=1)

test = pd.read_csv("../data/test.csv")
test_data = test.values
test_data = test_data.reshape(test_data.shape[0], height * width)
predictions = model.predict(test_data, batch_size, 1)

data = list(predictions)
data = list(map(functions.convert_result, data))
submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": data})
submissions.to_csv("../data/results.csv", index=False, header=True)
