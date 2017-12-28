import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Lambda, Flatten
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

train = pd.read_csv("../data/train.csv")

train_images = train.iloc[:, 1:].values
train_values = train.iloc[0:, 0].values.astype('int32')

train_images = train_images.reshape(train_images.shape[0], 28, 28)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

mean_px = train_images.mean().astype(np.float32)
std_px = train_images.std().astype(np.float32)

train_values = to_categorical(train_values)
num_classes = train_values.shape[1]

model = Sequential()
model.add(Lambda(
    lambda x: (x - mean_px) / std_px,
    input_shape=(28, 28, 1)
))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

gen = image.ImageDataGenerator()
train_images, test_images, train_values, test_values = train_test_split(train_images, train_values, test_size=0.10,
                                                                        random_state=42)
batches = gen.flow(train_images, train_values, batch_size=64)
val_batches = gen.flow(test_images, test_values, batch_size=64)

history = model.fit_generator(batches, batches.n, epochs=3,
                              validation_data=val_batches, validation_steps=val_batches.n)

model.optimizer.lr = 0.01
gen = image.ImageDataGenerator()
batches = gen.flow(train_images, train_values, batch_size=64)
model.fit_generator(batches, batches.n, epochs=1)

test = pd.read_csv("../data/test.csv")
test_data = test.values
test_data = test_data.values
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
predictions = model.predict_classes(test_data, verbose=0)

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                            "Label": predictions})
submissions.to_csv("../data/results.csv", index=False, header=True)
