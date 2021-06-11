import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Activation 
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from fsnn.converter import FSNNConverter

    # img = load_image('rsz_mnist_output_10.png')
    # img = test_x[1].reshape(1, 784)
    # img_class = test_y[1]
    # print(model.predict(img))
    # digit = np.argmax(model.predict(img), axis=-1)
# print(digit)

EPOCHS = 50
BATCH_SIZE = 784
VERBOSE = 1
N_CLASSES = 10
HIDDEN_LAYERS = 128
VALIDATION = 0.2
DROPOUT = 0.3
RESHAPE = 784

model = Sequential() 
    
model.add(Dense(8, input_shape=(RESHAPE,), activation='relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(64, activation='relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(N_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.9), metrics=['accuracy'])
model.summary()
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape(60000, RESHAPE)
test_x = test_x.reshape(10000, RESHAPE)

train_x = train_x.astype('float32')
train_x = train_x / 255.0

test_x = test_x.astype('float32')    
test_x =  test_x / 255.0

train_y = np_utils.to_categorical(train_y, N_CLASSES)
test_y = np_utils.to_categorical(test_y, N_CLASSES)

history  = model.fit(train_x, train_y, epochs=10, batch_size=BATCH_SIZE, verbose=VERBOSE, validation_split=VALIDATION)
score    = model.evaluate(test_x[:1000], test_y[:1000], verbose=VERBOSE)

print("Accuracy: {}".format(score[1] * 100))

imgs = []
for i in range(0, 10000):
    imgs.append(test_x[i].reshape(1, 784)[0])
a = test_x[0].reshape(1, 784)
print(model.predict(a))
print(np.argmax(model.predict(a)))

conv = FSNNConverter()
conv.set_k(16)
conv.set_alpha(18)
conv.set_input(imgs[0:1000])
conv.set_weights(model.get_weights())
conv.convert(model='mnist_snn.json')
