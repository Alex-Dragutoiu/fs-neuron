import numpy as np

from keras.models import Sequential
from keras.layers import Activation 
from keras.layers.core import Dense 
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy

from fsnn.converter import FSNNConverter

training = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target = np.array([[0],[1],[1],[0]], "float32")

model = Sequential() 
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(training, target, epochs=1000, batch_size=4, verbose=1)
print(model.predict(training))

conv = FSNNConverter()
conv.set_k(8)
conv.set_alpha(2)
conv.set_input(training)
conv.set_weights(model.get_weights())
conv.convert(model='xor_snn.json')

