# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from fsnn.converter import FSNNConverter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#scale Features from 0 to 1
# transforms = Normalizer()
# transforms = MinMaxScaler(feature_range=(0, 5))
# X_train = transforms.fit_transform(X_train)
# X_test = transforms.fit_transform(X_test)

# x_train = preprocessing.normalize(X_train, axis=0)
# x_test = preprocessing.normalize(X_test, axis=0)

# X_train = X_train / X_train.max(axis=0)
# X_test = X_test / X_test.max(axis=0)

# define the keras model
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=500, batch_size=10)

# evaluate the keras model
pred, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
print(y_test)
conv = FSNNConverter()
conv.set_k(8)
conv.set_alpha(2)
conv.set_input(X_test)
conv.set_weights(model.get_weights())
conv.convert(model='diabetes_snn.json')