from layers import Sequential, Dense
import numpy as np
from utils import load_mnist_60k

X, y = load_mnist_60k(10)

print(X.shape, y.shape)

model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.fit(np.array(X), np.array(y))
