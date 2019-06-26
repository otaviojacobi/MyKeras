from layers import Sequential, Dense
import numpy as np
from utils import load_mnist_60k


X = np.array([[2,3],[3,4],[4,5]])
y = np.array([[0.2] , [0.3], [0.4]])


model = Sequential()
model.add(Dense(2, input_shape=(2,)))
model.add(Dense(1))


model.fit(np.array(X), np.array(y))
