from layers import Sequential, Dense
import numpy as np

model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.fit([np.zeros(784)], [1])

print(model)