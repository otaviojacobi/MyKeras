from layers import Sequential, Dense
import numpy as np

model = Sequential()
model.add(Dense(3, input_shape=(3,)))
model.add(Dense(4))
model.add(Dense(2))

model.fit([np.array([1,1,1])], [1])

print(model)