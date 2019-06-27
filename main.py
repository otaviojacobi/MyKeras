from layers import Sequential, Dense
import numpy as np
from utils import load_mnist_60k


X = np.array([[0.13], [0.42]])
y = np.array([[0.9], [0.23]])


model = Sequential()
model.add(Dense(2, input_shape=(1,)))
model.add(Dense(1))

model.set_initial_weights([
   np.array([[0.4, 0.1], [0.3, 0.2]]),
   np.array([[0.7, 0.5, 0.6]])
])


model.fit(np.array(X), np.array(y), reg_factor=0, epochs=1)
#y_pred = model.predict(X)

#for idx in range(len(y)):
#    print(np.argmax(y_pred[idx]), np.argmax(y[idx]))
