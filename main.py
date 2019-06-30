from layers import Sequential, Dense
import numpy as np
from utils import load_ionosphere

(X_train, y_train), (X_test, y_test)  = load_ionosphere(0.7, normalize=True, shuffled=True)

model = Sequential()
model.add(Dense(15, input_shape=(X_train.shape[1],)))
model.add(Dense(15))
model.add(Dense(y_train.shape[1]))

model.fit(X_train, y_train, 
         reg_factor=0.0, 
         epochs=1000, 
         learning_rate=0.1,
         batch_size=32,
         validation_data=(X_test, y_test),
         verbose=True
)

model.plot_error()

# model.plot_train_error()

