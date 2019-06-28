from layers import Sequential, Dense
import numpy as np
from utils import load_wine

(X_train, y_train), (X_test, y_test)  = load_wine(0.8, normalize=True, shuffled=True)

model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],)))
model.add(Dense(y_train.shape[1]))

model.fit(X_train, y_train, 
         reg_factor=0.0, 
         epochs=1000, 
         learning_rate=0.1,
         batch_size=32,
         validation_data=(X_test, y_test)
)

model.plot_error()

# model.plot_train_error()

