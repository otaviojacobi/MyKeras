from abc import ABC, abstractmethod
import activations
import numpy as np
from activations import sigmoid
from utils import unison_shuffled_copies
import matplotlib.pyplot as plt
#from copy import deepcopy

class Layer(ABC):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_type(self):
        pass

class Dense(Layer):

    def __init__(self, units, activation='sigmoid', use_bias=True, input_shape=None):
        self.units = units
        self.use_bias = use_bias

        if activation in activations.VALID_ACTIVATION_FUNCTIONS:
            if activation == 'linear':
                self.activate = activations.linear
            elif activation == 'sigmoid':
                self.activate = activations.sigmoid
            elif activation == 'softmax':
                self.activate = activations.softmax
        else:
            raise Exception('Invalid input shape parameter')

        if self.__is_valid_input_shape(input_shape):
            self.input_shape = input_shape 
        else:
            raise Exception('Invalid input shape parameter')

    def get_type(self):
        return 'dense'

    def __is_valid_input_shape(self, input_shape):
        return isinstance(input_shape, tuple) or input_shape is None

class Sequential:

    def __init__(self, *args):
        self.model = Model(list(args))

    def __str__(self):
        return self.model.__str__()

    def add(self, layer):
        self.model.add(layer)

    def compile(self):
        self.model.compile()
        return self

    def set_initial_weights(self, weights):
        return self.model.set_initial_weights(weights)

    def predict(self, x_predict):
        return self.model.predict(x_predict)

    def fit(self, x_training, y_training, reg_factor=0.25, learning_rate=0.0001, epochs=1000, batch_size=1, shuffle=False, validation_data=None, verbose=True):
        self.model.fit(x_training, y_training, reg_factor, learning_rate, epochs, batch_size, shuffle, validation_data, verbose)
        return self

    def plot_train_error(self):
        return self.model.plot_train_error()

    def plot_validation_error(self):
        return self.model.plot_validation_error()

    def plot_error(self):
        return self.model.plot_error()

class Model:

    def __init__(self, layers):
        if not self.__are_layers_valid(layers):
            raise Exception('Use only proper layers in model creation')
        self.layers = layers
        self.__weights_inited = False

    def __str__(self):
        return self.__layers_as_str()

    def add(self, layer):
        if self.__is_layer_valid(layer):
            self.layers.append(layer)
        else:
            raise Exception('Use only proper layers in model creation')

    def compile(self):
        pass

    def fit(self, x_training, y_training, reg_factor, learning_rate, epochs, batch_size, shuffle, validation_data, verbose):
        if not self.__are_x_y_valid(x_training, y_training):
            raise Exception('Invalid vectors for training')
        if len(x_training)//batch_size < 1:
            raise Exception('Batch size too large')

        self.has_validation_data = False
        if validation_data != None and len(validation_data) != 2 and self.__are_x_y_valid(validation_data[0], validation_data[1]):
            raise Exception('Validation data not valid')
        elif validation_data != None:
            self.has_validation_data = True

        if shuffle:
            x_training, y_training = unison_shuffled_copies(x_training, y_training)

        self.weights_matrix = self.__init_weights_matrix()
        self.errors = np.zeros(epochs)
        self.validation_error = np.zeros(epochs)
        for epoch in range(epochs):
            for batch in range(len(x_training)//batch_size):
                J, grads = self.__cost_and_grad(x_training[batch*batch_size:(batch+1)*batch_size], y_training[batch*batch_size:(batch+1)*batch_size], reg_factor)
                for idx in range(len(self.weights_matrix)):
                    self.weights_matrix[idx] = self.weights_matrix[idx] - learning_rate * grads[idx]
            self.errors[epoch] = J
            if verbose:
                if epoch in [0,1,2,3,4,5, epochs-2, epochs-1, epochs] or epoch % 100 == 0:

                    y_predicted_train = np.array([np.argmax(k) for k in self.predict(x_training)])
                    y_training_maxed = np.array([np.argmax(k) for k in y_training])
                    scored_train = int(100*( np.count_nonzero(y_predicted_train == y_training_maxed) )/float(len(y_training_maxed)))
                    print("Correct guesses in training set %d%% (epoch %d/%d) err: %f"%(scored_train , epoch, epochs, self.errors[epoch] ))
                    
                    if self.has_validation_data:
                        y_predicted_validation = np.array([np.argmax(k) for k in self.predict(validation_data[0])])
                        y_validation_maxed = np.array([np.argmax(k) for k in validation_data[1]])
                        scored_test = int(100*( np.count_nonzero(y_predicted_validation == y_validation_maxed) )/float(len(y_predicted_validation)))
                        print("Correct guesses in test set %d%% (epoch %d/%d)"%(scored_test, epoch, epochs ))

            if self.has_validation_data:
                J_validation, _ = self.__cost_and_grad(validation_data[0], validation_data[1], reg_factor)
                self.validation_error[epoch] = J_validation

        self.is_trained = True


    def predict(self, x_predict):
        size = x_predict.shape[0]
        activation_value = self.__add_bias(x_predict, size)
        activation_values = [activation_value]
        for layer_idx in range(len(self.layers)-1):
            z = np.matmul(activation_value, self.weights_matrix[layer_idx].T)
            activation_value = self.__add_bias(self.layers[layer_idx].activate(z), size)
            activation_values.append(activation_value)

        return self.layers[-1].activate(np.matmul(activation_values[-1], self.weights_matrix[-1].T))

    def set_initial_weights(self, weights):
        sample_matrix = self.__init_weights_matrix()
        if len(sample_matrix) != len(weights):
            raise Exception('Inited weights are not in correct format')

        for idx in range(len(sample_matrix)):
            if sample_matrix[idx].shape != weights[idx].shape:
                raise Exception('Inited weights are not in correct format')

        self.__weights_inited = True
        self.weights_matrix = weights

    def plot_train_error(self):

        if not self.is_trained:
            raise Exception('First train your model (fit) before plotting')
        plt.plot(self.errors)
        plt.show()

    def plot_validation_error(self):
        if not self.is_trained:
            raise Exception('First train your model (fit) before plotting')
        plt.plot(self.validation_error)
        plt.show()

    def plot_error(self):
        if not self.is_trained:
            raise Exception('First train your model (fit) before plotting')
        plt.plot(self.errors)
        if self.has_validation_data:
            plt.plot(self.validation_error)
        plt.legend(['train error', 'test error'])
        plt.show()


    def __cost_and_grad(self, x_training, y_training, reg_factor):
        self.m = x_training.shape[0]
        activation_value = self.__add_bias(x_training, self.m)

        activation_values = [activation_value]
        regularization = 0
        Z = []
        ## Foward prop
        for layer_idx in range(len(self.layers)-1):
            z = np.matmul(activation_value, self.weights_matrix[layer_idx].T)
            Z.append(z)
            activation_value = self.__add_bias(self.layers[layer_idx].activate(z), self.m)
            activation_values.append(activation_value)
            regularization += np.sum(np.power(self.weights_matrix[layer_idx][:, 1:], 2))

        H = self.layers[-1].activate(np.matmul(activation_values[-1], self.weights_matrix[-1].T))
        J = np.sum(np.multiply(-y_training, np.log(H)) - np.multiply((1-y_training), np.log(1-H)))/self.m
        
        # THIS IS WRONG: TODO: FIXME
        J += (reg_factor/(2*self.m)) * regularization

        ## Back prop
        last_sigma = H - y_training
        sigmas = [last_sigma]
        for layer_idx in range(len(self.layers)-1):
            old_sigma_and_weights = np.matmul(sigmas[-1], self.weights_matrix[-layer_idx-1])
            sigmoid_derivative = sigmoid(self.__add_bias(Z[-layer_idx-1], self.m), derivative=True)
            sigma = np.multiply(old_sigma_and_weights, sigmoid_derivative)[:, 1:]
            sigmas.append(sigma)
        sigmas.reverse()

        grads = []
        for idx in range(len(sigmas)):
            delta = (np.matmul(sigmas[idx].T, activation_values[idx]))/self.m
            grad = delta + ((reg_factor / self.m) * np.insert(self.weights_matrix[idx][:, 1:], 0, values=np.zeros(len(self.weights_matrix[idx])), axis=1))
            grads.append(grad)

        #TODO: roll/unroll thetas into single vector...(and with that, add advanced minimize functions like Adam)
        return J, grads

    def __cost_function(self, last_activation_values, y):
        return np.multiply(y, last_activation_values) + (np.multiply((1-y), np.log(1-last_activation_values)))

    def __are_x_y_valid(self, X, y):
        same_amt_rows = X.shape[0] == y.shape[0]
        return same_amt_rows and len(X.shape) == 2 and len(y.shape) == 2

    def __add_bias(self, arr, size):
        return np.insert(arr, 0, values=np.ones(size), axis=1)

    def __init_weights_matrix(self):

        if self.__weights_inited:
            return self.weights_matrix

        weights_matrix = []
        for layer_idx in range(len(self.layers)):
            if layer_idx == 0:
                weight_mat = np.random.random((self.layers[layer_idx].units, self.layers[layer_idx].input_shape[layer_idx] + 1))
            else:
                weight_mat = np.random.random((self.layers[layer_idx].units, self.layers[layer_idx-1].units + 1))
            
            weights_matrix.append(weight_mat)
        
        return weights_matrix

    def __is_layer_valid(self, layer):
        return isinstance(layer, Dense)

    def __are_layers_valid(self, layers):
        return all(list(map(lambda layer : self.__is_layer_valid(layer), layers)))

    def __layers_as_str(self):

        types_counter = {}
        table = '_________________________________________________________________\n'
        table += 'Layer (type)                 Output Shape              Param #\n'
        table += '=================================================================\n'

        for layer_idx in range(len(self.layers)):
            layer_type = self.layers[layer_idx].get_type()

            if layer_type in types_counter.keys():
                types_counter[layer_type] += 1
            else:
                types_counter[layer_type] = 1

            table += '{}_{}({})\n'.format(layer_type.lower(), types_counter[layer_type], layer_type.capitalize())
            if layer_idx == len(self.layers) - 1:
                table += '=================================================================\n'
            else:
                table += '_________________________________________________________________\n'

        return table