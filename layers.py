from abc import ABC, abstractmethod
import activations
import numpy as np

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

    def fit(self, x_training, y_training):
        self.model.fit(x_training, y_training)
        return self

class Model:

    def __init__(self, layers):
        if not self.__are_layers_valid(layers):
            raise Exception('Use only proper layers in model creation')
        self.layers = layers

    def __str__(self):
        return self.__layers_as_str()

    def add(self, layer):
        if self.__is_layer_valid(layer):
            self.layers.append(layer)
        else:
            raise Exception('Use only proper layers in model creation')

    def compile(self):
        pass

    def fit(self, x_training, y_training):

        weights_matrix = self.__init_weights_matrix()

        # forward propagate
        # TODO: VECTORIZE !!!!!!!
        for x_no_bias in x_training:
            current_values = x_no_bias
            for layer_idx in range(len(self.layers)):
                current_values = self.__add_bias_and_vstack(current_values)
                current_values = self.layers[layer_idx].activate(np.matmul(weights_matrix[layer_idx], current_values))

        print(weights_matrix)
        print(current_values)

    def __add_bias_and_vstack(self, array):
        values = np.hstack(array)
        values = np.insert(values, 0, 1)
        return np.vstack(values)

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

    def __init_weights_matrix(self):
        weights_matrix = []
        for layer_idx in range(len(self.layers)):
            if layer_idx == 0:
                weight_mat = np.random.random((self.layers[layer_idx].units, self.layers[layer_idx].input_shape[layer_idx] + 1))
            else:
                weight_mat = np.random.random((self.layers[layer_idx].units, self.layers[layer_idx-1].units + 1))
            
            weights_matrix.append(weight_mat)
        
        return weights_matrix