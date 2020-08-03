import numpy as np
from enum import Enum, auto


# Each row represents a single test-case
# The last element in the row is the expected result
input_data_set = np.array(
    [[0, 0, 0],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]
)


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


class Layer:
    class LayerType(Enum):
        IN = auto()
        HIDDEN = auto()
        OUT = auto()

    def __init__(self, layer_id, layer_type, input_size, neuron_count=1, act_func=None):
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.data_input = np.empty(input_size)
        self.weight_sum = np.empty(input_size)
        if layer_type == self.LayerType.OUT and neuron_count != 1:
            raise ValueError('Output layer must have only one neuron')
        if layer_type == self.LayerType.IN:
            if act_func is not None:
                raise ValueError('Activation function cannot be defined for the input layer')
            if input_size != neuron_count:
                raise ValueError('Input size must be equal to neuron count for the input layer')
            self.weight = np.identity(input_size)
            self.bias = 0
            self.act_func = lambda x: x
        else:
            self.weight = np.random.random_sample((neuron_count, input_size))
            self.bias = np.random.random_sample()
            self.delta = np.empty(neuron_count)
            self.act_func = act_func
            print('Initialized layer', layer_type.name, layer_id, 'with weight matrix\n', self.weight,
                  '\nand bias %.4f' % self.bias)

    def compute(self, data_input):
        self.data_input = data_input
        self.weight_sum = self.weight @ data_input
        self.weight_sum += np.full_like(self.weight_sum, self.bias)
        return np.vectorize(self.act_func)(self.weight_sum)

    def compute_layer_delta(self, delta):
        if self.layer_type == self.LayerType.IN:
            raise ValueError('Input layer can not have deltas')
        self.delta = delta
        return delta @ self.weight

    def update_weights(self, learn_rate, verbose):
        grad = np.array([self.act_func(x=x, derivative=True) for x in self.weight_sum])
        d_weight = np.array([self.delta * grad]).T * self.data_input
        d_bias = np.sum(grad) * np.sum(self.delta)
        if verbose:
            print('---------------------')
            print('\t', self.layer_type.name, self.layer_id, 'input', self.data_input)
            print('\t', self.layer_type.name, self.layer_id, 'sum', self.weight_sum)
            print('\t', self.layer_type.name, self.layer_id, 'grad', grad)
            print('\t', self.layer_type.name, self.layer_id, 'delta', self.delta)
            print('\t', self.layer_type.name, self.layer_id, 'delta * grad * input\n', d_weight)
            print('\t', self.layer_type.name, self.layer_id, 'd_bias\n', d_bias)
        self.weight += learn_rate * d_weight
        self.bias += learn_rate * d_bias
        if verbose:
            print('\t', self.layer_type.name, self.layer_id, 'weight\n', self.weight)


class Neuronet:
    layers = []

    def add_layer(self, layer_id, layer_type, input_size, neuron_count=1, act_func=None):
        self.layers.append(Layer(layer_id=layer_id,
                                 layer_type=layer_type,
                                 input_size=input_size,
                                 neuron_count=neuron_count,
                                 act_func=act_func))

    # hidden_layer_neuron_count is a list of neuron counts for each hidden layer
    # e.g., [3, 4] is two hidden layers with 3 and 4 neurons
    def __init__(self, input_size, hidden_layer_neuron_count, act_func):
        self.add_layer(layer_id=0,
                       layer_type=Layer.LayerType.IN,
                       input_size=input_size,
                       neuron_count=input_size)
        next_layer_input_size = input_size
        for i in range(len(hidden_layer_neuron_count)):
            self.add_layer(layer_id=i+1,
                           layer_type=Layer.LayerType.HIDDEN,
                           input_size=next_layer_input_size,
                           neuron_count=hidden_layer_neuron_count[i],
                           act_func=act_func)
            next_layer_input_size = hidden_layer_neuron_count[i]
        self.add_layer(layer_id=len(hidden_layer_neuron_count) + 1,
                       layer_type=Layer.LayerType.OUT,
                       input_size=next_layer_input_size,
                       act_func=act_func)
        self.epoch_count = 0
        print('Neuronet created!')

    def iterate(self, input_row):
        for layer in self.layers:
            input_row = layer.compute(input_row)
        return input_row

    def backpropagate(self, delta_out):
        delta = np.array([delta_out])
        for idx in range(len(self.layers)-1, 0, -1):
            delta = self.layers[idx].compute_layer_delta(delta)

    def update_weight(self, learn_rate, verbose):
        for layer in self.layers[1:]:
            layer.update_weights(learn_rate, verbose)

    def epoch(self, data_set, learn_rate, verbose):
        if verbose:
            print('----------------------------------------')
        epoch_errors = np.empty(np.shape(data_set)[0])
        epoch_results = np.empty(np.shape(data_set)[0])
        for idx, row in enumerate(data_set):
            epoch_results[idx] = self.iterate(row[0:-1])
            epoch_errors[idx] = row[-1] - epoch_results[idx]
            self.backpropagate(epoch_errors[idx])
            self.update_weight(learn_rate, verbose)
        mse = np.square(epoch_errors).mean()
        self.epoch_count += 1
        print('<%u>' % self.epoch_count, 'MSE = %.3f,' % mse, epoch_results)
        return mse

    def learn(self, data_set, learn_rate, thresold=-1.0, max_epochs=-1, verbose=False):
        while True:
            mse = self.epoch(data_set=data_set, learn_rate=learn_rate, verbose=verbose)
            if (thresold > 0 and mse <= thresold) or (0 < max_epochs <= self.epoch_count):
                break

    def reset_epochs(self):
        self.epoch_count = 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    net = Neuronet(input_size=2,
                   hidden_layer_neuron_count=[2, 2],
                   act_func=sigmoid)
    net.learn(data_set=input_data_set,
              learn_rate=0.1,
              thresold=0.005,
              verbose=False)
