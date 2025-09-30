import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing


class Network():
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv

        self.params = self._initialize_weights()


    def _initialize_weights(self):
        # number of neurons in each layer
        input_layer = self.sizes[0]
        hidden_layer_1 = self.sizes[1]
        hidden_layer_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # random initialization of weights
        np.random.seed(self.random_state)
        params = {
            'W1': np.random.rand(hidden_layer_1, input_layer) - 0.5,
            'W2': np.random.rand(hidden_layer_2, hidden_layer_1) - 0.5,
            'W3': np.random.rand(output_layer, hidden_layer_2) - 0.5,
        }

        return params


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.

        The method should return the output of the network.
        '''
        # First hidden layer: W1 * x + b1, then sigmoid
        z1 = np.dot(self.params['W1'], x_train)
        a1 = self.activation_func(z1)
        
        # Second hidden layer: W2 * a1 + b2, then sigmoid  
        z2 = np.dot(self.params['W2'], a1)
        a2 = self.activation_func(z2)
        
        # Output layer: W3 * a2 + b3, then softmax
        z3 = np.dot(self.params['W3'], a2)
        final_output = self.output_func(z3)
        
        # Store intermediate values for backpropagation
        # DO I REALLY NEED THIS
        output = {
            'x': x_train,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'final_output': final_output
        }
        
        return output


    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.

        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().

        '''
        # Unpack cache values
        x = output['x']
        a1 = output['a1']
        a2 = output['a2']
        z1 = output['z1']
        z2 = output['z2']
        z3 = output['z3']
        final_output = output['final_output']
        
        # Calculate output layer gradients
        # dL/dz3 = dL/da3 * da3/dz3
        dL_da3 = self.cost_func_deriv(y_train, final_output)
        da3_dz3 = self.output_func_deriv(z3)
        dz3 = dL_da3 * da3_dz3
        
        # Gradient for W3: dL/dW3 = dz3 * a2^T
        dW3 = np.outer(dz3, a2)
        
        # Backpropagate to second hidden layer
        # dL/da2 = W3^T * dz3
        da2 = np.dot(self.params['W3'].T, dz3)
        # dL/dz2 = dL/da2 * da2/dz2
        da2_dz2 = self.activation_func_deriv(z2)
        dz2 = da2 * da2_dz2
        
        # Gradient for W2: dL/dW2 = dz2 * a1^T
        dW2 = np.outer(dz2, a1)
        
        # Backpropagate to first hidden layer
        # dL/da1 = W2^T * dz2
        da1 = np.dot(self.params['W2'].T, dz2)
        # dL/dz1 = dL/da1 * da1/dz1
        da1_dz1 = self.activation_func_deriv(z1)
        dz1 = da1 * da1_dz1
        
        # Gradient for W1: dL/dW1 = dz1 * x^T
        dW1 = np.outer(dz1, x)
        
        # Return gradients dictionary
        weights_gradient = {
            'W1': dW1,
            'W2': dW2, 
            'W3': dW3
        }
        
        return weights_gradient


    def _update_weights(self, weights_gradient, learning_rate):
        '''
        TODO: Update the network weights according to stochastic gradient descent.
        '''
        # Update each weight matrix using gradient descent
        # W_new = W_old - learning_rate * gradient
        for weight_name in self.params:
            self.params[weight_name] -= learning_rate * weights_gradient[weight_name]


    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def compute_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        # Run forward pass to get network output
        output = self._forward_pass(x)
        
        # Extract the final output probabilities
        final_output = output['final_output']
        
        # Return the index of the highest probability (predicted class)
        return np.argmax(final_output)



    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                
                if cosine_annealing_lr:
                    learning_rate = cosine_annealing(self.learning_rate, 
                                                     iteration, 
                                                     len(x_train), 
                                                     self.learning_rate)
                else: 
                    learning_rate = self.learning_rate
                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                
                self._update_weights(weights_gradient, learning_rate=learning_rate)

            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
