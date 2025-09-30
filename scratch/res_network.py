import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):


    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        '''
        TODO: Initialize the class inheriting from scratch.network.Network.
        The method should check whether the residual network is properly initialized.
        '''
        # Call parent class initialization
        super().__init__(sizes, epochs, learning_rate, random_state)
        
        # Check if residual network is properly initialized
        # For residual connections to work, the hidden layers must have the same size
        if len(sizes) < 3:
            raise ValueError("Residual network requires at least 3 layers (input, hidden, output)")
        
        # Check if hidden layers have the same size for residual connection
        hidden_layers = sizes[1:-1]  # All layers except input and output
        if len(set(hidden_layers)) > 1:
            raise ValueError("For residual connections, all hidden layers must have the same size")
        
        # Store the hidden layer size for residual connections
        self.hidden_size = sizes[1]
        
        print(f"Residual network initialized with hidden layer size: {self.hidden_size}")
        
        


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''
        # First hidden layer: W1 * x + b1, then sigmoid
        z1 = np.dot(self.params['W1'], x_train)
        a1 = self.activation_func(z1)
        
        # Second hidden layer with residual connection: W2 * a1 + b2, then sigmoid + residual
        z2 = np.dot(self.params['W2'], a1)
        a2 = self.activation_func(z2) + a1  # Residual connection: add input to output
        
        # Output layer: W3 * a2 + b3, then softmax
        z3 = np.dot(self.params['W3'], a2)
        final_output = self.output_func(z3)
        
        # Store intermediate values for backpropagation
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
        The method should also account for the residual connection in the hidden layer.

        '''
        # Unpack cache values
        x = output['x']
        a1 = output['a1']
        a2 = output['a2']
        z1 = output['z1']
        z2 = output['z2']
        z3 = output['z3']
        final_output = output['final_output']
        
        # Calculate output layer gradients (same as regular network)
        # dL/dz3 = dL/da3 * da3/dz3
        dL_da3 = self.cost_func_deriv(y_train, final_output)
        da3_dz3 = self.output_func_deriv(z3)
        dz3 = dL_da3 * da3_dz3
        
        # Gradient for W3: dL/dW3 = dz3 * a2^T
        dW3 = np.outer(dz3, a2)
        
        # Backpropagate to second hidden layer (with residual connection)
        # dL/da2 = W3^T * dz3
        da2 = np.dot(self.params['W3'].T, dz3)
        
        # For residual connection: a2 = sigmoid(z2) + a1
        # So dL/da1 has TWO paths:
        # 1. Direct path: dL/da1 += da2 (from residual connection)
        # 2. Through activation: dL/da1 += W2^T * (da2 * sigmoid_deriv(z2))
        
        # Path 1: Direct residual connection
        da1_residual = da2  # Direct gradient from residual connection
        
        # Path 2: Through activation function
        da2_dz2 = self.activation_func_deriv(z2)
        dz2 = da2 * da2_dz2
        da1_activation = np.dot(self.params['W2'].T, dz2)
        
        # Total gradient for a1: sum of both paths
        da1 = da1_residual + da1_activation
        
        # Gradient for W2: dL/dW2 = dz2 * a1^T
        dW2 = np.outer(dz2, a1)
        
        # Backpropagate to first hidden layer
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