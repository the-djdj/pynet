from numpy import dot, exp, random

class Network:
    """ The neural network class. This defines the neural network and all of
        it's methods."""

    def __init__(self):
        """ The constructor. This sets a random seed for starting values, and
            creates the matrix for computing values."""
        # Create the seed for random number generation
        random.seed(1)

        # Convert the weights to a 3x1 matrix, with values in [-1, 1], mean = 0
        self.weights = 2 * random.random((3, 1)) - 1


    def sigmoid(self, x):
        """ A simple method that applies the Sigmoid function to a value."""
        return 1 / (1 + exp(-x))


    def sigmoid_derivative(self, x):
        """ A simple method that applies the derivative of a Sigmoid function to
            a value."""
        return x * (1 - x)


    def train(self, inputs, outputs, iterations):
        """ The method which allows a model to make accurate predictions by
            continually adjusting the weights of the model."""
        # Make adjustments according to the specified number of iterations
        for iteration in range(iterations):
            # Get the training data from the neuron
            output = self.think(inputs)

            # Get an error rate for back-propagation
            error = outputs - output

            # Calculate weight adjustments
            adjustments = dot(inputs.T, error * self.sigmoid_derivative(output))

            # And adjust the weights
            self.weights += adjustments


    def think(self, inputs):
        """ The method which 'thinks', by passing the input data as floats
            through the Sigmoid function."""
        # Convert the inputs to floats
        inputs = inputs.astype(float)

        # Calculate the dot product
        output = self.sigmoid(dot(inputs, self.weights))

        # And return our processed data
        return output
