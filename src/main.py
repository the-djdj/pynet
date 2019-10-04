from numpy import array, transpose
from Neural import Neuron

if __name__ == '__main__':
    """ The main method. This acts as a small demo for the neural network."""
    # Get the number of data points per element
    n = int(input("Enter the number of data points per element: "))

    # Create the neuron class
    network = Neuron(n)
    inputs  = list()
    outputs = list()

    # Show what our weights are
    print("Randomly generated weights: ")
    print(network.weights)

    # Print some useful output
    print("Enter training data, line by line. Type EOF to finish.")

    # Collect the training data
    line  = None
    index = 0
    while line != "EOF":
        # Get the line of input
        line = str(input("Line {}: ".format(index)))

        # Append the data to our inputs variable
        if line != "EOF":
            inputs.append(list(map(int, line.split(" "))))

        # And increment our line counter
        index += 1

    # Make sure that inputs is an array
    inputs = array(inputs)

    # Collect the training results
    outputs.append(list(map(int, str(input("Results: ")).split(" "))))
    outputs = transpose(outputs)

    # Use these values to train the network
    network.train(inputs, outputs, 15000)

    # Show what our weights are
    print("Trained weights: ")
    print(network.weights)

    line = str(input("Enter user data: "))

    print("New output data: ")
    print(network.think(array([ int(x) for x in line.split(" ") ])))
