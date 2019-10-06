from numpy import append, array, transpose
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
    print("\nRandomly generated weights: ")
    print(network.weights)

    # Print some useful output
    print("\nEnter training data, line by line. Type EOF to finish.")

    # Collect the training data
    line  = None
    index = 0
    while line != "EOF":
        # Get the line of input
        line = str(input("Line {}: ".format(index)))

        # Append the data to our inputs variable
        if line != "EOF":
            inputs.append(list(map(float, line.split(" "))))

        # And increment our line counter
        index += 1

    # Reset line for next time we use it
    line = None

    # Make sure that inputs is an array
    inputs = array(inputs)

    # Collect the training results
    outputs.append(list(map(float, str(input("Results: ")).split(" "))))
    outputs = transpose(outputs)

    # Use these values to train the network
    network.train(inputs, outputs, 15000)

    # Show what our weights are
    print("\nTrained weights: ")
    print(network.weights)

    # Get user input in a loop
    while line != "EOF":
        # Get the line of input
        line = str(input("\nEnter user data: "))

        # Perform the calculation
        if line != "EOF":
            # Calculate the output
            output = network.think(array([float(x) for x in line.split(" ")]))

            # Show the user the output
            print("New output data: {}".format(output))

            # Check the data's correctness
            if str(input("Does this look correct? [Yn]")).lower() == 'y':
                # If it is correct, add it to the network, and recalculate
                inputs = append(inputs, list([[float(x) for x in line.split(" ")]]), axis = 0)

                # Add the result to the output
                outputs = append(outputs, [output], axis = 0)

                # Use these values to train the network
                network.train(inputs, outputs, 15000)

                # Show what our weights are
                print("\nTrained weights: ")
                print(network.weights)
