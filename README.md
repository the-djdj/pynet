# PyNet
A simple neural network implemented in Python.

## Installation
PyNet uses `anaconda` for managing dependencies. To install this project's environment, use:
```shell
conda env create -f environment.yml
```

Once the environment has been set up, it can be activated thus:
```shell
conda activate env
```

When you're finished, deactivate it using:
```shell
conda deactivate
```

## Usage
PyNet is bundled as a simple library for calculating library. The basis for this is a `Neuron` object, which can be created thus:
```Python
network = Neuron(n_data_points_per_entry)
```
When you've collected some input and output data, you can make train the neural network:
```Python
network.train(inputs, outputs, iterations)
```
and get our results based on a new bit of data
```Python
network.think(data)
```

## Data
PyNet requires that all data be in an n-dimensional numpy array, and n cannot change during the run of the network. An example of properly formatted data would look like this:
```Python
# n = 2
inputs = [[0 0]
          [0 1]
          [1 0]
          [1 1]]
          
outputs = [[0]
           [1]
           [1]
           [0]]
```
The outputs take the form of a vector, also as a numpy array, and each value must correlate to the row in inputs that it represents.


## Demo
The `main.py` file contains a handy demo on how to properly use PyNet. It is pretty well documented, and allows for data to be fetched and results assessed.
