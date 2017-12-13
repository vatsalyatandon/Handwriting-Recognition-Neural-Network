# Handwriting Recognition Neural Network

The repository contains a dataset of handwrtitten images from MNIST and an Artificial Neural Network on python that trains from the dataset and executes the algorithm on a mini batch of 10,000 test cases repeated over 30 generations to map the accuracy. The algorithm learns using a **stochastic gradient descent** to speed up learning.

## How to Execute

Execute the following command in the Python shell.

```import mnist_loader.py
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
```
Now we'll set up a Neural Network with 30 hidden neurons.

```import network.py
net = network.Network([784,30,10])
```
Finally, we'll use stochastic gradient descent to learn from the MNIST training_data over 30 epochs, with a mini-batch size of 10, and a learning rate of η=3.0

`net.SGD(training_data, 30, 10, 3.0, test_data=test_data)`

Note - `python` libraries used for this are - 
`numpy
scikit-learn
scipy
Theano`
