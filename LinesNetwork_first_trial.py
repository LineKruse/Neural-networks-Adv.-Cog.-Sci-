
#Using scripts and data from Nielsen (2015) book

import mnist_loader

#Import data 
ining_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#Import network 
import network

#Define parameters in the network 
net = network.Network([784, 30, 10]) 
#784 input neurons (because the image has 784 pixels) 
#30 hidden neurons 
#10 output neurons (becuase it is trying to classify numbers, and there has to be a neuron for each possible output value)

#Train networks 
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#30 epochs 
#mini-batches of size 10 
#learning rate of 3.0 