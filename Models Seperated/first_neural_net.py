import numpy as np
from numpy import exp, array, random, dot

#neural network class
class NeuralNetwork():

    def __init__(self):

        #seeding the network to ensure it generates the same numbers everytime the program runs
        #usefull for debugging latter
        random.seed(1)

        #assigning values to a 3*1 matrix with a range -1 to 1 with mean 0
        self.synaptic_weights = 2 * random.random((3,1))-1

    # this is our activation function which will be used to squash data probability 0 and 1
    def sigmoid(self,x):return 1/(1 + np.exp(-x))

    def sigmoid_dirivative(self,x):
        return x*(1-x)

    #training the network 10 000 times
    def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
        for i in range(number_of_training_iterations):

            #we then assign predictions to output by calling predict function
            output = self.predict(training_set_inputs)

            #calculating error so we know how to adjust weights
            error= training_set_outputs - output

            #we then multiply the eror by input and by the gradient of our activation function(sigmoid)
            #otherwise known as Gradient Decent!
            adjustment = dot(training_set_inputs.T, error*self.sigmoid_dirivative(output))

            #adjusting the weights according to the gradient decent
            self.synaptic_weights += adjustment

    #we use the sigmoid directly in predict function  to pass parameters to our neurons
    #dot product is the result of multiplying the two matrixes
    def predict(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    #initilization of the neural network
    neural_network = NeuralNetwork()
    print('Random starting syptnaptic weights:')
    print(neural_network.synaptic_weights)

    #creating a training set of inputs and outputs
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #training the network 10000 times to make changes to the weights
    neural_network.train(training_set_inputs,training_set_outputs,10000)
    print('New schematic weights after training:')
    print(neural_network.synaptic_weights)
    print('The computer predics output should be (>0.80 means 1 and <0.50 means 0):')

    #predicting an output given a new input
    print(neural_network.predict(array([1,0,1])))