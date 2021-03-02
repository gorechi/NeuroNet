#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import scipy.special

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        print(self.wih)
        print(self.who)
        pass

    def train(self):
        pass

    def query(self):
        pass

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)