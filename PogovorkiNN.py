#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import scipy.special
import matplotlib.pyplot as plt
import pickle
from test import *
import random

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        pass

    def activation_function (self, x):
        return scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # Преобразуем список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        # Преобразуем список целевых значений в двухмерный массив
        targets = numpy.array(targets_list, ndmin=2).T
        # Рассчитываем исходящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # Рассчитываем исходящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # Рассчитываем значение ошибки на выходе
        output_errors = targets - final_outputs
        # Рассчитываем распространение ошибки по узлам скрытого слоя
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # Делаем поправку весовых коэффициентов скрытого слоя
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # Делаем поправку весовых коэффициентов входного слоя
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        return True

    def query(self, inputs_list):
        # Преобразуем список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        # Рассчитываем исходящие сигналы для скрытого слоя
        hidden_inputs = self.activation_function(numpy.dot(self.wih, inputs))
        # Рассчитываем исходящие сигналы для выходного слоя
        final_inputs = self.activation_function(numpy.dot(self.who, hidden_inputs))
        return final_inputs

    def test(self, data_set):
        scorecard = []
        for record in data_set:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            print (correct_label, 'истинный маркер')
            inputs = numpy.asfarray(all_values[1:]) / 255 * 0.99 - 0.01
            outputs = self.query(inputs)
            label = numpy.argmax(outputs)
            print(label, 'ответ сети')
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)
        return scorecard

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return True


# Константы
input_nodes = 254
hidden_nodes = 2000
output_nodes = 2
learning_rate = 0.01

def load_net(file):
    with open(file, 'rb') as f:
        net = pickle.load(f)
    return net

# Создаем сеть
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#n = load_net('n_saved.pickle')

# Читаем тренировочные данные
yesData = readfile('yes.txt', 1)
noData = readfile('no.txt', 0)
trainData = yesData + noData
random.shuffle(trainData)

#Учим сеть
for data in trainData:
    n.train(data[1:], data[0])

#Проверяем сеть
testData = str_to_data('Не замочив ног лужу не перепрыгнешь.', 1)
print (n.query(testData[1:]))
