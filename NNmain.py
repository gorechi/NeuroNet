#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import scipy.special
import matplotlib.pyplot as plt

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

# Константы
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2

# Создаем сеть
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Загружаем тренировочные данные
data_file = open('mnist_train.csv', 'r')
trainig_data_list = data_file.readlines()
data_file.close()

# Загружаем тестовые данные
data_file = open('mnist_test.csv', 'r')
test_data_list = data_file.readlines()
data_file.close()


for record in trainig_data_list:
    all_values = record.split(',')
    inputs = numpy.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets [int(all_values[0])] = 0.99
    n.train(inputs, targets)


all_values = test_data_list[4].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()
data_set = numpy.asfarray(all_values[1:])/255*0.99-0.01
scorecard = n.test(test_data_list)
print (scorecard)
scorecard_array = numpy.asarray(scorecard)
print('Эффективность: ', scorecard_array.sum()/scorecard_array.size)

#n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#print(n.query([1,0.5,-1.5]))