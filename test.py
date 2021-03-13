#!/usr/bin/python
# -*- coding: utf-8 -*-
from random import randint as dice
import csv

def readcsv (file, row):
    result = []
    with open(file, encoding='utf-8') as r_file:
        # Создаем объект reader, указываем символ-разделитель ","
        file_reader = csv.reader(r_file, delimiter=",")
        for line in file_reader:
                result.append(line[row])
        return result

def writefile(file, source):
    new_file = open(file, 'a')
    for line in source:
        new_file.write(line + '\n')
    new_file.close()


def splitfile(filename, filename1, filename2, weight):
    newfile = open(filename, 'r', encoding='utf-8')
    file1 = open(filename1, 'a')
    file2 = open(filename2, 'a')
    for line in newfile:
        if dice(1, weight) == 1:
            file2.write(line)
        else:
            file1.write(line)
    newfile.close()
    file1.close()
    file2.close()
    return True

def readfile(filename, target):
    if target not in [0, 1]:
        return False
    result = [0.01]*2
    result[target] = 0.99
    filelines = []
    with open(filename, 'r', encoding='utf-8') as newfile:
        for line in newfile:
            if len(line) < 255:
                temp = [0] * 255
                temp[0] = result
                line = line.rstrip('\n')
                for i in range (len(line)):
                    temp[i+1] = ((ord(line[i])) ** (1.0/3) / 11)
                filelines.append(temp)
    return filelines

def str_to_data(input_string, target):
    if target not in [0, 1]:
        return False
    result = [0.01]*2
    result[target] = 0.99
    line = [0]*255
    line[0] = result
    for i in range(len(input_string)):
        line[i+1] = ((ord(input_string[i])) ** (1.0 / 3) / 11)
    return line
