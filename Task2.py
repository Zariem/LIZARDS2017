
import numpy
import math
import sklearn

from sklearn.metrics import accuracy_score

filepath = "DataTask2/train.csv"
filepathTest = "DataTask2/test.csv"

def read_data(filename=filepath):
    data = []
    first = True
    with open(filename) as csv:
        for line in csv:
            if first:
            	first = False
            	continue
            data.append([float(x) for x in line.split(',')[1:]])
    return data

def get_ys(data):
	bla = []
	bla = [int(line[0]) for line in data]
	return bla

def get_xsTrain(data):
	bla = []
	bla = [line[1:] for line in data]
	return bla

def write_to_csv(values, filename):
    lines = ["Id,y\n"]
    for id, val in enumerate(values):
        lines.append("%d,%d\n" % (1000 + id, val))
    with open(filename + ".csv", "w") as text_file:
        text_file.writelines(lines)

def accuracy(real_value, prediction):
    return accuracy_score(real_value, prediction)


dataTrain = read_data()
xsTrain = get_xsTrain(dataTrain)
ys = get_ys(dataTrain)
dataTest = read_data(filepathTest)

print(ys)

write_to_csv(ys, "task2_out")

