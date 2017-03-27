
import numpy
import math
import sklearn
import scipy

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm, preprocessing

filepath = "DataTask1/train.csv"
filepathTest = "DataTask1/test.csv"

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
	bla = [line[0] for line in data]
	return bla

def get_xsTrain(data):
	bla = []
	bla = [line[1:] for line in data]
	return bla

def extendFeatures(data):
    bla = []
    for line in data:
        new_line = []
        for value in line:
            #new_line.append(value) #-28
            #new_line.append(math.log(math.fabs(value))) #-57
            #new_line.append(math.log(math.fabs(value)) * value) #-47
            #new_line.append(math.tanh(value)) #-60
            #new_line.append(math.sin(value)) #-62
            #new_line.append(1/value) #-65
            #new_line.append(math.exp(value)) #-59
            #new_line.append(math.exp(value**2.0)) #-59
            #new_line.append(math.exp(-value)) #-59
            #new_line.append(math.sqrt(math.fabs(value))) #-49
            #new_line.append(math.asinh(value)) #-52
            #new_line.append(math.cosh(value)) #-46
            #new_line.append(math.sqrt(math.exp(value))) #-29 !!!
            new_line.append(math.exp(value)**0.1) #-29 !!!
            #new_line.append(math.sqrt(math.exp(-value)))
        #new_line.append(math.fsum(line))
        bla.append(new_line)
    return bla

def transformOut(data):
    bla = numpy.log(numpy.abs(data))
    return bla

def apply_lin_reg(xTrain, yTrain, xTest):
	regressor = linear_model.LinearRegression()
	regressor.fit(xTrain, yTrain)
	predictions = regressor.predict(xTest)
	return predictions


def write_to_csv(values, filename):
    lines = ["Id,y\n"]
    for id, val in enumerate(values):
        lines.append("%d,%f\n" % (900 + id, val))
    with open(filename + ".csv", "w") as text_file:
        text_file.writelines(lines)

def error(real_value, prediction):
    return mean_squared_error(real_value, prediction)**0.5

def exp_kernel(x, y):
    return numpy.dot(numpy.exp(numpy.transpose(x),y))

dataTrain = read_data()
X = extendFeatures(get_xsTrain(dataTrain))
Y = get_ys(dataTrain)
dataTest = read_data(filepathTest)



from sklearn.covariance import EmpiricalCovariance, MinCovDet


xT = numpy.transpose(X)
'''
i = 0
for feature in xT:
    j = 0
    for feature2 in xT:
        c = scipy.stats.pearsonr(feature, feature2)
        print(str(i) + "," + str(j) + ": "+str(c))
        j += 1
    i += 1
'''
j = 0
for feature in xT:
    c = scipy.stats.pearsonr(feature, Y)
    print(str(j) + ": "+str(c))
    j += 1
