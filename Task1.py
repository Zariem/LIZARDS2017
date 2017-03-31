
import numpy
import math
import sklearn

from sklearn import linear_model
from sklearn import kernel_ridge
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
            new_line.append(value) #-28
            #new_line.append(math.log(math.fabs(value))) #-57
            #new_line.append(math.log(math.fabs(value)) * value) #-47
            #new_line.append(math.tanh(value)) #-60
            #new_line.append(math.sin(value)) #-62
            #new_line.append(1/value) #-65
            #new_line.append(math.exp(value)) #-59
        ##new_line.append(math.exp(value*(1/2.0))) #-25
        ##new_line.append(math.exp(value*(1/3.0))) #-28
            #new_line.append(math.exp(-value)) #-59
            #new_line.append(math.sqrt(math.fabs(value))) #-49
            #new_line.append(math.asinh(value)) #-52
            #new_line.append(math.cosh(value)) #-46
        #new_line.append(math.fsum(line))
        bla.append(new_line)
    return bla

def extendFeatures2(data):
    poly = preprocessing.PolynomialFeatures(degree = 3)
    return poly.fit_transform(data)

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
xsTrain = get_xsTrain(dataTrain)
ys = get_ys(dataTrain)
dataTest = read_data(filepathTest)

xsTrainExtended = extendFeatures2(xsTrain)
dataTestExtended = extendFeatures2(dataTest)

our_scorer = make_scorer(error, greater_is_better = False)
predictors = []
'''
for x in range(10,20):
    for y in [0.03]:
        print("x: " + str(x) + ", y: " + str(y))
        #predictor = kernel_ridge.KernelRidge(alpha = y, kernel = 'poly', degree = x)
        #predictor = svm.SVR(kernel = 'poly', degree=4)
        #predictor.kernel
        predictor = linear_model.Ridge(alpha = y, normalize = True)
        #predictor = linear_model.LinearRegression()
        #predictor = AdaBoostRegressor(base_estimator = linear_model.LinearRegression(), loss = 'square')
        #predictor = AdaBoostRegressor(base_estimator = svm.SVR(kernel = 'poly', degree = x), loss = 'square')
        predictors.append(predictor)
        #scores = cross_val_score(predictor, xsTrain, ys, cv = 10, scoring = our_scorer)
        scores = cross_val_score(predictor, xsTrainExtended, ys, cv = x, scoring = our_scorer)
        print(numpy.mean(scores))
'''
#predictor = AdaBoostRegressor(base_estimator = svm.SVR(kernel = 'poly', degree=4), loss = 'square')
#predictor = kernel_ridge.KernelRidge(alpha=2.66, kernel='poly', degree = 4) #with extendFeatures: scores 18.820558609 in CrossValidation

predictor = linear_model.Ridge(alpha=0.03,normalize=True) #with extendFeatures2, degree=3: scores 18.8093282323 in CV

#model = predictor.fit(xsTrain, ys)
#values = model.predict(dataTest)

model = predictor.fit(xsTrainExtended, ys)
values = model.predict(dataTestExtended)

write_to_csv(values, "task1_ridge_polydeg3_normalized_out")
