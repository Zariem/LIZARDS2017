
import numpy

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn import svm

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



dataTrain = read_data()
xsTrain = get_xsTrain(dataTrain)
ys = get_ys(dataTrain)
dataTest = read_data(filepathTest)

our_scorer = make_scorer(error, greater_is_better = False)
predictors = []

for x in range(1,10):
    print(x)
    predictor = svm.SVR(kernel = 'poly', degree=x)
    predictors.append( predictor)
    scores = cross_val_score(predictor, xsTrain, ys, cv = 10, scoring = our_scorer)
    print(numpy.mean(scores))

predictor = svm.SVR(kernel = 'poly', degree=4)

model = predictor.fit(xsTrain, ys)
values = model.predict(dataTest)
write_to_csv(values, "task1_out")
