
from sklearn import linear_model

filepath = "DataTask0/train.csv"
filepathTest = "DataTask0/test.csv"

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
	bla = [line[1:11] for line in data]
	return bla

def apply_lin_reg(xTrain, yTrain, xTest):
	regressor = linear_model.LinearRegression()
	regressor.fit(xTrain, yTrain)
	predictions = regressor.predict(xTest)
	return predictions


def write_to_csv(values, filename):
    lines = ["Id,y\n"]
    for id, val in enumerate(values):
        lines.append("%d,%f\n" % (10000 + id, val))
    with open(filename + ".csv", "w") as text_file:
        text_file.writelines(lines)

dataTrain = read_data()
xsTrain = get_xsTrain(dataTrain)
ys = get_ys(dataTrain)
dataTest = read_data(filepathTest)

values = apply_lin_reg(xsTrain,ys,dataTest)
write_to_csv(values, "out")
