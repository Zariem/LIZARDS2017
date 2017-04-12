
import numpy
import math
import sklearn

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn import neighbors


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
	bla = numpy.reshape(bla, [len(bla), 15])
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

scorer = make_scorer(accuracy)

#classifier = OneVsRestClassifier(LinearSVC(random_state=0)) #score = [0.60891089 0.565 0.69 0.63819095 0.66834171]
#classifier = OneVsOneClassifier(LinearSVC(random_state=0)) #score = [0.72772277 0.625 0.68 0.77889447 0.73366834]
#classifier = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0) #score = [0.65346535 0.68 0.71 0.70854271 0.63316583]

n_neighbors = 15
#classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform') #score = [0.85643564  0.8 0.81 0.84924623 0.79899497]
classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance') #score = [0.87128713 0.805 0.81 0.85929648 0.81407035]

scores = cross_val_score(classifier, xsTrain, ys, cv=5, scoring=scorer)
print(scores) #(higher is better)


classifier.fit(xsTrain, ys)
ysTest = classifier.predict(dataTest)

#print(ysTest)

write_to_csv(ysTest, "task2_out")
