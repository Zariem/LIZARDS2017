
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
from numpy import average


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

#classifier = OneVsRestClassifier(LinearSVC(random_state=0)) #score = 0.634088710881
#classifier = OneVsOneClassifier(LinearSVC(random_state=0)) #score = 0.70905711727
#classifier = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0) #score = 0.67703477785


n_neighbors = 15
#classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform') #score = 0.822935369919
classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance') #score = 0.831930792577

scores = cross_val_score(classifier, xsTrain, ys, cv=5, scoring=scorer)
print(average(scores)) #(higher is better)


classifier.fit(xsTrain, ys)
ysTest = classifier.predict(dataTest)

#print(ysTest)

write_to_csv(ysTest, "task2_out")