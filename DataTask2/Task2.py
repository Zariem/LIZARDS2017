
import numpy
import math
import sklearn

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
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

def reduceData(data, columns):
    result = []
    for line in data:
        newline = []
        for col in columns:
            newline.append(line[col])
        result.append(newline)
    return result

n_neighbors = 15
#classifier = KNeighborsClassifier(n_neighbors, weights='uniform') #score = 0.822935369919
#classifier = KNeighborsClassifier(n_neighbors, weights='distance') #score = 0.831930792577


#classifier = DecisionTreeClassifier() #score = 0.792934026568
#classifier = RandomForestClassifier(n_estimators=10) #score = 0.828964923628

#classifier = AdaBoostClassifier() #score = 0.728927210309
#classifier = GradientBoostingClassifier() #score = 0.83893551918

#classifier = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=None) #score=0.769988108861
#classifier = QuadraticDiscriminantAnalysis(reg_param=0.33) #score = 0.844975272402


#using a voting classifier
clf1 = KNeighborsClassifier(n_neighbors, weights='distance') #score = 0.831930792577
clf2 = GradientBoostingClassifier() #score = 0.83893551918
clf2_5 = GradientBoostingClassifier(learning_rate=0.55) #score = 0.84699049705
clf3 = QuadraticDiscriminantAnalysis(reg_param=0.33) #score = 0.844975272402
clf3_5 = QuadraticDiscriminantAnalysis(reg_param=0.0) #score = 0.857060102493
#classifier = VotingClassifier(estimators=[('knn', clf1), ('gbc', clf2), ('qda', clf3)], voting='hard') #score = 0.85495104234
classifier = VotingClassifier(estimators=[('knn', clf1), ('gbc', clf2), ('gbc2', clf2_5), ('qda', clf3), ('qda2', clf3_5)], voting='hard') #score = 0.856980745311

#classifier = GradientBoostingClassifier(learning_rate=0.55) #score = 0.84699049705 -> but replacing clf2 scores only 0.845930941838 on voting classifier
#classifier = QuadraticDiscriminantAnalysis(reg_param=0.0) #score = 0.857060102493 -> but replacing clf3 scores only 0.847995522165 on voting classifier...

#columns = [1,3,8,13] # most important features
columns = [0,1,3,4,6,8,10,11,12,13,14] # without redundant features
#columns = range(15)
xsTrain = reduceData(xsTrain, columns)
dataTest = reduceData(dataTest, columns)

scores = cross_val_score(classifier, xsTrain, ys, cv=5, scoring=scorer)

print(average(scores)) #(higher is better)
'''
for param in range(40,60,1):
    parameter = param/100;
    print(parameter)
    classifier = GradientBoostingClassifier(learning_rate = parameter) #score = 0.83893551918
    scores = cross_val_score(classifier, xsTrain, ys, cv=5, scoring=scorer)
    print(average(scores)) #(higher is better)
'''

classifier.fit(xsTrain, ys)
ysTest = classifier.predict(dataTest)

#print(ysTest)

write_to_csv(ysTest, "task2_out")
