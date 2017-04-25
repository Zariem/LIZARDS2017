import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from numpy import average

def write_to_csv(values, filename):
    lines = ["Id,y\n"]
    for id, val in enumerate(values):
        lines.append("%d,%d\n" % (1000 + id, val))
    with open(filename + ".csv", "w") as text_file:
        text_file.writelines(lines)

def accuracy(real_value, prediction):
    return accuracy_score(real_value, prediction)


def add_averages(data):
    result = []
    for line in data:
        result.append(line.append(numpy.mean(line)))
    return result


#Load Data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

#Extract classes and features
ys = train.loc[:, 'y']
xs = train.loc[:, 'x1':'x100']

#Prepare scorer
scorer = make_scorer(accuracy)

#Make Classifier
classifier = RandomForestClassifier(n_estimators=15)

#Compute scores
scores = cross_val_score(classifier, xs, ys, cv=10, scoring=scorer)

#Print average result
print(average(scores))




