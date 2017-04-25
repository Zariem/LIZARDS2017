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
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from numpy import average


def write_to_csv(indices, predictions, filename):
    lines = ["Id,y\n"]
    for index in range(len(indices)):
        lines.append("%d,%d\n" % (indices[index], predictions[index]))
    with open(filename + ".csv", "w") as text_file:
        text_file.writelines(lines)


def accuracy(real_value, prediction):
    return accuracy_score(real_value, prediction)


def add_averages(data):
    result = []
    for line in data:
        result.append(line.append(np.mean(line)))
    return result


# Load Data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
sample = pd.read_csv("sample.csv")

# Extract ids, classes and features
ids = sample['Id'].values
ys = train.loc[:, 'y']
xs = train.loc[:, 'x1':'x100']

# Prepare scorer
scorer = make_scorer(accuracy)

# Feature Selection

# Linear
# classifier = LinearSVC(C=0.01, penalty="l1", dual=False).fit(xs, ys)
# Score: 0.80

##########################
#
# CLASSIFIERS
#
##########################

# Extra Tree
# classifier = ExtraTreesClassifier()
# 0.90

# Random Tree
# classifier = RandomForestClassifier(n_estimators=15)
# score = 0.91

# 1 vs Rest with LinearSVC
# classifier = OneVsRestClassifier(LinearSVC(random_state=0))
# score = 0.41



#################################
#                               #
# Fit Classifiers and Predict   #
#                               #
#################################


# Fit Classifier
classifier = classifier.fit(xs, ys)

# Make new model
model = SelectFromModel(classifier, prefit=True)
xs_new = model.transform(xs)

# Compute scores
scores = cross_val_score(classifier, xs_new, ys, cv=10, scoring=scorer)

# Print average result
print(average(scores))

# Predict / Test classifier
predictions = classifier.predict(test)

# Write to CSV
write_to_csv(ids, predictions, "out")
