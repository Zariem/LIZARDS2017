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

import tensorflow as tf


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


FEATURE_NAMES = ['x' + str(k) for k in range(1, 101)]  # 'x1', 'x2', ..., 'x100'
LABEL_NAME = 'y'
COLUMN_NAMES = [LABEL_NAME] + FEATURE_NAMES  # 'y', 'x1', 'x2', ..., 'x100'

global_cutoff = 40000

input_set = pd.read_hdf("train.h5")
training_set = input_set.loc[range(global_cutoff), :]
evaluation_set = input_set.loc[range(global_cutoff, 45324), :]

testing_xs = pd.read_hdf("test.h5")

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_NAMES]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[100, 15, 40],
                                            n_classes=5)


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURE_NAMES}
    labels = tf.constant(np.int32(data_set[LABEL_NAME].values))
    return feature_cols, labels


classifier.fit(input_fn=lambda: input_fn(training_set), steps=1000)

evaluated_accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(evaluation_set), steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(evaluated_accuracy_score))

# Write to CSV
# write_to_csv(ids, predictions, "outNN")
