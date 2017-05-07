import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold, train_test_split, cross_val_score
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


# Variables
FEATURE_NAMES = ['x' + str(k) for k in range(1, 101)]  # 'x1', 'x2', ..., 'x100'
LABEL_NAME = 'y'
COLUMN_NAMES = [LABEL_NAME] + FEATURE_NAMES  # 'y', 'x1', 'x2', ..., 'x100'

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_NAMES]

# Load Data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
sample = pd.read_csv("sample.csv")

# Extract ids, classes and features
ids = sample['Id'].values
ys = train.loc[:, 'y']
xs = train.loc[:, 'x1':'x100']

# Specially for tensor data
x_train, y_train, x_test, y_test = train_test_split(xs, ys, test_size=0.4, random_state=356)
train, validation = train_test_split(train, test_size=0.4, random_state=356)


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURE_NAMES}
    labels = tf.constant(np.int32(data_set[LABEL_NAME].values))
    return feature_cols, labels


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

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 5, 10],
                                            n_classes=5)

classifier.fit(input_fn=lambda: input_fn(train))

accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(validation), steps=1000)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
print("")

# Extra Tree
# classifier = ExtraTreesClassifier()
# 0.90

# Random Tree
# classifier = RandomForestClassifier(n_estimators=15)
# score = 0.91

# 1 vs Rest with LinearSVC
# classifier = OneVsRestClassifier(LinearSVC(random_state=0))
# score = 0.41

# MLP-Classifier
# classifier = MLPClassifier(hidden_layer_sizes=(10))
# score = 0.89


# Keras
# classifier = Sequential()
# classifier.add(Dense(units=100, input_dim=100))
# classifier.add(Activation('relu'))
# classifier.add(Dense(units=10))
# classifier.add(Activation('softmax'))
# classifier.compile(loss='categorical_crossentropy',
#                    optimizer='sgd',
#                    metrics=['accuracy'])

#################################
#                               #
# Fit Classifiers and Predict   #
#                               #
#################################

'''
# Fit Classifier
classifier = classifier.fit(xs, ys)

# Make new model
# model = SelectFromModel(classifier, prefit=True)
# xs_new = model.transform(xs)

# Compute scores
scores = cross_val_score(classifier, xs, ys, cv=10, scoring='accuracy_score')

# Print average result
print(scores.mean())
'''


def give_test(test_set=test):
    test_arr = np.asarray(test_set)
    return test_arr


# Predict / Test classifier
predictions = list(classifier.predict(input_fn=give_test))

# Write to CSV
write_to_csv(ids, predictions, "out")
