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


def add_averages(data):
    result = []
    for line in data:
        result.append(line.append(np.mean(line)))
    return result


# Variables
FEATURE_NAMES = ['x' + str(k) for k in range(1, 129)]  # 'x1', 'x2', ..., 'x100'
LABEL_NAME = 'y'
COLUMN_NAMES = [LABEL_NAME] + FEATURE_NAMES  # 'y', 'x1', 'x2', ..., 'x100'
TRAIN_LOAD = 7000  # out of 8000 samples


feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_NAMES]

# Load Data
train_labeled = pd.read_hdf("data/train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("data/train_labeled.h5", "train")
test = pd.read_hdf("data/test.h5", "test")
sample = pd.read_csv("data/sample.csv")


# indices = np.random.permutation(train.shape[0])
# training_idx, test_idx = indices[:80], indices[80:]
# training, validation = train.loc[training_idx, :], train.loc[test_idx, :]


# Extract ids, classes and features
ids = sample['Id'].values
ys_labeled = train_labeled.loc[:, 'y']
xs_labeled = train_labeled.loc[:, 'x1':'x128']
xs_unlabeled = train_unlabeled.loc[:, 'x1':'x128']


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURE_NAMES}
    labels = tf.constant(np.int32(data_set[LABEL_NAME].values))
    return feature_cols, labels


# Prepare scorer
scorer = make_scorer(accuracy_score)

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
                                            n_classes=10)
print("Before fit")
classifier.fit(input_fn=lambda: input_fn(train_labeled[:TRAIN_LOAD]))
print("Done fit")
accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(train_labeled[TRAIN_LOAD:]), steps=1)["accuracy"]

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
    return test_set


# Predict / Test classifier
predictions = list(classifier.predict(input_fn=give_test))

# Write to CSV
write_to_csv(ids, predictions, "out")
