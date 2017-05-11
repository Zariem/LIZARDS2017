import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn import DNNClassifier


with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(c))


def write_to_csv(indices, predictions, filename):
    lines = ["Id,y\n"]
    for index in range(len(indices)):
        lines.append("%d,%d\n" % (indices[index], predictions[index]))
    with open(filename + ".csv", "w") as text_file:
        text_file.writelines(lines)


def write_data_csv(ids, data, filename):
    FEATURE_NAMES = ['x' + str(k) for k in range(1, 101)]  # 'x1', 'x2', ..., 'x100'
    LABEL_NAME = 'y'
    COLUMN_NAMES = [LABEL_NAME] + FEATURE_NAMES  # 'y', 'x1', 'x2', ..., 'x100'
    headers = ",".join(["Id"] + COLUMN_NAMES)
    data_arr = np.asarray(data)
    i = 0
    with open(filename, 'w') as file:
        file.write(headers + "\n")
        for row in data_arr:
            line = str(ids[i]) + ","
            line += ",".join([str(v) for v in row])
            line += "\n"
            i += 1
            file.write(line)


# Variables
FEATURE_NAMES = ['x' + str(k) for k in range(1, 101)]  # 'x1', 'x2', ..., 'x100'
LABEL_NAME = 'y'
COLUMN_NAMES = [LABEL_NAME] + FEATURE_NAMES  # 'y', 'x1', 'x2', ..., 'x100'

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_NAMES]

# Get ids
sample = pd.read_csv("sample.csv")
ids = sample['Id'].values

# Load Data and rewrite data into csv files
train = pd.read_hdf("train.h5", "train")
write_data_csv(ids, train, "train_as.csv")

test = pd.read_hdf("test.h5", "test")
write_data_csv(ids, test, "test_as.csv")

indices = np.random.permutation(train.shape[0])


#List of filenames
filenames = ["train_as.csv", "test_as.csv"]

# Pass files to tensorflow
filename_queue = tf.train.string_input_producer(filenames)
# Get a reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
print(key, value)

train_load = np.int32((len(indices)*0.90))
training_idx, test_idx = indices[:train_load], indices[train_load:]
training, validation = train.loc[training_idx, :], train.loc[test_idx, :]

# Extract ids, classes and features
ys = train.loc[:, 'y']
xs = train.loc[:, 'x1':'x100']
# xs_train, ys_train = xs[training_idx], ys[test_idx]
print(training['x1'].values)
print({k: tf.constant(training[k].values, shape=[training[k].size, 1]) for k in FEATURE_NAMES})


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURE_NAMES}
    labels = tf.constant(np.int32(data_set[LABEL_NAME].values))
    return feature_cols, labels


classifier = DNNClassifier(
    feature_columns=[training, validation],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
    ))

print("Before fit")
classifier.fit(input_fn=lambda: input_fn(training))
print("Done fit")

accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(validation), steps=1000)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
print("")


def give_test(test_set=test):
    return test_set


# Predict / Test classifier
predictions = list(classifier.predict(input_fn=give_test))

# Write to CSV
write_to_csv(ids, predictions, "out")
