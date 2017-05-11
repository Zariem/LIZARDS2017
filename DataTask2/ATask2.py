import numpy
import math
import sklearn

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
    result = [int(line[0]) for line in data]
    return result


def get_xsTrain(data):
    result = [line[1:] for line in data]
    result = numpy.reshape(result, [len(result), 15])
    return result


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


dataTrain = read_data()
# dataTrain = numpy.reshape(dataTrain, [1000, 16])
#  print(dataTrain)
#  dataTrain = add_averages(dataTrain)
xsTrain = get_xsTrain(dataTrain)
ys = get_ys(dataTrain)
dataTest = read_data(filepathTest)

scorer = make_scorer(accuracy)


# classifier = OneVsRestClassifier(LinearSVC(random_state=0)) #score = 0.634088710881
# classifier = OneVsOneClassifier(LinearSVC(random_state=0)) #score = 0.70905711727
# classifier = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0) #score = 0.67703477785

def reduce_data(data, columns):
    result = []
    # print(len(data))
    for line in data:
        newline = []
        for col in columns:
            if col >= len(line):
                # print(col, len(line), line)
                continue
            newline.append(line[col])
        result.append(newline)
    return result


n_neighbors = 15
# classifier = KNeighborsClassifier(n_neighbors, weights='uniform') #score = 0.822935369919
# classifier = KNeighborsClassifier(n_neighbors, weights='distance') #score = 0.831930792577


# classifier = DecisionTreeClassifier() #score = 0.792934026568
# classifier = RandomForestClassifier(n_estimators=10) #score = 0.828964923628

# classifier = AdaBoostClassifier() #score = 0.728927210309
# classifier = GradientBoostingClassifier() #score = 0.83893551918

# classifier = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=None) #score=0.769988108861
# classifier = QuadraticDiscriminantAnalysis(reg_param=0.33) #score = 0.844975272402

# classifier = svm.SVC(decision_function_shape='ovo') #score = 0.6
# classifier = svm.LinearSVC() #score = 0.65
# classifier = svm.LinearSVC(multi_class='crammer_singer') #score = 075, 0.839 voting

# classifier = BaggingClassifier() #score = 0.81, 0.85 voting

# columns = [1,3,8,13] # most important features
columns = [0, 1, 2, 3, 4, 5, 9, 10]  # most important features
# columns = [0, 1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 14]  # without redundant
# columns = range(15)
xsTrain = reduce_data(xsTrain, columns)
dataTest = reduce_data(dataTest, columns)

# using a voting classifier
clf1 = KNeighborsClassifier(n_neighbors, weights='distance')  # score = 0.831930792577
clf2 = GradientBoostingClassifier()  # score = 0.83893551918
clf2_5 = GradientBoostingClassifier(learning_rate=0.55)  # score = 0.84699049705
clf3 = QuadraticDiscriminantAnalysis(reg_param=0.33)  # score = 0.844975272402
clf3_5 = QuadraticDiscriminantAnalysis(reg_param=0.0)  # score = 0.857060102493
clf4 = svm.SVC();

clf5 = BaggingClassifier(n_estimators=10)  # 0.81, 0.85 vote
scores = cross_val_score(clf5, xsTrain, ys, cv=10, scoring=scorer)
print(average(scores))
clf6 = RandomForestClassifier(n_estimators=15)  # 0.83
scores = cross_val_score(clf6, xsTrain, ys, cv=10, scoring=scorer)
print(average(scores))

# classifier = VotingClassifier(estimators=[('knn', clf1), ('gbc', clf2), ('qda', clf3)], voting='hard') #score = 0.85495104234
classifier = VotingClassifier(
    estimators=[('gbc2', clf2_5), ('qda2', clf3_5), ('svc', clf4),
                ('bag', clf5), ('for', clf6)],
    voting='hard')  # score = 0.856980745311

# classifier = GradientBoostingClassifier(learning_rate=0.55) #score = 0.84699049705 -> but replacing clf2 scores only 0.845930941838 on voting classifier
# classifier = QuadraticDiscriminantAnalysis(reg_param=0.0) #score = 0.857060102493 -> but replacing clf3 scores only 0.847995522165 on voting classifier...



scores = cross_val_score(classifier, xsTrain, ys, cv=10, scoring=scorer)

print(average(scores))  # (higher is better)
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

# print(ysTest)

write_to_csv(ysTest, "Atask2_out")
