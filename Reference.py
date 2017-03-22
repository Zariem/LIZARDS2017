#!/usr/bin/env python

import nibabel as nib
import os
import numpy as np
import csv
import math

from importlib import reload

from sklearn.metrics import log_loss, make_scorer
score_log_loss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
from sklearn import ensemble, svm, cross_validation
from sklearn.model_selection import cross_val_score

from sklearn import linear_model
from src.caching import cache, load_cached
from src.util import write_to_csv, read_ys, write_ass3
from src.datasets import datasets
from evolutionary_search import EvolutionaryAlgorithmSearchCV

import src.features as f

DIR = "data/set_train"
DIR_TEST = "data/set_test"


def preprocess(dir, prefix, extract, filerange):
    result = []
    print("Preprocessing")
    #for i in range(1, 10):
    for i in range(filerange[0]+1, filerange[1]+1):
        file = os.path.join(dir, prefix + "_" + str(i) + ".nii")
        print("Processing file %s" % i)
        result.append(extract(nib.load(file)))

    return result


def train(data, ys):
    #reg = svm.SVC(probability=True, gamma = 3.1622776601683792e-08, kernel = 'rbf', C = 5623413.2519034911) # opt for rand_forest + extr_single_proj
    reg = linear_model.LinearRegression()
    reg.fit(np.array(data), ys)
    return reg

def predict(reg, xs):
    return reg.predict(xs)
    #probas = reg.predict_proba(xs)
    #probas = np.transpose(probas)[1]
    #return np.clip(probas, 0.001, 0.999)

def test(data, ys):
    reg = train(data[0:150], ys[0:150])
    ys_pred = predict(reg, data[151:278])

    print("log loss:")
    print(log_loss(ys[151:278], ys_pred))

def test_cross_val(data, ys, reg):
    score = cross_val_score(reg, data, ys, scoring=score_log_loss, cv=cross_validation.StratifiedKFold(ys, n_folds=10))
    print("std: ", score.std())
    print("mean: ", score.mean())
    print(score)


def run(data, ys, data_test):
    write_to_csv(run_return(data, ys, data_test), "final_sub")

def run_return(data, ys, data_test):
    reg = train(data, ys)
    return predict(reg, data_test)

feature_sets = {
    "default": [
        [f.experiment, [], {}],
        [f.histogram, [0, 500, 20], {}],
        [f.extract_features_single_proj, [], {}],
    ],
}

def find_good_SVM_parameters(data, ys):
    paramgrid = {"kernel": ["rbf"],
                 "C"     : np.logspace(-9, 9, num=25, base=10),
                 "gamma" : np.logspace(-9, 9, num=25, base=10)}
    cv = EvolutionaryAlgorithmSearchCV(estimator=svm.SVC(probability=True),
                                       params=paramgrid,
                                       scoring=score_log_loss,
                                       cv=cross_validation.StratifiedKFold(ys, n_folds=4),
                                       verbose=1,
                                       population_size=50,
                                       gene_mutation_prob=0.10,
                                       gene_crossover_prob=0.5,
                                       tournament_size=3,
                                       generations_number=20,
                                       n_jobs=4)
    cv.fit(data, ys)

def assignment_3():
    X_train = load_cached(feature_sets["default"], "train")
    X_test = load_cached(feature_sets["default"], "test")
    genders, ages, healths = read_ys(assignment=3)
    #find_good_SVM_parameters(X_train, Y)
    #test(X_train, Y)
    #test_cross_val(X_train, Y, svm.SVC(probability=True, gamma = 3.1622776601683792e-08, kernel = 'rbf', C = 5623413.2519034911))
    gender_predictions = run_return(X_train, genders, X_test)
    age_predictions = run_return(X_train, ages, X_test)
    health_predictions = run_return(X_train, healths, X_test)

    values = []
    for brainIndex, (gender, age, health) in enumerate(zip(gender_predictions, age_predictions, health_predictions)):
        values.append((brainIndex, "gender", (gender >= 0.5)))
        values.append((brainIndex, "age", (age >= 0.5)))
        values.append((brainIndex, "health", (health >= 0.5)))

    write_ass3(values, "output")


def assignment_2():
    X_train = load_cached(feature_sets["default"], "train")
    X_test = load_cached(feature_sets["default"], "test")
    Y = read_ys(assignment=2)
    #find_good_SVM_parameters(X_train, Y)
    #test(X_train, Y)
    #test_cross_val(X_train, Y, svm.SVC(probability=True, gamma = 3.1622776601683792e-08, kernel = 'rbf', C = 5623413.2519034911))
    run(X_train, Y, X_test)
    

if __name__ == "__main__":
    assignment_3()
