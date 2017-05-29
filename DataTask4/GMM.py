import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
import math

ENABLE_LOG = False
FEATURE_NAMES = ['x' + str(k) for k in range(1,129)] # 'x1', 'x2', ..., 'x128'
LABEL_NAME = 'y'
COLUMN_NAMES = [LABEL_NAME] + FEATURE_NAMES # 'y', 'x1', 'x2', ..., 'x100'

TEST_FEATURE_NAMES = ['x' + str(k) for k in range(1,10)] # 'x1', 'x2', ..., 'x9'

def write_to_csv(indices, predictions, filename):
	lines = ["Id,y\n"]
	for index in range(len(indices)):
		lines.append("%d,%d\n" % (indices[index], predictions[index]))
	with open(filename + ".csv", "w") as text_file:
		text_file.writelines(lines)

def log(text):
	if ENABLE_LOG:
		print(text)

# Load Data
train_labeled = pd.read_hdf("data/train_labeled.h5", "train") # 9000 data points of form: y, x1, x2, ..., x128
train_unlabeled = pd.read_hdf("data/train_unlabeled.h5", "train") # 21000 data points of form: x1, x2, ..., x128

# Load training and sample set (for ids)
test = pd.read_hdf("data/test.h5", "test")
sample = pd.read_csv("data/sample.csv")

# Split labelled data in CV-Style (X sets, train on X-1, evaluate on 1)
# Takes pandas dataframes and returns X pandas dataframes
def split(data, amountOfSets=10):
	first = data.first_valid_index()
	last = data.last_valid_index()
	length = last - first + 1
	if (amountOfSets > length):
		print("error on split: chose more sets than data available")
	set_size = math.floor(length/amountOfSets)
	
	dataframes = []
	for set_index in range(0,amountOfSets):
		set_start = first + (set_size * set_index)
		set_end = first + (set_size * (set_index + 1))
		# for the last set, add the elements that are left over to it
		# so we get a slightly larger last set (instead of a smaller one)
		if (set_index == (amountOfSets - 1)):
			set_end = last + 1
		df = data.loc[range(int(set_start),int(set_end)), :]
		dataframes.append(df)
	return dataframes

# Splits a set in CV style in as many sets as given by `amountOfSets`
# returns one set for evaluation (the set given by `evalSetIndex`)
# and returns another set for training, which consists of all sets except the evaluation set.
def getTrainAndEvalData(data, amountOfSets=10, evalSetIndex=0):
	init = False
	dataframes = split(data, amountOfSets)
	if (evalSetIndex >= amountOfSets):
		print("error: evaluation set index was out of range!")
	eval_set = dataframes[evalSetIndex]
	for i in range(amountOfSets):
		if (not (i == evalSetIndex)):
			if (init == False):
				log("init was false at i = " + str(i))
				train_set = dataframes[i]
				init = True
			else:
				log("appending dataframe [" + str(i) + "] to train_set")
				train_set = train_set.append(dataframes[i])
		else:
			log("i was [" + str(i) + "] and that was the evalSetIndex")
	return [train_set, eval_set]

# returns the amount of elements that are in a given cluster
def amountOfDataInCluster(dataframe, clusterIndex):
	return float(len(dataframe[dataframe[LABEL_NAME].isin([clusterIndex])]))

# returns the percentage of elements that are in a given cluster of that data set
# this method is for hard assignments (will be used for initialisation of the pi-values)
def pi_hard(dataframe, clusterIndex):
	return ((amountOfDataInCluster(dataframe, clusterIndex)) / len(dataframe))

# returns the mean vector (mean of all features) of all data points in a given cluster of the given data set
# this method is for hard assignments (will be used for initialisation of the mu-values)
def mu_hard(dataframe, clusterIndex):
	dataInCluster = dataframe[dataframe[LABEL_NAME].isin([clusterIndex])]
	rawDataInCluster = np.matrix(dataInCluster.loc[:, FEATURE_NAMES].values)
	mean = rawDataInCluster.mean(0)
	#return np.squeeze(np.asarray(mean))
	return mean

# returns the covariance matrix of a cluster in a set of data
# this method is for hard assignments (will be used for initialisation of the cov-matrix)
def cov_hard(dataframe, clusterIndex):
	dataInCluster = dataframe[dataframe[LABEL_NAME].isin([clusterIndex])] # get all observations of one cluster
	rawDataInCluster = np.matrix(dataInCluster.loc[:, FEATURE_NAMES].values) # now we can leave the ys away and take the raw data
	n = len(rawDataInCluster) # how many data points we have in that cluster
	log(str(n) + " elements in cluster " + str(clusterIndex))
	meanMatrix = np.mean(rawDataInCluster, axis=0) # we can rely on broadcasting
	stdDeviation = rawDataInCluster - meanMatrix
	nonNormalisedCov = np.matmul(np.transpose(stdDeviation), stdDeviation)
	covarianceMatrix = 1.0/(n-1) * nonNormalisedCov
	return covarianceMatrix

# gives the probability of a data point being in the area around the given mean with given covariance
# for efficiency, this method takes the determinant and inverse of the covariance matrix instead of simply
# the covariance matrix itself (so the inverse doesn't need to be found for each data point in the cluster)
def getProbability(data_point, mean, det_of_covariance, inv_of_covariance):
	p = len(data_point) # how many features we are looking at
	factor = math.sqrt(math.pow(2*math.pi, p) * det_of_covariance)
	log("\nfactor:")
	log(factor)
	deviation = data_point - mean
	log("\ndeviation:")
	log(deviation)
	power = -0.5 * (np.matmul(deviation, np.matmul(inv_of_covariance, np.transpose(deviation))))
	power = power.item((0,0)) # because before that, power was still a numpy matrix
	return 1.0/factor * math.exp(power)

# returns the probabilities of the data points in the unlabeled dataframe to be in a given cluster - not normalised
# i.e. returns an array for which the i'th entry corresponds to the probability of observation i being in that cluster
# the cluster index is not given, but pi, mean and covariance need to belong to that cluster
# need to normalise these values later on
def gamma_raw(unlabeled_dataframe, pi, mean, covariance):
	inv_of_covariance = np.linalg.inv(covariance) # note that covariance is singular if we have less samples than features
	det_of_covariance = np.linalg.det(covariance)
	unlabeled_data = unlabeled_dataframe.values
	gamma_non_normalised = [pi * getProbability(data_point, mean, det_of_covariance, inv_of_covariance) for data_point in unlabeled_data]
	return gamma_non_normalised

# returns the probabilities of the data points of the unlabeled dataframe to be in each cluster - normalised
def gamma(unlabeled_dataframe, pis, means, covariances, eta = 0.1):
	eta_matrix = eta * np.identity((covariances[0]).shape[0])
	gammas = []
	if (len(pis) != len(means)):
		print("Error, arguments had the wrong sizes on calculating gamma!")
		return
	if (len(means) != len(covariances)):
		print("Error, arguments had the wrong sizes on calculating gamma!")
		return
	for clusterIndex in range(len(pis)):
		pi = pis[clusterIndex]
		mean = means[clusterIndex]
		cov = covariances[clusterIndex] + eta_matrix # regularisation, to make it invertible
		gammas.append(gamma_raw(unlabeled_dataframe, pi, mean, cov))
	# normalise
	sum_of_gammas = np.sum(gammas, axis=0)
	return np.divide(gammas, sum_of_gammas)

# returns the probabilistic amount of data points that are in the given cluster
# note: n_labeled is the total amount of labeled data we are considering
def pi_soft(gammas, hard_pis, n_labeled, clusterIndex):
	n_unlabeled = gammas.shape[1] # amount of unlabeled data points we are considering
	gamma = gammas[clusterIndex] # the probabilities of each data point to be in the given cluster
	hard_pi = hard_pis[clusterIndex] # the probability of a labaled data point to be in a given cluster
	return ((np.sum(gamma) + (hard_pi*n_labeled)) / (n_unlabeled + n_labeled)) # how many data points we estimate to be in given cluster

# returns the means of the soft clusters
# note: n_labeled is the total amount of labeled data we are considering
def mu_soft(unlabeled_dataframe, gammas, hard_pis, hard_means, n_labeled, clusterIndex):
	mu_labeled = hard_means[clusterIndex] # the mean of those data points
	n_unlabeled = gammas.shape[1] # the amount of unlabeled data points which we are considering
	gamma = gammas[clusterIndex] # the probabilities of these unlabeled data points to be in the given cluster

	unlabeled_data = unlabeled_dataframe.values
	
	sum_of_gammas = np.sum(gamma)
	weighted_data = np.transpose(np.multiply(gamma, np.transpose(unlabeled_data)))

	return np.divide((np.sum(weighted_data,axis=0) + (n_labeled * mu_labeled)), (sum_of_gammas + n_labeled))

# calculates the covariance matrix given the current gammas and means for the data in a 
def cov_soft(unlabeled_dataframe, labeled_dataframe, gammas, means, clusterIndex):
	mean = means[clusterIndex]
	gamma = gammas[clusterIndex]
	sum_of_gammas = np.sum(gamma)

	dataInCluster = labeled_dataframe[labeled_dataframe[LABEL_NAME].isin([clusterIndex])]
	labeled_data = dataInCluster.loc[:, FEATURE_NAMES].values
	unlabeled_data = unlabeled_dataframe.values
	n_labeled = len(labeled_data)
	log("we have " + str(n_labeled) + " labeled data points.")

	labeled_deviation = np.subtract(labeled_data, mean)
	unlabeled_deviation = np.subtract(unlabeled_data, mean)
	unlabeled_deviation_weighted = np.transpose(np.multiply(np.sqrt(gamma), np.transpose(unlabeled_deviation)))

	unlabeled_covariance = np.matmul(np.transpose(unlabeled_deviation_weighted), unlabeled_deviation_weighted)
	labeled_covariance = np.matmul(np.transpose(labeled_deviation), labeled_deviation)

	return np.divide((unlabeled_covariance + labeled_covariance), (sum_of_gammas + n_labeled - 1))

def initialiseGMM(labeled_dataframe, clusterIndexRange):
	pis = []
	means = []
	covariances = []
	for clusterIndex in clusterIndexRange:
		pis.append(pi_hard(labeled_dataframe, clusterIndex))
		means.append(mu_hard(labeled_dataframe, clusterIndex))
		covariances.append(cov_hard(labeled_dataframe, clusterIndex))
	return [pis, means, covariances]

def iterateGMM(unlabeled_dataframe, labeled_dataframe, pis, hard_pis, means, hard_means, covariances, clusterIndexRange, eta=0.1):
	# E-Step:
	gammas = gamma(unlabeled_dataframe, pis, means, covariances, eta)
	n_labeled = len(labeled_dataframe)

	new_pis = []
	new_means = []
	new_covariances = []
	# M-Step:
	for clusterIndex in clusterIndexRange:
		new_pis.append(pi_soft(gammas, hard_pis, n_labeled, clusterIndex))
		new_means.append(mu_soft(unlabeled_dataframe, gammas, hard_pis, hard_means, n_labeled, clusterIndex))
		new_covariances.append(cov_soft(unlabeled_dataframe, labeled_dataframe, gammas, means, clusterIndex))
	return [new_pis, new_means, new_covariances]

def predictGMM(eval_data, pis, means, covariances, eta=0.1):
	prediction_probabilities = gamma(eval_data, pis, means, covariances, eta)
	ys_pred = np.argmax(prediction_probabilities, axis=0) # gives the indices of the maximal elements for each data point, which are exactly their cluster indices
	return ys_pred

def semiSupervisedGMM(unlabeled_dataframe, labeled_dataframe, evaluation_dataframe, clusterIndexRange, max_iterations=1000, accuracy_cutoff=0.99, eta=0.1):
	eval_data = evaluation_dataframe.loc[:, FEATURE_NAMES]
	eval_ys = evaluation_dataframe.loc[:, LABEL_NAME]
	print("eval_ys")
	print(eval_ys)

	[hard_pis, hard_means, covariances] = initialiseGMM(labeled_dataframe, clusterIndexRange)

	pis = hard_pis
	means = hard_means
	# we can re-use covariances
	iteration = 0
	while True:
		[pis, means, covariances] = iterateGMM(unlabeled_dataframe, labeled_dataframe, pis, hard_pis, means, hard_means, covariances, clusterIndexRange, eta)

		# predict the data of evaluation_dataframe
		ys_pred = predictGMM(eval_data, pis, means, covariances, eta)
		print("ys_pred:")
		print(ys_pred)
		accuracy = accuracy_score(eval_ys, ys_pred)

		print("iteration " + str(iteration) + ": accuracy = " + str(accuracy))
		# until accuracy is high enough or max_iterations is reached
		if (iteration > max_iterations):
			break
		if (accuracy > accuracy_cutoff):
			break
		iteration = iteration + 1

	return [pis, means, covariances]

# -------------------------------------------
# ----------------- RUNNING -----------------
# -------------------------------------------

[train_full, eval_full] = getTrainAndEvalData(train_labeled,10,0)
[pis, means, covariances] = semiSupervisedGMM(train_unlabeled, train_full, eval_full, range(10))
ys_pred = predictGMM(test, pis, means, covariances)
print(ys_pred)

# -------------------------------------------
# ----------------- TESTING -----------------
# -------------------------------------------

'''
#print(train_labeled)
#print(train_labeled.loc[range(1,10), :])
#print(train_unlabeled.first_valid_index()) # how to find the first index. also use last_valid_index
df1 = train_labeled.loc[range(0,2500), :] # how to extract a range of observations
#df2 = train_labeled.loc[range(20,23), :] 
#print(df1.append(df2)) # how to append one data frame to another
[train_data, eval_data] = getTrainAndEvalData(df1,10,2)
#print(train_data)
#observations = train_data.loc[:, FEATURE_NAMES]
#print(observations) # our observations as pandas dataframe
#print(len(observations))
#print(observations.values) # our data with which we can train
#print(len(observations.values)) # how many observations we have
#print(len(observations.values[0])) #how many features we have
#labels = train_data.loc[:, LABEL_NAME]
#print(labels) # our labels as pandas dataframe
#print(labels.values) # our labels as raw data
#print(len(labels.values)) # how many labels we have
'''

'''
pi0 = pi_hard(train_full,0)
cov0 = cov_hard(train_full, 0)
mu0 = mu_hard(train_full,0)
print("\npi of cluster 0:") # it's a simple number, counting the amount of points in that cluster
print(pi0)
print("\nmu of cluster 0:") # it's a 1-dimensional matrix, mu being a row vector - note in the formulas we see mu as col vector
print(mu0)
print("\ncov of cluster 0:") # it's a symmetric square matrix of dimension 128 (we have 128 features)
print(cov0)
print(cov0.shape)

eta = 0.05
cov0_regularised = cov0 + (eta * np.identity(cov0.shape[0]))

df_unlabeled1 = train_unlabeled.loc[range(9000,9050),:]
print(df_unlabeled1)
gamma = gamma_raw(df_unlabeled1, pi0, mu0, cov0_regularised)
print(gamma)
'''

'''
m = [[1,2,3],[8,3,7]]
print(np.sum(m)) # gives 24
print(np.sum(m, axis=0)) # gives [9 5 10]
print(np.sum(m, axis=1)) # gives [6 18]

print(np.multiply(0.5, m)) # gives [[0.5 1. 1.5] [4. 1.5 3.5]]
print(np.multiply([2,1,3], m)) # gives [[2 2 9] [16 3 21]]
print(np.divide(m,[5,2,3])) # gives [[0.2 1. 1.] [1.6 1.5 2.3333333]]

'''

'''
gamma = [0.1, 0.3, 0.05, 0.5] # probabilities of data points to be in cluster k
data_unlabeled = [[1,1,1],[2,-2,2],[3,6,9],[4,5,6]]
# find the weighted data point
weighted_data = np.transpose(np.multiply(gamma, np.transpose(data_unlabeled)))
print(weighted_data) # gives [[0.1 0.1 0.1] [0.6 -0.6 0.6] [0.15 0.3 0.45] [2. 2.5 3.]]
print(np.sum(weighted_data,axis=0)) # gives [2.85 2.3 4.15]
print(len(weighted_data)) # gives 4
print(np.divide(np.sum(weighted_data, axis=0), len(weighted_data))) # gives the weighted mean [0.7125 0.575 1.0375]
print()

data_labeled = [[9,3,1], [4,3,7]]
mean = [3,5,5]

deviation_labeled = np.subtract(data_labeled, mean)
deviation_unlabeled = np.subtract(data_unlabeled, mean)
weighted_deviation_unlabeled = np.transpose(np.multiply(np.sqrt(gamma), np.transpose(deviation_unlabeled)))
print(deviation_labeled) # gives [[6 -2 -4] [1 -2 2]]
print(deviation_unlabeled) # gives [[-2 -4 -4] [-1 -7 -3] [0 1 4] [1 0 1]]
print(weighted_deviation_unlabeled)

cov_unlabeled = np.matmul(np.transpose(weighted_deviation_unlabeled), weighted_deviation_unlabeled) # gives the covariance of the unlabeled data
cov_labeled = np.matmul(np.transpose(deviation_labeled), deviation_labeled)

print(cov_unlabeled)
print(cov_labeled)

print("covariance is:")
print(np.divide((cov_unlabeled + cov_labeled), (np.sum(gamma) + len(data_labeled) - 1)))
'''

'''
eval_data = eval_full.loc[:, FEATURE_NAMES]
correct_labels = eval_full.loc[:, LABEL_NAME]

pis = []
means = []
covariances = []
for clusterIndex in range(10):
	print("Cluster " + str(clusterIndex))
	pis.append(pi_hard(train_full, clusterIndex))
	means.append(mu_hard(train_full, clusterIndex))
	covariances.append(cov_hard(train_full, clusterIndex))
gammas = gamma(eval_data, pis, means, covariances, eta=0.1)
print("pi")
print(pis) # the probability of any data point to be in a certain cluster
print(np.sum(pis)) # should be 1
print(gammas)
print(gammas.shape[0]) # gives 10 (the amount of clusters)
print(gammas.shape[1]) # gives 900 (the amount of data points that are evaluated)
print(gammas[0]) # gives the probabilities of each data point to be in cluster 0
print(np.sum(gammas[0])) # gives the probabilistic amount of unlabeled data points in that cluster

n_labeled = len(train_full)
print(n_labeled)
print(pi_soft(gammas, pis, n_labeled, 0)) # gives the probabilistic amount of labeled and unlabeled data points in cluster 0

soft_pis = []
for clusterIndex in range(10):
	print("probability for a data point to be in cluster " + str(clusterIndex) + ":")
	soft_pi = pi_soft(gammas, pis, n_labeled, clusterIndex)
	print(soft_pi)
	soft_pis.append(soft_pi)
print(np.sum(soft_pis)) # should sum up to 1
'''
'''
x =[[1, 5, 12, 41], [0, 12, -4, 8], [8, 23, 65, 1]]
print(np.argmax(x)) # gives 10, which is the index of element 65
print(np.argmax(x, axis=0)) # gives [2 2 2 0], the indices of 8, 23, 65, and 41
print(np.argmax(x, axis=1)) # gives [3 1 2], the indices of 41, 12 and 65
'''