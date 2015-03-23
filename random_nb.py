from __future__ import division
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from joblib import Parallel, delayed
from math import sqrt
import time
import itertools

from helper_functions import k_fold, discretize
from ranking_metrics import exp_loss


def select_vars(x_vars, max_feats):
	"""
	randomly select (without replacement) max_feats variables from the list x_vars
	:param x_vars: list of available features
	:param max_feats: number of features to select
	:return:
	"""
	return np.random.choice(x_vars, max_feats, replace=False)


def bootstrap(num_rows):
	"""
	return bootstrapped sample of x rows
	:param x:
	:return:
	"""
	sampled_rows = np.random.choice(num_rows, num_rows)
	return sampled_rows


def _fit_parallel(x, y, bin_sz, max_bins, max_feats, nb):
	"""
	PRIVATE Function to compute the naive bayes models in parallel
	:return:
	"""
	# bootstrap and prepare data for multinomial model
	if bin_sz == "random":
		bin_sz = np.random.randint(2, max_bins)  # 2 to max_bins bins

	# put the data in bins
	sampled_rows = bootstrap(x.shape[0])
	binned_x = discretize(x.iloc[sampled_rows, :], bin_sz)
	boot_y = y.iloc[sampled_rows]

	# choose random variables
	ran_vars = select_vars(binned_x.columns.values, max_feats)

	# fit naive bayes model
	nb = nb.fit(binned_x[ran_vars], boot_y)
	# return model, and the variables used and bin size (needed for prediction later)
	return nb, ran_vars, bin_sz


def _predict_prob_parallel(x, classifier, bin_sz, ran_vars):
	"""
	Private function for predicting (using average probability method) in parallel
	:param x:
	:param classifier:
	:param bin_sz:
	:param ran_vars:
	:return:
	"""
	binned_x = discretize(x, bin_sz)
	return classifier.predict_proba(binned_x[ran_vars])


def _predict_majority_parallel(x, classifier, bin_sz, ran_vars):
	"""
	Private function for predicting (using majority vote method) in parallel
	:param x:
	:param classifier:
	:param bin_sz:
	:param ran_vars:
	:return:
	"""
	binned_x = discretize(x, bin_sz)
	return classifier.predict(binned_x[ran_vars])


# TODO implement an option of either gaussian or multinomial nb
# TODO if gaussian model, use a mix of gaussian and multinomial
# TODO purely gaussian model
class RNB(object):
	"""
	Random Naive Bayes

	creates num_learners naive bayes models
	each model uses a subset of the available features
	the data for each model is binned into a random number of bins (unless bin_sz is an integer)
	the data can be split into at most max_bins bins
	alpha is the smoothing parameter from the sklearn model
	n_jobs determines how many cores to use
	verbose controls whether progress should be printed out
	"""

	def __init__(self, num_learners=50, max_feats="sqrt", bin_sz="random", max_bins=10, alpha=1, n_jobs=1, verbose=0):
		self.num_learners = num_learners
		self.max_feats = max_feats
		self.bin_sz = bin_sz
		self.max_bins = max_bins
		self.alpha = alpha
		self.n_jobs = n_jobs
		self.verbose = verbose

		# store the model information for the last model created
		self.feats_selected = None  # a list of the random variables used in each model
		self.classifiers = None  # a list with all the models
		self.bin_history = None  # a list with the bin sizes of all the models

		# save the results from tuning (if called)
		self.tuning_grid = None
		self.tuning_results = None

	def fit(self, x, y):
		"""
		Main part of algorithm
		:param x:
		:param y:
		:return:
		"""
		# select number of features for classifier
		if type(self.max_feats) != int:
			if self.max_feats == "half":
				self.max_feats = int(x.shape[1] / 2)
			elif self.max_feats == "all":
				self.max_feats = int(x.shape[1])
			else:  # default to square root
				self.max_feats = int(sqrt(x.shape[1]))

		# initialize classifiers
		nbs = []
		for i in range(self.num_learners):
			nbs.append(MultinomialNB(alpha=self.alpha))

		# fit the classifiers in parallel
		time_start = time.time()
		par_nb = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
			delayed(_fit_parallel)(x, y, self.bin_sz, self.max_bins, self.max_feats, nb)
			for nb in nbs)
		time_end = time.time()
		total_time = time_end - time_start
		if self.verbose > 0:
			print "Time to fit classifiers", round(total_time, 1)

		# save results for use in prediction
		classifiers, feats_selected, bin_sz = zip(*par_nb)
		self.feats_selected = feats_selected
		self.classifiers = classifiers
		self.bin_history = bin_sz
		return self

	def fit_cv(self, x, y, k_folds=10, metric='auc'):
		"""
		call fit for each of the k-folds of cross validation, bag results
		:return:
		"""
		if metric not in ['error', 'auc', 'exp']:
			metric = 'auc'
			print "no performance metric chosen (from [auc, error, exp]), defaulting to auc."
		performances = []

		splits = k_fold(y, k_folds)
		for i in xrange(k_folds):
			train_indices = splits[i]
			all_indices = xrange(x.shape[0])

			# split into test and train, normalize/bin each separately
			# (binning is automatically done within fit and predict methods)
			x_train = x.iloc[train_indices, :]
			y_train = y.iloc[train_indices]

			# fit model
			rnb = self.fit(x_train, y_train)

			# test model on held-out data
			test_indices = [elt for elt in all_indices if elt not in train_indices]  # this is really slow
			x_test = x.iloc[test_indices, :]
			y_test = y.iloc[test_indices]

			# save results for some metric
			if metric == 'error':
				result = rnb.score(x_test, y_test)
			elif metric == 'exp':
				result = exp_loss(rnb.predict_proba(x_test)[:, 1], y_test)
			else:
				result = rnb.auc_score(x_test, y_test)
			performances.append(result)
		return sum(performances) / len(performances)

	def predict_proba(self, x):
		"""
		Give probability estimates for each class
		:param x:
		:return:
		"""
		rval = np.array(Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
			delayed(_predict_prob_parallel)(x, self.classifiers[i], self.bin_history[i], self.feats_selected[i])
			for i in xrange(self.num_learners))).sum(axis=0)
		return rval / self.num_learners

	def predict_majority(self, x):
		"""
		Get probability estimates using majority vote method
		:param x:
		:return:
		"""
		rval = np.array(Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
			delayed(_predict_majority_parallel)(x, self.classifiers[i], self.bin_history[i], self.feats_selected[i])
			for i in xrange(self.num_learners))).sum(axis=0) / self.num_learners
		# return in two column format like sklearn
		return np.asarray([(1 - rval), rval]).T

	def predict(self, x, method="avg"):
		"""
		give predicted classification
		:param x:
		:return:
		"""
		if method == 'majority':
			probs = self.predict_majority(x)[:, 1]
			rval = np.where(probs >= .5, 1, 0)
		else:
			probs = self.predict_proba(x)
			rval = np.where(probs[:, 0] < probs[:, 1], 1, 0)
		return rval

	def score(self, x, y, method="avg"):
		"""
		find classification accuracy for a data set
		this follows the same style as sklearn
		:param x:
		:param y:
		:return:
		"""
		pred = self.predict(x, method=method)
		rval = np.where(pred == y, 1., 0.)
		return rval.mean()

	def error_rate(self, probs, y):
		"""
		find classification error rate of a prediction
		this saves time if the probability's have already been calculated,
		otherwise just used the 'score' method
		"""
		if len(probs.shape) == 2:
			pred = np.where(probs[:, 0] < probs[:, 1], 1, 0)
		elif len(probs.shape) == 1:
			pred = np.where(probs >= 0.5, 1, 0)
		else:
			print "error"
			return 0
		rval = np.where(pred == y, 0., 1.)
		return rval.mean()

	def auc_score(self, x, y):
		"""
		internal method for calculating auc. this is used in the tuning method to
		find the hyperparameters that maximize auc.
		:param x:
		:param y:
		:return:
		"""
		probs = self.predict_proba(x)
		fpr, tpr, thresholds = roc_curve(y, probs[:, 1], pos_label=1)
		return auc(fpr, tpr)

	def tune(self, x, y, k_folds, metric, repeats=1, max_feats_vec=7, num_learners_vec=50, max_bins_vec=10, alpha_vec=1.):
		"""
		Use cross validation method to find best tuning parameters
		A grid search with all possible combinations of max_feats_vec, num_learners_vec,
			max_bins_vec, and alpha_vec is created, and models are tested using CV with
			each of these parameter combinations

		:param x: training input
		:param y: training target
		:param k_folds: number of folds of cross-validation
		:param metric: which metric should be optimized?
				error rate ('error'), auc ('auc'), or exponential rank loss ('exp')?
		:param repeats: should cross-validation be repeated more than once (for more accurate results)?
		:param num_learners_vec: a list or integer
				specifies the number of learners to train the model on
		:param max_feats_vec: how many randomly selected features should each naive bayes use?
		:param max_bins_vec: what is the largest bin size the data should be put into?
				Note: actual bin size is chosen randomly for each model, this parameter
				just determines the biggest that the random size could be.
		:param alpha_vec: the smoothing parameter (sklearn uses 1 as default)
		:return: the best model found, trained on all the given data
		"""
		if metric not in ['error', 'auc', 'exp']:
			metric = 'auc'
			print "no performance metric chosen (from [auc, error, rank]), defaulting to auc."

		# initialize grid search
		if type(max_feats_vec) == int:
			max_feats_vec = [max_feats_vec]
		if type(num_learners_vec) == int:
			num_learners_vec = [num_learners_vec]
		if type(max_bins_vec) == int:
			max_bins_vec = [max_bins_vec]
		if type(alpha_vec) != list:
			alpha_vec = [alpha_vec]
		hyperparameters = [max_feats_vec, num_learners_vec, max_bins_vec, alpha_vec]
		grid = list(itertools.product(*hyperparameters))

		# store info on best results
		cv_performance = []
		# search over grid
		for p in grid:
			# set the hyperparameters
			self.max_feats = p[0]
			self.num_learners = p[1]
			self.max_bins = p[2]
			self.alpha = [3]

			# repeat the k-fold cv and average results (if necessary)
			cv_perf = []
			for i in xrange(repeats):
				result = self.fit_cv(x, y, k_folds=k_folds, metric=metric)
				cv_perf.append(result)
			cv_perf_avg = sum(cv_perf) / len(cv_perf)
			cv_performance.append(cv_perf_avg)
			print p[0], "features,",\
				p[1], "classifiers,",\
				p[2], "bins at most,",\
				p[3], "alpha:",\
				"Average", metric, "=", round(cv_perf_avg, 3)
		cv_performance = np.array(cv_performance)

		# extract best hyperparameters and build a model using them
		if metric == 'auc':
			# maximize auc
			best_tune = grid[np.where(np.max(cv_performance) == cv_performance)[0][0]]
		else:
			# minimize error and exp loss
			best_tune = grid[np.where(np.min(cv_performance) == cv_performance)[0][0]]
		print best_tune

		self.tuning_grid = grid
		self.tuning_results = cv_performance

		self.max_feats = best_tune[0]
		self.num_learners = best_tune[1]
		self.max_bins = best_tune[2]
		self.alpha = best_tune[3]
		best_fit = self.fit(x, y)
		return best_fit
