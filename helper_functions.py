from __future__ import division
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random


def k_fold(y, k):
	"""
	given the target variable y (a numpy array),
	and number of folds k (int),
	this returns a list of length k sublists each containing the
	row numbers of the items in the training set
	NOTE: THIS IS STRATAFIED K-FOLD CV (i.e. classes remained balanced)
	"""
	targets = np.unique(y)
	rval = []
	for fold in range(k):
		in_train = []
		for tar in targets:
			# how many can be select from?
			num_in_this_class = len(y[y == tar])

			# how many will be selected
			num_in_training = int(round(num_in_this_class * (k-1)/k))

			# indices of those who can be selected
			in_this_class = np.where(y == tar)[0]

			# add selected indices to the list of training samples
			in_train += random.sample(in_this_class, num_in_training)
		rval.append(np.array(in_train))
	return np.array(rval)


def load_data(filepath, target_file='movies_and_scores.csv', pct=50):
	"""
	This loads and merges all the dataframes into a single dataframe

	:param filepath: YOU MUST SPECIFY LOCATION OF DATA
			(e.g. '/Users/robert/Documents/UMN/5525_ML/Project/Data/' or
				'C:\Project\Data\'
			DON'T FORGET THE \ AT THE END!!!

	:param target_file:
	:param pct: percentile on which to split movies, in each year, into binary target
	:return: a nice clean dataframe, and a list of all the columns from the NLP
	"""
	# process targets file
	targets = pd.read_csv(filepath + target_file)
	# drop all pre-1980 movies
	targets = targets[(targets.year >= 1980) & (targets.year < 2010)]
	# add binary target
	targets['dummy'] = targets.groupby('year').box_office.transform(lambda x: np.percentile(x, pct))
	targets['target'] = (targets.box_office >= targets.dummy).astype('int')
	# drop non-predictive targets
	drop_vars = ['imdbID', 'rtID', 'merge_title',
		'distributor', 'genre', 'gross', 'days',
		'dummy', 'genre_Multiple_Genres']
	targets.drop(drop_vars, axis=1, inplace=True)

	# import whether movie has a top 100 director or actor
	directors = pd.read_csv(filepath + 'movie_directors_in_top_100.csv')
	actors = pd.read_csv(filepath + 'movie_actors_in_top_100.csv')

	# clean up ratings
	ratings = pd.read_csv(filepath + 'movies_all_user_ratings.csv')
	ratings['num_ratings'] = ratings.freq_ge0_l1 + ratings.freq_ge1_l2 + ratings.freq_ge2_l3 + ratings.freq_ge3_l4 + ratings.freq_ge4_l5
	ratings['pct_rated_1'] = ratings.freq_ge0_l1 / ratings.num_ratings
	ratings['pct_rated_2'] = ratings.freq_ge1_l2 / ratings.num_ratings
	ratings['pct_rated_3'] = ratings.freq_ge2_l3 / ratings.num_ratings
	ratings['pct_rated_4'] = ratings.freq_ge3_l4 / ratings.num_ratings
	ratings['pct_rated_5'] = ratings.freq_ge4_l5 / ratings.num_ratings

	# clean review results
	reviews = pd.read_table(filepath + 'NLPBasedResults.tsv', sep='\t')
	reviews.drop(['Unnamed: 570'], axis=1, inplace=True)
	reviews_cols = reviews.columns[1:].tolist()
	reviews.rename(columns={'MovieId': 'id'}, inplace=True)

	# merge
	rval = pd.merge(targets, directors, on='id', how='left')
	rval.Directors_in_top100.fillna(0, inplace=True)

	rval = pd.merge(rval, actors, on='id', how='left')
	rval.Actors_in_top100.fillna(0, inplace=True)

	rval = pd.merge(rval, ratings, on='id', how='left')
	rating_vars = [elt for elt in ratings.columns if elt != 'id']
	for var in rating_vars:
		rval[var].fillna(0, inplace=True)

	rval = pd.merge(rval, reviews, on='id', how='left')
	for var in reviews_cols:
		rval[var].fillna(0, inplace=True)
	return rval, reviews_cols


def split(data, start_yr=1995, train_yrs=10, test_yrs=5, predictive_vars=None, target_var='target', rank_var='box_office', pca_vars=None, num_pc=10):
	"""
	Partition the data into training and testing sets.
	training set is based on start year and number of training years being used
	test set is the next test_yrs years
	this also datasets with keys for determining a movies rank (this isn't super necessary,
		but is used to create the tables that list top movies vs top predicted)

	:param data: a pandas data frame
	:param start_yr: when to start counting training years (e.g. 1980)
	:param train_yrs: how many years to train on?
	:param test_yrs: how many years to test on?
	:param predictive_vars: all predictive variables used by default,
			however this can be used to specify a subset of the variables
			that will be used
	:param target_var:
	:param rank_var: for creating the ranking data sets
	:return:
	"""
	if predictive_vars is None:
		# take all variables except keys
		predictive_vars = [elt for elt in data.columns if elt not in ['title', 'id', target_var, 'year', rank_var]]

	# split off training data
	train_end = start_yr + train_yrs
	x_train = data[predictive_vars][(data.year >= start_yr) &
				   (data.year < train_end)]

	# pca transformation on review variables
	pca = PCA(n_components=num_pc, whiten=True)
	x_train_pca = pca.fit_transform(x_train[pca_vars])
	x_train.drop(pca_vars, axis=1, inplace=True)
	pca_names = []
	for i in range(x_train_pca.shape[1]):
		pca_names.append('review_pca_' + str(i))
	# x_train_pca_df = pd.DataFrame(x_train_pca, columns=pca_names)
	x_names = x_train.columns.tolist() + pca_names
	x_train = pd.DataFrame(np.hstack((x_train, x_train_pca)), columns=x_names)


	# clean up target variable
	y_train = data[target_var][(data.year >= start_yr) &
				   (data.year < train_end)]

	# create ranking variables for summaries
	# train_rank = data[[rank_var, 'title', 'year']][(data.year >= start_yr) &
	# 			   (data.year < train_end)]
	# ranks = np.empty(len(train_rank[rank_var]), int)
	# ranks[np.argsort(-1 * train_rank[rank_var])] = np.arange(len(train_rank[rank_var])) + 1
	# train_rank['rank'] = ranks

	# split off test data
	test_start = start_yr + train_yrs
	test_end = start_yr + train_yrs + test_yrs
	x_test = data[predictive_vars][(data.year >= test_start) &
				  (data.year < test_end)]

	# apply same pca transformation to test set
	x_test_pca = pca.transform(x_test[pca_vars])
	x_test.drop(pca_vars, axis=1, inplace=True)
	# x_test_pca_df = pd.DataFrame(x_test_pca, columns=pca_names)
	x_test = pd.DataFrame(np.hstack((x_test, x_test_pca)), columns=x_names)

	# clean up targets on test set
	y_test = data[target_var][(data.year >= test_start) &
				  (data.year < test_end)]

	# clean up rankings on test set
	# test_rank = data[[rank_var, 'title', 'year']][(data.year >= test_start) &
	# 			  (data.year < test_end)]
	# ranks = np.empty(len(test_rank[rank_var]), int)
	# ranks[np.argsort(-1 * test_rank[rank_var])] = np.arange(len(test_rank[rank_var])) + 1
	# test_rank['rank'] = ranks
	# return x_train, x_test, y_train, y_test, train_rank, test_rank
	return x_train, x_test, y_train, y_test



def discretize(x, num_bins=10):
	"""
	transform input into binned features
	:param x:
	:param num_bins: how many bins should the features be partitioned into?
	:return: the data set x after all features are made discrete
	"""
	cont_vars = []
	bin_vars = []
	for v in [elt for elt in x.columns if elt not in ['id', 'target', 'year']]:
		unique_vals = np.sort(pd.unique(x[v]))
		if np.array_equal(unique_vals, np.array([0, 1])):
			bin_vars.append(v)
		else:
			cont_vars.append(v)
	newx = x[bin_vars]
	for v in cont_vars:
		bins = [np.min(x[v])]
		for b in range(num_bins - 1):
			bins.append(np.percentile(x[v], (b + 1) * 100.0 / num_bins))
		bins.append(np.max(x[v]))
		bins = np.sort(pd.unique(bins))
		newx[v] = np.digitize(x[v], bins)
	return newx