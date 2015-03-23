from __future__ import division
import numpy as np
from sklearn.metrics import average_precision_score


def height(neg_prob, pos_probs):
	"""
	NOT USED AT THE MOMENT
	calculate the height or a specific negative label
	use this for zero-one loss ranking
	:param neg_prob:
	:param pos_probs:
	:return:
	"""
	mis_ranked = np.where(pos_probs <= neg_prob)
	return len(mis_ranked)


def exp_loss(probs, labels, p=1, pos_label=1):
	"""
	exponential loss similar to Rudin's pnorm loss for two class classification
	:param probs: list of predict probabilities of the positive class
	:param labels: actual classification labels
	:param p: power. p=1 is rankboost
	:param pos_label: value of labels for positive class (default is 1)
	:return: objective function value
	"""
	pos_probs = probs[np.where(labels == pos_label)]
	neg_probs = probs[np.where(labels != pos_label)]
	rval = 0
	for n in neg_probs:
		r = np.sum(np.exp(-1 * (pos_probs - n)))
		rval += np.power(r, p)
	return rval


def pnorm_obj(lambda_t, classifiers, x, labels, p=1, pos_label=1):
	"""
	NOT USED AT THE MOMENT
	:param lambda_t:
	:param classifiers:
	:param x:
	:param labels:
	:param p:
	:param pos_label:
	:return:
	"""
	pos_x = x[np.where(labels == pos_label)]
	neg_x = x[np.where(labels != pos_label)]
	rval = 0
	for n in neg_x:
		m_lambda = 0
		for j, klass in enumerate(classifiers):
			m_lambda += lambda_t[j] * (klass.predict_proba(pos_x) - klass.predict_proba(n))
		r = np.sum(np.exp(-1 * m_lambda))
		rval += np.power(r, p)
	return rval


def top_k_ranks(prob, rank, k=25, order_by='actual'):
	"""
	Print a table with the top ranked movies (either based on actual ranking or prediction)
	:param prob:
	:param rank:
	:param k:
	:param order_by:
	:return:
	"""
	if order_by == 'actual':
		argsort_rank = np.argsort(rank['rank'])[0:k]
	else:
		argsort_rank = np.argsort(-1 * prob)[0:k]

	prob_rank = np.empty(len(prob), int)
	prob_rank[np.argsort(-1 * prob)] = np.arange(len(prob)) + 1

	print "  Actual Rank\t\t|\t  Predicted Rank\t\t|\t  Predicted Prob\t\t|\tBox Office\t\t|\tTitle"
	print "-" * 90
	for i in argsort_rank:
		print "\t\t", rank.iloc[i, 3], "   \t\t|\t\t   ",\
			prob_rank[i], "   \t\t|\t\t   ",\
			round(prob[i], 3), "   \t\t|\t\t   ",\
			rank.iloc[i, 0], "   \t\t|\t\t   ",\
			rank.iloc[i, 1]


def mean_avg_prec(actual, prediction, k):
	"""
	Mean average precision, that is the average precision is averaged over the first k predicted items
	:param actual:
	:param prediction:
	:param k:
	:return:
	"""
	order_by = np.argsort(-1 * prediction)
	actual_sorted = np.array(actual.iloc[order_by])
	prediction_sorted = prediction[order_by]

	rval = np.zeros(k)
	for i in xrange(k):
		if np.alltrue(actual_sorted[0:(i + 1)] == np.zeros(i + 1)):
			# bug in sklearn code!
			rval[i] = 0.
		else:
			rval[i] = average_precision_score(actual_sorted[0:(i + 1)], prediction_sorted[0:(i + 1)])
	return rval.mean()