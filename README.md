## Overview

Classification accuracy as well as the precision of the probability estimates of
a naive bayes classifier can be improved by combining boostrapped naive bayes
classifiers using random features into a single ensemble.

My approach to an ensemble of discrete naive bayes closely follows the general idea of Breiman’s random forest[1]. Each base classifier is a discrete naive bayes implemented via Scikit-Learn[2]. The base classifiers are trained on a bootstrapped samples of the data, and random subsets of features. To induce diversity, continuous variables are binned into random partition lengths. The maximum number of bins is treated as a hyper-parameter. I've had success using bin sizes of 70, but results may be pretty similar for any number of bins over 50 (depending on size of data). A naive bayes classifier with bin sizes this large would lead to overfitting, but in an ensemble this is analogous to random forest’s use of unpruned decision trees.

Finally, the hyperparameters of the random naive bayes model can be selected based on accuracy, AUC, or exponential loss. I experimented with adaptations to exponential loss such as the P-Norm Push described in Rudin[3], but ultimately decided on setting p = 1 which reduces to the objective function that is also used in the RankBoost algorithm[19].

[1] Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[2] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duch- esnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

[3] Cynthia Rudin. The p-norm push: A simple convex ranking algorithm that concentrates at the top of the list. J. Mach. Learn. Res., 10:2233–2271, December 2009.

[4] Yoav Freund, Raj Iyer, Robert E. Schapire, and Yoram Singer. An efficient boosting algorithm for combining preferences. J. Mach. Learn. Res., 4:933–969, December 2003.
