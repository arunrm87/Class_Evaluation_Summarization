#Copyright [2016] Arun Ramachandran

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

"""
Module to perform soft-impute of the sparse matrix
"""
from __future__ import print_function

import numpy as np
import copy
from sklearn.utils.extmath import randomized_svd


def check_convergence(old_matrix, new_matrix, threshold):
	"""
	old_matrix		:	A, the original matrix before reduction
	new_matrix		:	B, matrix after reduction based on Frobenius norm
	threshold		:	threshold value for finding convergence
	Returns a boolean to indicate if there is convergence
	"""
	A = old_matrix
	B = new_matrix
	difference = A - B
	change = np.sum(difference ** 2) ## Sum of squares of differences
	new = np.sqrt(change)
	old = np.sum(A ** 2) ## Sum of squares of original matrix
	old = np.sqrt(old)
	##old = np.sqrt((A ** 2).sum())
	if (old != 0):
		return ((new / old) < threshold)
	else:
		return True


def svd(A, hyperparam):
	"""
	A			:	old matrix which has to be decomposed using SVD
	hyperparam	:	Regularization parameter which prevents overfitting of data
	Returns a new matrix of dimension k(obtained after SVD), and the value k
	"""	
	(U, s, V) = np.linalg.svd(A, full_matrices=False, compute_uv=True)
	s_new = np.maximum(s - hyperparam, 0)
	k = (s_new > 0).sum()
	s_new = s_new[:k]
	U_new = U[:, :k]
	V_new = V[:k, :]
	S_new = np.diag(s_new)
	B = np.dot(U_new, np.dot(S_new, V_new))
	return B, k
	

def dense(A, hyperparam, threshold, iterations):
	"""
	A			:	original term-document matrix (a sparse matrix)
	hyperparam	:	Regularization parameter which prevents overfitting of data
	threshold	:	threshold value for finding convergence	
	iterations	:	Number of iterations before termination(if not converged)
	Returns a dense matrix, with dimensions reduced from rank(r)
	"""
	for i in range(iterations):
		B,rank = svd(A, hyperparam)
		converged = check_convergence(A, B, threshold)
		A = B
		if (converged):
			break
	print (i)
	return A


def sparse_dense(summary):
	text_copy = copy.deepcopy(summary)
## Find a suitable value for the hyperparameter, some random value like 0.5, or based
## on some heuristic like (rank of original matrix/10), or (max_singular_value of the
##original matrix / 20)
	_, s, _ = randomized_svd(summary, 1, n_iter=5)
	hyperparameter = s[0] / 50

	term_document_matrix_rank = np.linalg.matrix_rank(summary)
	iterations = int(term_document_matrix_rank / 10) 


	A_new = dense(text_copy, hyperparameter, 0.02, iterations)

	##print (len(A_new) * len(A_new[0]))
	##print (np.sum(A_new == 0))
	##print (np.sum(A_new < 0))
	##print (np.sum(summary == 0))
	return A_new


if __name__ == '__main__':
	sparse_dense(construct_occur_matrix(sys.argv[1]))
