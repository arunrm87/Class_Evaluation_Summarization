"""
Run using bigram_vectorizer.py <text_file>"
"""
from __future__ import print_function
from lp_optimizer import ilp_solve
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import numpy as np
import sys


def main():
	# pre-defined Stop Words
	stopwords = text.ENGLISH_STOP_WORDS.union("None", "blank", "\n")

	text_corpus = []	# training_corpus - information extracted from the parsed document

	bigram_freq = []
	sentences_length = []

	text_file = open(sys.argv[1])

	# extracting corpus
	for line in text_file:
		text_corpus.append(line)
		sentences_length.append(len(line))	# sentence lengths computed

	# Creating a document-term matrix based on bigram frequencies, with the use of custom stopwords
	bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=stopwords)

	# transforming corpus to feature matrices
	text_Matrix = bigram_vectorizer.fit_transform(text_corpus).toarray()

	#print (bigram_vectorizer.get_feature_names())

	# Our current matrix is a document-term matrix, we have to convert it into a term-document matrix
														# for applying the SVD or SOFT-IMPUTE algorithms
	texts = np.array(text_Matrix)
	text_Transpose = texts.transpose()

	## Finding the rank of the term-document matrix
	print (np.linalg.matrix_rank(text_Transpose))

	# calculating bigram frequencies
	n_rows, n_cols = text_Transpose.shape
	for i in range(n_rows):
		bigram_freq.append(sum(text_Transpose[i]))

	print (ilp_solve(text_Transpose, bigram_freq, sentences_length))


if __name__ == '__main__':
	main()