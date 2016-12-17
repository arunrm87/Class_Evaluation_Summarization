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


def construct_occur_matrix(file):
	"""
	builds a co-occurrence matrix b/w bigrams and sentences present in the input file
	returns the built matrix
	"""
	# pre-defined Stop Words
	stopwords = text.ENGLISH_STOP_WORDS.union("None", "blank", "\n")

	text_corpus = []	# training_corpus - information extracted from the parsed document
	bigram_freq = []
	sentences_length = []
	summary = ""

	text_file = open(file)
	# extracting corpus
	for line in text_file:
		text_corpus.append(line)
		sentences_length.append(len(line))	# sentence lengths computed

	# Creating a document-term matrix based on bigram frequencies, with the use of custom stopwords
	bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=stopwords)

	# transforming corpus to feature matrices
	text_Matrix = bigram_vectorizer.fit_transform(text_corpus).toarray()

	# Our current matrix is a document-term matrix, we have to convert it into a term-document matrix
														# for applying the SVD or SOFT-IMPUTE algorithms
	texts = np.array(text_Matrix)
	text_Transpose = texts.transpose()

	## Finding the rank of the term-document matrix
	#print (np.linalg.matrix_rank(text_Transpose))

	return text_Transpose


def summarize():
	"""
	returns the summary of a text
	"""
	bigram_freq = []
	sentences_length = []
	summary = ""
	text_corpus = []

	text_file = sys.argv[1]

	# extracting corpus
	for line in open(text_file):
		text_corpus.append(line)
		sentences_length.append(len(line))	# sentence lengths computed

	# constructing occurrence matrix
	matrix = construct_occur_matrix(text_file)

	# calculating bigram frequencies
	n_rows, n_cols = matrix.shape
	for i in range(n_rows):
		bigram_freq.append(sum(matrix[i]))

	# identifying the sentences to be included in the summary
	result = ilp_solve(matrix, bigram_freq, sentences_length, 200)
	
	# Summary
	print ("\nNumber of sentences selected for summarization: ", result['y'].count(1.0))

	print ("\nSummarized Text:\n")
	for val, index in zip(result['y'], range(len(text_corpus))):
		if val == 1:
			summary += text_corpus[index]
	
	return summary

if __name__ == '__main__':
	print(summarize())