"""
Run using summarizer.py <input_file> <summary_output_file> <matrix_option(1/2)>

1 - Normal ILP Approach
2 - ILP + Soft-Impute Approach
"""
from __future__ import print_function
from sparse_to_dense import sparse_dense
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
	omitted_words = ["[blank]\n", "blank\n", "Blank\n", "[Blank]\n"]

	text_corpus = []	# training_corpus - information extracted from the parsed document
	bigram_freq = []
	sentences_length = []
	summary = ""

	text_file = open(file)
	# extracting corpus
	for line in text_file:
		if line not in omitted_words:
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

	return text_Transpose


def summarize():
	"""
	returns the summary of a text
	"""
	omitted_words = ["[blank]\n", "blank\n", "Blank\n", "[Blank]\n"]
	bigram_freq = []
	sentences_length = []
	summary = []
	text_corpus = []

	text_file = sys.argv[1]

	# extracting corpus
	for line in open(text_file):
		if line not in omitted_words:
			text_corpus.append(line)
			sentences_length.append(len(line))	# sentence lengths computed

	allowed_options = [1,2]
	# constructing sparse occurrence matrix
	if int(sys.argv[3]) in allowed_options:
		opt_matrix = construct_occur_matrix(text_file)
	# constructing soft-imputed sparse matrix
	if int(sys.argv[3]) == 2:
		opt_matrix = sparse_dense(opt_matrix)
	elif int(sys.argv[3]) not in allowed_options:
		raise ValueError("Options should either be 1 (BaseLine) OR 2 (SoftImpute)!")

	# calculating bigram frequencies
	n_rows, n_cols = opt_matrix.shape
	for i in range(n_rows):
		bigram_freq.append(sum(opt_matrix[i]))

	# identifying the sentences to be included in the summary
	result = ilp_solve(opt_matrix, bigram_freq, sentences_length, 200)
	
	# Summary
	print ("\nNumber of sentences selected for summarization: ", result['y'].count(1.0))

	print ("\nSummarized Text:\n")
	for val, index in zip(result['y'], range(len(text_corpus))):
		if val == 1:
			summary.append(text_corpus[index])
	
	out_file = open(sys.argv[2], 'w')
	for line in summary:
		out_file.write(line)

	out_file.close()

	return ''.join(summary)


if __name__ == '__main__':
    help_text = """
Run using summarizer.py <input_file> <summary_output_file> <matrix_option(1/2)>
    
1 - Normal ILP Approach
2 - ILP + Soft-Impute Approach
"""
    if sys.argv[1] in ('--help', '-h', '-?', '?'):
        print(help_text)
    elif len(sys.argv) != 4:
        print("Invalid option/argument")
        print(help_text)
    else:
    	print(summarize())
