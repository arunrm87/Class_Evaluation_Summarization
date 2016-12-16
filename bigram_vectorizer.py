from __future__ import print_function
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
import numpy as np
import sys



## We don't follow the approach as in Assignment 3(using load_files), because we don't classify based on categories, so we just extract from the corresponding parsed document into a list and use it as a corpus
#training_corpus = datasets.load_files("Training", description=None, load_content=True, encoding='utf-8', decode_error='ignore')

## Preferred stopwords. The decision regarding removal of standard stopwords will be taken later
stopwords = text.ENGLISH_STOP_WORDS.union("None", "blank", "\n")

## Training_corpus, which is just information extracted from the parsed document
interest_corpus = []
learning_corpus = []
muddiest_corpus = []

bigram_freq = []
sentences_length = []
## Each summary has 3 documents, one with interesting points in the class, one with the muddiest points, and one with the points of learning.
## We may later choose to combine all 3 documents into one
interest_file = open("output1_interest")

for line in interest_file:
	interest_corpus.append(line)
	sentences_length.append(len(line))

#print (interest_corpus)
## Creating a document-term matrix based on bigram frequencies, with the use of custom stopwords
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),stop_words=stopwords)

interest_Matrix = bigram_vectorizer.fit_transform(interest_corpus).toarray()

## Our current matrix is a document-term matrix, we have to convert it into a term-document matrix for applying the SVD or SOFT-IMPUTE algorithms
interest_Transpose = np.array(interest_Matrix)
interest_Transpose = interest_Transpose.transpose()

## Finding the rank of the term-document matrix
print (np.linalg.matrix_rank(interest_Transpose))

# calculating bigram frequencies
n_rows, n_cols = interest_Transpose.shape
for i in range(n_rows):
	bigram_freq.append(sum(interest_Transpose[i]))

# ===================================================================================
muddiest_file = open("output1_muddiest")

for i in range(0,52):
	muddiest_corpus.append(muddiest_file.readline())

## Creating a document-term matrix based on bigram frequencies, with the use of custom stopwords
bigram_vectorizer = CountVectorizer(ngram_range=(1, 1),stop_words=stopwords)

muddiest_Matrix = bigram_vectorizer.fit_transform(muddiest_corpus).toarray()

## Our current matrix is a document-term matrix, we have to convert it into a term-document matrix for applying the SVD or SOFT-IMPUTE algorithms
muddiest_Transpose = np.array(muddiest_Matrix)
muddiest_Transpose = muddiest_Transpose.transpose()

## Finding the rank of the term-document matrix
print (np.linalg.matrix_rank(muddiest_Transpose))



learning_file = open("output1_learning")

for i in range(0,52):
	learning_corpus.append(learning_file.readline())

## Creating a document-term matrix based on bigram frequencies, with the use of custom stopwords
bigram_vectorizer = CountVectorizer(ngram_range=(1, 1),stop_words=stopwords)

learning_Matrix = bigram_vectorizer.fit_transform(learning_corpus).toarray()

## Our current matrix is a document-term matrix, we have to convert it into a term-document matrix for applying the SVD or SOFT-IMPUTE algorithms
learning_Transpose = np.array(learning_Matrix)
learning_Transpose = learning_Transpose.transpose()

## Finding the rank of the term-document matrix
print (np.linalg.matrix_rank(learning_Transpose))
