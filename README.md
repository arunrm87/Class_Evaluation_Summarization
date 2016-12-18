# Class_Evaluation_Summarization

Problem Description
-------------------
Student course feedback is generated daily in both classrooms
and online course discussion forums. Traditionally, instructors
manually analyze these responses in a costly manner. In
this implementation, we summarize student course feedback based
on the integer linear programming (ILP) framework.

Implementaion Details
---------------------
	- Since the proposed method in the paper is unsupervised, no training
	set is needed.
	
	- Run **bigram_vectorizer.py <input_file> <summary_output_file> <algorithm(1/2)>**
	<input_file> contains a set of sentences which need to be summarized.
	<summary_output_file> will contain the summary of the course feedback.
	If <algorithm> is 1, we use ILP.
	If <algorithm> is 2, we use ILP + Soft-Impute.
	We have limited the maximum size of the summary to 200 characters.

Dataset Overview
---------------- 
    - Obtained from http://www.coursemirror.com/download/dataset
    - The data includes both the students' responses and the standard summaries 
    created by a teaching assistant.
   
References
----------
Linear optimization function in Python - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.optimize.linprog.html
