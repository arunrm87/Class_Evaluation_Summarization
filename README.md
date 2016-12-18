# Class Evaluation Summarization

Problem Description
-------------------
Student course feedback is generated daily in both classrooms
and online course discussion forums. Traditionally, instructors
manually analyze these responses in a costly manner. In
this implementation, we summarize student course feedback based
on the integer linear programming (ILP) framework.

Implementaion Details
---------------------
	- This is an unsupervised training method and so, no training set is needed.
	
	- Run summarizer.py <input_file> <summary_output_file> <algorithm(1/2)>
	<input_file> contains a set of sentences which need to be summarized.
	<summary_output_file> will contain the summary of the course feedback.
	If <algorithm> is 1, we use ILP.
	If <algorithm> is 2, we use ILP + Soft-Impute.
	We have limited the maximum size of the summary to 200 characters.

External dependencies
---------------------
Python modules: numpy, scikit-learn, pulp

Tested on dataset
-----------------
    - Obtained from http://www.coursemirror.com/download/dataset
    - The data includes both the students' responses and the standard summaries 
    created by a teaching assistant.
   
   (Check the report.pdf file for algorithm working and test results)

Authors
-------
1) Arun Ramachandran
2) Sriram Sundar
3) Swaminathan Sivaraman

This project was developed as part of the CSE 537 (Artificial Intelligence)
final course project at Stony Brook University.
