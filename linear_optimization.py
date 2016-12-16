from __future__ import print_function
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
import numpy as np
import sys
from pulp import *

# Problem variable creation
problem = LpProblem("Class Evaluation Summary",LpMaximize)

# The variables correspond to bigrams with values from 0 to 1(continuous) and weights corresponding to sentence frequencies
x1=LpVariable("X1",0,1)
x2=LpVariable("X2",0,1)
x3=LpVariable("X3",0,1)
x4=LpVariable("X4",0,1)

y1 = LpVariable("Y1", 0, 1, LpInteger)
y2 = LpVariable("Y2", 0, 1, LpInteger)
# Objective function
problem += 0.01*x1 + 0.7*x2 + 0.025*x3 + 0.046*x4

# Constraints
problem += x1 >= y1
problem += x1 <= y1 + y2
problem += x2 >= y2
problem += x2 <= y2
problem += x3 >= y2
problem += x3 <= y1 + y2
problem += x4 >= y1
problem += x4 <= y1
problem += 10*y1 + 15*y2 <= 20


# Problem answer data is written to a .lp file
problem.writeLP("OPTIMIZED_SOLUTION.lp")

problem.solve()

print ("Status : ", LpStatus[problem.status])

# Each variable name with its optimum value
for var in problem.variables():
    print (var.name, " = ", var.varValue)
    
# Optimized objective function value
print ("Optimum summary value = ", value(problem.objective))

