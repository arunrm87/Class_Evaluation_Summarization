"""
Module to solve the required Linear Programming problems invloved here
"""
from __future__ import print_function
from   pulp  import (LpProblem, LpMaximize, LpVariable, LpInteger,
                     lpSum)
import numpy as np

def ilp_solve(co_occur_matrix, weights, lengths, max_length=None, concepts_discrete=True):
    """
    co_occur_matrix  : A, Co-occurence matrix of dimensions NxM
    weights          : w, List of length N of Weight values (float)
    lengths          : l, List of length M of sentence lengths (int)
    max_length       : L, If set to some number, max. length cannot be greater than L
    concepts_discrete: Set to True if z should be discrete in [0,1] and False
                      if z should be continuous
    
    # Change this to return only 'y' later
    Returns a dict with 'y' and 'z' as keys. 'y' contains a list (size M) of 0s and 1s
    to say if a sentence should be included or not. 'z' the same but for concepts
    from 1 to N. 
    """
    # Smaller variable names
    A = co_occur_matrix
    w = weights
    l = lengths

    # N represents concepts, M represents sentences 
    N = len(A)
    M = len(A[0])
    L = max_length

    if len(w) != N:
        raise ValueError("weights not the same dimension as A[rows]")
    if L and len(l) != M:
        raise ValueError("lengths not the same dimension as A[cols]")

    # Target variables
    # y[i] decides if sentence[i] is included or not. Only Binary values allowed
    y = [LpVariable("y%s"%i, 0, 1, LpInteger) for i in range(M)]
    # z[i] decides if concept[j] is included or not.
    args = [LpInteger] if concepts_discrete else []
    z = [LpVariable("z%s"%i, 0, 1, *args) for i in range(N)]

    # Objective function
    problem  = LpProblem("Class Evaluation Summary", LpMaximize)
    problem += lpSum([w[i]*z[i] for i in range(N)]), "Obj. function"
    #for i in range(N):
    #    problem += w[i]*z[i]

    # Constraints
    for i in range(N):
        problem += lpSum([A[i][j]*y[j] for j in range(M)]) >= z[i], "z cond 1 (i) -> sum_{j=0_to_M-1}(A[%s][j]y[j] >= z[%s]" % (i,i)
        for j in range(M):
            problem += A[i][j] <= z[i], "z cond 2 (i,j) -> A[%s][%s] <= z[%s]" % (i,j,i)

    if L:
        problem += lpSum([l[j]*y[j] for j in range(M)]) <= L, "length constraint"

    # Solve
    problem.solve()

    # Assign and return
    res_y = [0 for i in range(M)]
    res_z = [0 for i in range(N)]
    for var in problem.variables():
        char, index = list(var.name)
        if char == 'y':
            res_y[int(index)] = var.varValue
        elif char == 'z':
            res_z[int(index)] = var.varValue
        # DEBUG
        print(var.name, '=', var.varValue)

    return {'y': res_y, 'z': res_z}

