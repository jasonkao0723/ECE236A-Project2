#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:52:18 2020

@author: hjl
"""

import numpy as np
import cvxpy as cp
import math
from random import shuffle
    

def generator(n, prob_inf, T):
    
    
    ppl = np.random.binomial(size=n, n=1, p= prob_inf)    # ppl is the population
    
    
    col_weight = math.ceil(math.log(2)*T/(n*prob_inf))
    X = np.zeros((T,n))
    X[0:col_weight,:] = 1
    idx = np.random.rand(*X.shape).argsort(0)
    X = X[idx, np.arange(X.shape[1])]
    y_temp = X @ ppl #result vector
    y = np.ones_like(y_temp)*(y_temp>=1) #test results
    
    return X,ppl, y #return population and test results

def generator_nonoverlapping(n, q, p, m, T):
    
    ppl = np.zeros(n)    # ppl is the population
    A = np.zeros((m,n)) #family structure matrix
    A[0:1,:] = 1
    idx = np.random.rand(*A.shape).argsort(0)
    A = A[idx, np.arange(A.shape[1])]
    
    inf_families = np.random.binomial(size=m, n=1, p= q)
    
    for i in range(m):
        if inf_families[i] == 1:     #check if family is infected
            indices = A[i,:] == 1    #find the family members
            binom = np.random.binomial(size=np.sum(indices),n=1, p=p)
            ppl[indices] = (ppl[indices] + binom)>0
    
    
    col_weight = math.ceil(math.log(2)*T/(n*q*p)) 
    X = np.zeros((T,n))
    X[0:col_weight,:] = 1
    idx = np.random.rand(*X.shape).argsort(0)
    X = X[idx, np.arange(X.shape[1])]
    y_temp = X @ ppl
    y = np.ones_like(y_temp)*(y_temp>=1) #test results
    
    return X, ppl, y, A   #return family structured population, family assignment vector, test results
    
def add_noise_zchannel(y, p_noisy):
    
    y_noisy = np.zeros_like(y)
    indices = y==1
    noise_mask = np.random.binomial(size=np.sum(indices),n=1, p=1-p_noisy)
    y_noisy[indices] = y[indices]*noise_mask
    
    return y_noisy
    
def add_noise_bsc(y, p_noisy):
    
    y_noisy = np.zeros_like(y)
    noise_mask = np.random.binomial(size=y.shape[0],n=1, p=p_noisy)
    y_noisy = (y+noise_mask)%2
    
    return y_noisy
    

def lp(X,y):

    _, n = X.shape
    
    # cp
    z = cp.Variable(n)
    objective = cp.Minimize(cp.sum(z))
    constraints = [z >= 0, z <= 1]
    for t, y_t in enumerate(y):
        if y_t == 1:
            constraints.append((X[t, :] @ z) >= 1)
        else:
            constraints.append((X[t, :] @ z) == 0)
    
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    pred_s = z.value

    if z.value is None:
        pred_s = np.zeros((X.shape[1]))
    else:
        pred_s = np.round(z.value)

    return pred_s

def lp_nonoverlapping(X,y,A):

    _, n = X.shape
    
    # cp
    z = cp.Variable(n)
    

    A = A[A @ np.ones(n) > 0] # Delete empty families, i.e. families with zero members
    f = cp.Variable(A.shape[0])

    members_family = A @ np.ones(n)

    # Weights based on number of potentially infected people relative respected to family 
    b = (members_family- np.round(np.sum(X[y<1],0)>0) @ A.T )
    if (np.sum(b) > 0):

        c = 1 - b/(members_family)
        b = 1 - b/np.sum(b)
        b = b*c 
        b = b @ A

    else:
        b = np.ones(n)


    objective = cp.Minimize(cp.sum(cp.multiply(b,z))+cp.sum(f))
    constraints = [z >= 0, z <= 1, A.T @ f >= z , f >= 0, f <= 1]


    for t, y_t in enumerate(y):
        if y_t == 1:
            constraints.append((X[t, :] @ z ) >= 1)
        else:
            constraints.append((X[t, :] @ z ) == 0)
    
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if z.value is None:
        pred_s = np.zeros((X.shape[1]))
    else:
        pred_s = np.round(z.value>0.5)

    return pred_s

    
def lp_noisy_z(X,y):

    nt, n = X.shape
    
    # cp
    z = cp.Variable(n)
    sigma = cp.Variable(nt)

    objective = cp.Minimize(cp.sum(z)+cp.multiply(0.5,cp.sum(sigma)))
    constraints = [(z) >= 0,(z) <= 1, sigma >= 0]

    for t, y_t in enumerate(y):
        if y_t == 1:
            constraints.append(cp.sum(X[t, :] @ z) + sigma[t] >= 1)
            constraints.append(sigma[t] == 0)
        else:
            constraints.append((cp.sum(X[t, :] @ z) - sigma[t]) == 0)
            
    
    
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if z.value is None:
        pred_s = np.zeros((X.shape[1]))
    else:
        pred_s = np.round(z.value)

    return pred_s

    

def lp_noisy_bsc(X,y):

    nt, n = X.shape
    
    # cp
    z = cp.Variable(n)
    sigma = cp.Variable(nt)

    objective = cp.Minimize(cp.sum(z)+cp.multiply(0.5,cp.sum(sigma)))
    constraints = [(z) >= 0,(z) <= 1, sigma >= 0]

    for t, y_t in enumerate(y):
        if y_t == 1:
            constraints.append(cp.sum(X[t, :] @ z) + sigma[t] >= 1)
            constraints.append(sigma[t] <= 1)
        else:
            constraints.append((cp.sum(X[t, :] @ z) - sigma[t]) == 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()


    if z.value is None:
        pred_s = np.zeros((X.shape[1]))
        print('None')
    else:
        pred_s = np.round(z.value)

    return pred_s
    


def lp_noisy_z_nonoverlapping(X,y,A):

    nt, n = X.shape
    
    # cp
    z = cp.Variable(n)
    sigma = cp.Variable(nt)

    A = A[A @ np.ones(n) > 0] # Delete empty families, i.e. families with zero members
    f = cp.Variable(A.shape[0])

    members_family = A @ np.ones(n)

    # Weights based on number of potentially infected people relative respected to family 
    b = (members_family- np.round(np.sum(X[y<1],0)>0) @ A.T )
    if (np.sum(b) > 0):
        c = 1 - b/(members_family)
        b = 1 - b/np.sum(b)
        b = b*c
        b = b @ A
    else:
        b = np.ones(n)


    objective = cp.Minimize(cp.sum(cp.multiply(b,z))+cp.sum(f)+cp.multiply(0.5,cp.sum(sigma)))
    constraints = [(z) >= 0,(z) <= 1, sigma >= 0, A.T @ f >= z , f >= 0, f <= 1]

    for t, y_t in enumerate(y):
        if y_t == 1:
            constraints.append(cp.sum(X[t, :] @ z) + sigma[t] >= 1)
            constraints.append(sigma[t] == 0)
        else:
            constraints.append((cp.sum(X[t, :] @ z) - sigma[t]) == 0)
            
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if z.value is None:
        pred_s = np.zeros((X.shape[1]))
    else:
        pred_s = np.round(z.value>0.5)

    return pred_s

    
def lp_noisy_bsc_nonoverlapping(X,y,A):
    nt, n = X.shape
    
    # cp
    z = cp.Variable(n)
    sigma = cp.Variable(nt)

    A = A[A @ np.ones(n) > 0] # Delete empty families, i.e. families with zero members
    f = cp.Variable(A.shape[0])

    members_family = A @ np.ones(n)

    # Weights based on number of potentially infected people relative respected to family 
    b = (members_family- np.round(np.sum(X[y<1],0)>0) @ A.T )
    if (np.sum(b) > 0):
        c = 1 - b/(members_family)
        b = 1 - b/np.sum(b)
        b = (b*c)
        b = b @ A
        b = b

    else:
        b = np.ones(n)

    objective = cp.Minimize(cp.sum(cp.multiply(b,z))+cp.sum(f)+cp.multiply(0.5,cp.sum(sigma)))
    constraints = [(z) >= 0,(z) <= 1, sigma >= 0, A.T @ f >= z , f >= 0, f <= 1]

    for t, y_t in enumerate(y):
        if y_t == 1:
            constraints.append(cp.sum(X[t, :] @ z) + sigma[t] >= 1)
            constraints.append(sigma[t] <= 1)
        else:
            constraints.append((cp.sum(X[t, :] @ z) - sigma[t]) == 0)
    

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if z.value is None:
        pred_s = np.zeros((X.shape[1]))
    else:
        pred_s = np.round(z.value>0.5)

    return pred_s


def get_stats(ppl, ppl_pred):
    FP = 0
    FN = 0
    for i, p in enumerate(ppl):
        if ppl_pred[i] != 0 and ppl_pred[i] != 1:
            raise Exception("ppl_pred has invalid element")
        if p == 0 and ppl_pred[i] == 1:
            FP += 1
        elif p == 1 and ppl_pred[i] == 0:
            FN += 1
    Hamming = FP + FN
    return FP, FN, Hamming



    

if __name__ == '__main__':
    
    #Change these values according to your needs, you can also define new variables.
    n = 1000                        # size of population
    m = 50                          #number of families
    p = 0.2                         #probability of infection
    q = 0.2                         #probability of a family to be chosen as infected
    T = 200                         #number of tests 
    p_noisy = 0.1                   #test noise
    
    
    