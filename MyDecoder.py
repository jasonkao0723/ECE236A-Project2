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
            constraints.append(cp.sum((X[t, :] @ z)) >= 1)
        else:
            constraints.append(cp.sum((X[t, :] @ z))== 0)
    
    
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
            constraints.append(cp.sum((X[t, :] @ z))>= 1)
        else:
            constraints.append(cp.sum((X[t, :] @ z)) == 0)
    
    
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
            constraints.append((cp.sum(X[t, :] @ z) + sigma[t]) >= 1)
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
            constraints.append((cp.sum(X[t, :] @ z) + sigma[t]) >= 1)
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
            constraints.append((cp.sum(X[t, :] @ z) + sigma[t]) >= 1)
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
            constraints.append((cp.sum(X[t, :] @ z) + sigma[t]) >= 1)
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


def plot_z_nonoverlap(N, q, p, m, n_runs, p_noisy, test_setSize):

    Title_fig = f"Z Noise on Tests q={q} p={p} m={m}"

    linestyle = ['o-', 'o--']
    linecolor = ['tab:blue','tab:orange','tab:green' ]


    n_n = np.shape(p_noisy)[0]
    predictAcc = np.zeros((3,test_setSize.shape[0],n_n))
    predictAcc_noisy_z = np.zeros((3,test_setSize.shape[0],n_n))
    predictAcc_noisy_z_nonoverlapping = np.zeros((3,test_setSize.shape[0],n_n))

    for i_n, i_noise in enumerate(p_noisy):

        counter = 0

        for i_test in test_setSize:

            for i_run in np.arange(n_runs):
                np.random.seed(i_run)

                [X,ppl, y, A] = generator_nonoverlapping(N, q, p, m, i_test)
                
                y = add_noise_zchannel(y,i_noise)

                pred_s = lp_nonoverlapping(X,y,A.copy())
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc[0,counter,i_n] = predictAcc[0,counter,i_n]+temp_FN/np.sum(ppl)
                    
                predictAcc[1,counter,i_n] = predictAcc[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc[2,counter,i_n] = predictAcc[2,counter,i_n]+temp_Hamming/N

                pred_s = lp_noisy_z(X,y)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc_noisy_z[0,counter,i_n] = predictAcc_noisy_z[0,counter,i_n]+temp_FN/np.sum(ppl)
                    
                predictAcc_noisy_z[1,counter,i_n] = predictAcc_noisy_z[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_noisy_z[2,counter,i_n] = predictAcc_noisy_z[2,counter,i_n]+temp_Hamming/N

                pred_s = lp_noisy_z_nonoverlapping(X,y,A.copy())
                temp_FP,temp_FN,temp_Hamming =  get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc_noisy_z_nonoverlapping[0,counter,i_n] = predictAcc_noisy_z_nonoverlapping[0,counter,i_n]+temp_FN/np.sum(ppl)
                
                predictAcc_noisy_z_nonoverlapping[1,counter,i_n] = predictAcc_noisy_z_nonoverlapping[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_noisy_z_nonoverlapping[2,counter,i_n] = predictAcc_noisy_z_nonoverlapping[2,counter,i_n]+temp_Hamming/N

            counter += 1
            
            
            
    predictAcc = predictAcc/n_runs
    predictAcc_noisy_z = predictAcc_noisy_z/n_runs
    predictAcc_noisy_z_nonoverlapping = predictAcc_noisy_z_nonoverlapping/n_runs

    plt.figure()
    plt.plot(test_setSize, predictAcc[0,:,0]*100,linestyle[0],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_z[0,:,0]*100,linestyle[0],label='LP_Z', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_z_nonoverlapping[0,:,0]*100,linestyle[0],label='LP_NO_Z', color=linecolor[2])
    plt.plot(test_setSize, predictAcc[0,:,1]*100,linestyle[1],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_z[0,:,1]*100,linestyle[1],label='LP_Z', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_z_nonoverlapping[0,:,1]*100,linestyle[1],label='LP_NO_Z', color=linecolor[2])


    plt.title(Title_fig)
    plt.xlabel("Number of Tests")
    plt.ylabel("FN Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')
    #plt.savefig(fig_filename+"_FP"+".png", dpi=150)

    plt.figure()
    plt.plot(test_setSize, predictAcc[1,:,0]*100,linestyle[0],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_z[1,:,0]*100,linestyle[0],label='LP_Z', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_z_nonoverlapping[1,:,0]*100,linestyle[0],label='LP_NO_Z', color=linecolor[2])
    plt.plot(test_setSize, predictAcc[1,:,1]*100,linestyle[1],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_z[1,:,1]*100,linestyle[1],label='LP_Z', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_z_nonoverlapping[1,:,1]*100,linestyle[1],label='LP_NO_Z', color=linecolor[2])

    plt.title(Title_fig)
    plt.xlabel("Number of Tests")
    plt.ylabel("FP Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')

    plt.figure()
    plt.plot(test_setSize, predictAcc[2,:,0]*100,linestyle[0],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_z[2,:,0]*100,linestyle[0],label='LP_Z', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_z_nonoverlapping[2,:,0]*100,linestyle[0],label='LP_NO_Z', color=linecolor[2])
    plt.plot(test_setSize, predictAcc[2,:,1]*100,linestyle[1],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_z[2,:,1]*100,linestyle[1],label='LP_Z', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_z_nonoverlapping[2,:,1]*100,linestyle[1],label='LP_NO_Z', color=linecolor[2])

    plt.title(Title_fig)
    plt.xlabel("Number of Tests")
    plt.ylabel("Hamming Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')

def plot_bsc_nonoverlap(N, q, p, m, n_runs, p_noisy, test_setSize):

    Title_fig = f"BSC Noise on Tests q={q} p={p} m={m}"

    linestyle = ['o-', 'o--']
    linecolor = ['tab:blue','tab:orange','tab:green' ]


    n_n = np.shape(p_noisy)[0]
    predictAcc = np.zeros((3,test_setSize.shape[0],n_n))
    predictAcc_noisy_bsc = np.zeros((3,test_setSize.shape[0],n_n))
    predictAcc_noisy_bsc_nonoverlapping = np.zeros((3,test_setSize.shape[0],n_n))

    for i_n, i_noise in enumerate(p_noisy):

        counter = 0

        for i_test in test_setSize:

            for i_run in np.arange(n_runs):
                np.random.seed(i_run)

                [X,ppl, y, A] = generator_nonoverlapping(N, q, p, m, i_test)
                y = add_noise_zchannel(y,i_noise)

                pred_s = lp_nonoverlapping(X,y,A.copy())
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc[0,counter,i_n] = predictAcc[0,counter,i_n]+temp_FN/np.sum(ppl)
                    
                predictAcc[1,counter,i_n] = predictAcc[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc[2,counter,i_n] = predictAcc[2,counter,i_n]+temp_Hamming/N

                pred_s = lp_noisy_bsc(X,y)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc_noisy_bsc[0,counter,i_n] = predictAcc_noisy_bsc[0,counter,i_n]+temp_FN/np.sum(ppl)
                    
                predictAcc_noisy_bsc[1,counter,i_n] = predictAcc_noisy_bsc[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_noisy_bsc[2,counter,i_n] = predictAcc_noisy_bsc[2,counter,i_n]+temp_Hamming/N

                pred_s = lp_noisy_bsc_nonoverlapping(X,y,A.copy())
                temp_FP,temp_FN,temp_Hamming =  get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc_noisy_bsc_nonoverlapping[0,counter,i_n] = predictAcc_noisy_bsc_nonoverlapping[0,counter,i_n]+temp_FN/np.sum(ppl)
                
                predictAcc_noisy_bsc_nonoverlapping[1,counter,i_n] = predictAcc_noisy_bsc_nonoverlapping[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_noisy_bsc_nonoverlapping[2,counter,i_n] = predictAcc_noisy_bsc_nonoverlapping[2,counter,i_n]+temp_Hamming/N

            counter += 1
            
            
            
    predictAcc = predictAcc/n_runs
    predictAcc_noisy_bsc = predictAcc_noisy_bsc/n_runs
    predictAcc_noisy_bsc_nonoverlapping = predictAcc_noisy_bsc_nonoverlapping/n_runs


    plt.figure()
    plt.plot(test_setSize, predictAcc[0,:,0]*100,linestyle[0],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_bsc[0,:,0]*100,linestyle[0],label='LP_BSC', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_bsc_nonoverlapping[0,:,0]*100,linestyle[0],label='LP_NO_BSC', color=linecolor[2])
    plt.plot(test_setSize, predictAcc[0,:,1]*100,linestyle[1],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_bsc[0,:,1]*100,linestyle[1],label='LP_BSC', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_bsc_nonoverlapping[0,:,1]*100,linestyle[1],label='LP_NO_BSC', color=linecolor[2])


    plt.title(Title_fig)
    plt.xlabel("Number of Tests")
    plt.ylabel("FN Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')

    plt.figure()
    plt.plot(test_setSize, predictAcc[1,:,0]*100,linestyle[0],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_bsc[1,:,0]*100,linestyle[0],label='LP_BSC', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_bsc_nonoverlapping[1,:,0]*100,linestyle[0],label='LP_NO_BSC', color=linecolor[2])
    plt.plot(test_setSize, predictAcc[1,:,1]*100,linestyle[1],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_bsc[1,:,1]*100,linestyle[1],label='LP_BSC', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_bsc_nonoverlapping[1,:,1]*100,linestyle[1],label='LP_NO_BSC', color=linecolor[2])

    plt.title(Title_fig)
    plt.xlabel("Number of Tests")
    plt.ylabel("FP Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')

    plt.figure()
    plt.plot(test_setSize, predictAcc[2,:,0]*100,linestyle[0],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_bsc[2,:,0]*100,linestyle[0],label='LP_BSC', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_bsc_nonoverlapping[2,:,0]*100,linestyle[0],label='LP_NO_BSC', color=linecolor[2])
    plt.plot(test_setSize, predictAcc[2,:,1]*100,linestyle[1],label='LP_NO', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_noisy_bsc[2,:,1]*100,linestyle[1],label='LP_BSC', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_noisy_bsc_nonoverlapping[2,:,1]*100,linestyle[1],label='LP_NO_BSC', color=linecolor[2])

    plt.title(Title_fig)
    plt.xlabel("Number of Tests")
    plt.ylabel("Hamming Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')


def plot_lp(N, q, n_runs, test_setSize):
    counter = 0


    predictAcc_1 = np.zeros((3,test_setSize.shape[0]))
    predictAcc_2 = np.zeros((3,test_setSize.shape[0]))

    for i_test in test_setSize:

        
        for i_run in np.arange(n_runs):
            np.random.seed(i_run)
            
            [X,ppl, y] = generator(N, 0.1, i_test)
            
            pred_s = lp(X,y)
            temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
            if (np.sum(ppl)>0):
                predictAcc_1[0,counter] = predictAcc_1[0,counter]+temp_FN/np.sum(ppl)
            predictAcc_1[1,counter] = predictAcc_1[1,counter]+temp_FP/(N-np.sum(ppl))
            predictAcc_1[2,counter] = predictAcc_1[2,counter]+temp_Hamming/N
            
            
            [X,ppl, y] = generator(N, 0.2, i_test)
            
            pred_s = lp(X,y)
            temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
            if (np.sum(ppl)>0):
                predictAcc_2[0,counter] = predictAcc_2[0,counter]+temp_FN/np.sum(ppl)
            predictAcc_2[1,counter] = predictAcc_2[1,counter]+temp_FP/(N-np.sum(ppl))
            predictAcc_2[2,counter] = predictAcc_2[2,counter]+temp_Hamming/N
        
        
        counter += 1
            
    predictAcc_1 = predictAcc_1/n_runs
    predictAcc_2 = predictAcc_2/n_runs


    plt.figure()
    plt.plot(test_setSize, predictAcc_1[0,:]*100,'o-',label='q=0.1')
    plt.plot(test_setSize, predictAcc_2[0,:]*100,'o-',label='q=0.2')
    plt.xlabel("Number of Tests")
    plt.ylabel("FN Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')


    plt.figure()
    plt.plot(test_setSize, predictAcc_1[1,:]*100,'o-',label='q=0.1')
    plt.plot(test_setSize, predictAcc_2[1,:]*100,'o-',label='q=0.2')
    plt.xlabel("Number of Tests")
    plt.ylabel("FP Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')


    plt.figure()
    plt.plot(test_setSize, predictAcc_1[2,:]*100,'o-',label='q=0.1')
    plt.plot(test_setSize, predictAcc_2[2,:]*100,'o-',label='q=0.2')
    plt.xlabel("Number of Tests")
    plt.ylabel("Hamming Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')


def plot_z_bsc(N, q, p_noisy, n_runs, test_setSize):

    counter = 0
    n_n = np.shape(p_noisy)[0]

    predictAcc_1 = np.zeros((3,test_setSize.shape[0],n_n))
    predictAcc_2 = np.zeros((3,test_setSize.shape[0],n_n))
    predictAcc_3 = np.zeros((3,test_setSize.shape[0],n_n))
    predictAcc_4 = np.zeros((3,test_setSize.shape[0],n_n))

    for i_n, i_noise in enumerate(p_noisy):
        counter = 0
        for i_test in test_setSize:


            for i_run in np.arange(n_runs):
                np.random.seed(i_run)

                [X, ppl, y] = generator(N, q, i_test)
                
                y_z = add_noise_zchannel(y,i_noise)
                
                pred_s = lp(X,y_z)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                if (np.sum(ppl)>0):
                    predictAcc_1[0,counter,i_n] = predictAcc_1[0,counter,i_n]+temp_FN/np.sum(ppl)
                predictAcc_1[1,counter,i_n] = predictAcc_1[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_1[2,counter,i_n] = predictAcc_1[2,counter,i_n]+temp_Hamming/N

                pred_s = lp_noisy_z(X,y_z)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                if (np.sum(ppl)>0):
                    predictAcc_2[0,counter,i_n] = predictAcc_2[0,counter,i_n]+temp_FN/np.sum(ppl)
                predictAcc_2[1,counter,i_n] = predictAcc_2[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_2[2,counter,i_n] = predictAcc_2[2,counter,i_n]+temp_Hamming/N

                y_bsc = add_noise_bsc(y,i_noise)
                
                pred_s = lp(X,y_bsc)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                if (np.sum(ppl)>0):
                    predictAcc_4[0,counter,i_n] = predictAcc_4[0,counter,i_n]+temp_FN/np.sum(ppl)
                predictAcc_4[1,counter,i_n] = predictAcc_4[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_4[2,counter,i_n] = predictAcc_4[2,counter,i_n]+temp_Hamming/N
                
                pred_s = lp_noisy_bsc(X,y_bsc)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                if (np.sum(ppl)>0):
                    predictAcc_3[0,counter,i_n] = predictAcc_3[0,counter,i_n]+temp_FN/np.sum(ppl)
                predictAcc_3[1,counter,i_n] = predictAcc_3[1,counter,i_n]+temp_FP/(N-np.sum(ppl))
                predictAcc_3[2,counter,i_n] = predictAcc_3[2,counter,i_n]+temp_Hamming/N


            counter += 1
            
            
    predictAcc_1 = predictAcc_1/n_runs
    predictAcc_2 = predictAcc_2/n_runs
    predictAcc_3 = predictAcc_3/n_runs
    predictAcc_4 = predictAcc_4/n_runs


    linecolor = ['tab:blue','tab:orange']

    Title_fig_1 = f' q = {q} p_noise = 0.1'
    Title_fig_2 = f' q = {q} p_noise = 0.2'

    plt.figure()
    plt.plot(test_setSize, predictAcc_1[0,:,0]*100,'o-',label='LP with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_2[0,:,0]*100,'o--',label='LP_Z with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_4[0,:,0]*100,'o-',label='LP with BSC Noise', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_3[0,:,0]*100,'o--',label='LP_BSC with BSC Noise', color=linecolor[1])
    plt.xlabel("Number of Tests")
    plt.ylabel("FN Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')
    plt.title(Title_fig_1)


    plt.figure()
    plt.plot(test_setSize, predictAcc_1[0,:,1]*100,'o-',label='LP with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_2[0,:,1]*100,'o--',label='LP_Z with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_4[0,:,1]*100,'o-',label='LP with BSC Noise', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_3[0,:,1]*100,'o--',label='LP_BSC with BSC Noise', color=linecolor[1])
    plt.xlabel("Number of Tests")
    plt.ylabel("FN Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')
    plt.title(Title_fig_2)


    plt.figure()
    plt.plot(test_setSize, predictAcc_1[1,:,0]*100,'o-',label='LP with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_2[1,:,0]*100,'o--',label='LP_Z with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_4[1,:,0]*100,'o-',label='LP with BSC Noise', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_3[1,:,0]*100,'o--',label='LP_BSC with BSC Noise', color=linecolor[1])
    plt.xlabel("Number of Tests")
    plt.ylabel("FP Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')
    plt.title(Title_fig_1)

    plt.figure()
    plt.plot(test_setSize, predictAcc_1[1,:,1]*100,'o-',label='LP with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_2[1,:,1]*100,'o--',label='LP_Z with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_4[1,:,1]*100,'o-',label='LP with BSC Noise', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_3[1,:,1]*100,'o--',label='LP_BSC with BSC Noise', color=linecolor[1])
    plt.xlabel("Number of Tests")
    plt.ylabel("FP Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')
    plt.title(Title_fig_2)


    plt.figure()
    plt.plot(test_setSize, predictAcc_1[2,:,0]*100,'o-',label='LP with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_2[2,:,0]*100,'o--',label='LP_Z with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_4[2,:,0]*100,'o-',label='LP with BSC Noise', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_3[2,:,0]*100,'o--',label='LP_BSC with BSC Noise', color=linecolor[1])
    plt.xlabel("Number of Tests")
    plt.ylabel("Hamming Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')
    plt.title(Title_fig_1)

    plt.figure()
    plt.plot(test_setSize, predictAcc_1[2,:,1]*100,'o-',label='LP with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_2[2,:,1]*100,'o--',label='LP_Z with Z Noise', color=linecolor[0])
    plt.plot(test_setSize, predictAcc_4[2,:,1]*100,'o-',label='LP with BSC Noise', color=linecolor[1])
    plt.plot(test_setSize, predictAcc_3[2,:,1]*100,'o--',label='LP_BSC with BSC Noise', color=linecolor[1])
    plt.xlabel("Number of Tests")
    plt.ylabel("Hamming Error Rate (%)")
    plt.grid(True)
    plt.legend( loc='lower left')
    plt.title(Title_fig_2)

def plot_lp_nonoverlap(N, q, comb, n_runs, test_setSize):

    n_comb = comb.shape[0]

    predictAcc_1 = np.zeros((3,test_setSize.shape[0],n_comb))
    predictAcc_2 = np.zeros((3,test_setSize.shape[0],n_comb))


    for i_comb in np.arange(n_comb):
        counter = 0
        for i_test in test_setSize:


            for i_run in np.arange(n_runs):
                np.random.seed(i_run)

                q = comb[i_comb,1]
                p = comb[i_comb,2]
                m = np.int(comb[i_comb,0])
                
                [X,ppl, y, A] = generator_nonoverlapping(N, q, p, m, i_test)
                

                pred_s = lp(X,y)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc_1[0,counter,i_comb] = predictAcc_1[0,counter,i_comb]+temp_FN/np.sum(ppl)
                    
                predictAcc_1[1,counter,i_comb] = predictAcc_1[1,counter,i_comb]+temp_FP/(N-np.sum(ppl))
                predictAcc_1[2,counter,i_comb] = predictAcc_1[2,counter,i_comb]+temp_Hamming/N


                pred_s = lp_nonoverlapping(X,y,A)
                temp_FP,temp_FN,temp_Hamming = get_stats(ppl, pred_s)
                
                if (np.sum(ppl)>0):
                    predictAcc_2[0,counter,i_comb] = predictAcc_2[0,counter,i_comb]+temp_FN/np.sum(ppl)
                
                predictAcc_2[1,counter,i_comb] = predictAcc_2[1,counter,i_comb]+temp_FP/(N-np.sum(ppl))
                predictAcc_2[2,counter,i_comb] = predictAcc_2[2,counter,i_comb]+temp_Hamming/N


            counter += 1
            
    predictAcc_1 = predictAcc_1/n_runs
    predictAcc_2 = predictAcc_2/n_runs

    linecolor = ['tab:blue','tab:orange','tab:green', 'k' ]

    plt.figure()
    for i_comb in np.arange(n_comb):

        q = comb[i_comb,1]
        p = comb[i_comb,2]
        m = np.int(comb[i_comb,0])
        label1 = f"LP    q={q} p={p} m={m}"
        label2 = f"LP_NO q={q} p={p} m={m}"
        
        plt.plot(test_setSize, predictAcc_1[0,:,i_comb]*100,'o-',label=label1, color=linecolor[i_comb])
        plt.plot(test_setSize, predictAcc_2[0,:,i_comb]*100,'o--',label=label2, color=linecolor[i_comb])
        plt.xlabel("Number of Tests")
        plt.ylabel("FN Error Rate (%)")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.55, 1),loc='upper right')

    plt.figure()
    for i_comb in np.arange(n_comb):
        
        q = comb[i_comb,1]
        p = comb[i_comb,2]
        m = np.int(comb[i_comb,0])
        label1 = f"LP    q={q} p={p} m={m}"
        label2 = f"LP_NO q={q} p={p} m={m}"

        plt.plot(test_setSize, predictAcc_1[1,:,i_comb]*100,'o-',label=label1, color=linecolor[i_comb])
        plt.plot(test_setSize, predictAcc_2[1,:,i_comb]*100,'o--',label=label2, color=linecolor[i_comb])
        plt.xlabel("Number of Tests")
        plt.ylabel("FP Error Rate (%)")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.55, 1),loc='upper right')

    plt.figure()
    for i_comb in np.arange(n_comb):

        q = comb[i_comb,1]
        p = comb[i_comb,2]
        m = np.int(comb[i_comb,0])
        label1 = f"LP    q={q} p={p} m={m}"
        label2 = f"LP_NO q={q} p={p} m={m}"

        plt.plot(test_setSize, predictAcc_1[2,:,i_comb]*100,'o-',label=label1, color=linecolor[i_comb])
        plt.plot(test_setSize, predictAcc_2[2,:,i_comb]*100,'o--',label=label2, color=linecolor[i_comb])
        plt.xlabel("Number of Tests")
        plt.ylabel("Hamming Error Rate (%)")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.55, 1),loc='upper right')
    


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    #Change these values according to your needs, you can also define new variables.
    #n = 1000                        # size of population
    #m = 50                          #number of families
    #p = 0.2                         #probability of infection
    #q = 0.2                         #probability of a family to be chosen as infected
    #T = 200                         #number of tests 

    n = 2
    q = 0.1
    n_runs = 1  # Runs 

    T = np.arange(100,800,100)
    plot_lp(n, 0.1, n_runs, T)

    T = np.arange(100,800,100)
    plot_lp(n, 0.2, n_runs, T)


    T = np.arange(100,800,100)  
    p_noisy = [0.1,0.2] #test noise
    plot_z_bsc(n, 0.1, p_noisy, n_runs, T)

    T = np.array([100,300,500,1000])
    comb = np.array([[20,0.1,0.8],[20,0.2,0.6],[200,0.1,0.8],[200,0.2,0.6]])
    plot_lp_nonoverlap(n, q, comb, n_runs, T)

    T = np.array([100,300,500,1000])
    p_noisy = [0.1,0.2] #test noise


    for i_comb in np.arange(comb.shape[0]):

        q = comb[i_comb,1]
        p = comb[i_comb,2]
        m = np.int(comb[i_comb,0])

        plot_z_nonoverlap(n, q, p, m, n_runs, p_noisy, T)
        plot_bsc_nonoverlap(n, q, p, m, n_runs, p_noisy, T)

