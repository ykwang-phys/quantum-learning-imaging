#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 21:47:44 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:01:13 2025

@author: yunkaiwang
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig,eigh,inv
import matplotlib.pyplot as plt
import random
import time
from scipy import integrate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import MultipleLocator, LogLocator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing



#N number of round to generate the g,d
#M size of the discretization of u
def gd_generation(N,M):
    I = np.random.rand(N,M)
    I=I/I.sum(axis=1,keepdims=True)
    d=I.sum(axis=0)/N
    #g=np.zeros([M,M])
    #for i in range(M):
    #    for j in range(M):
    #        g[i,j]=np.sum(I[:,i]*I[:,j])/N
    g = np.dot(I.T, I) / N        
    #print (g/g2)
    return d,g

def classify_P1(u):
    
    if -1/2<=u and u<=-1/3:
        return 1
    elif u>=-1/8 and u<=0:
        return 1
    elif u>=1/12 and u<=1/12+1/8:
        return 1
    elif u>=1/3 and u<=5/12:
        return 1
    else:
        return 0

def classify_P2(u):
    
    if -1/2<=u and u<=-1/4:
        return 1
    elif u>=-1/8 and u<=0:
        return 1

    elif u>=1/3 and u<=1/3+1/8:
        return 1
    else:
        return 0
 
def classify_P3(u):
    
    if -1/2<=u and u<=-3/8:
        return 1
    elif -1/8<=u and u<=0:
        return 1
    elif u>=1/4 and u<=1/2:
        return 1
    else:
        return 0
    
def classify_P4(u):
    """
    Classification function over u in [-1/2, 1/2] using at least 10
    non-contiguous intervals of varying lengths. The total length of
    intervals that return 1 is exactly 0.5.
    """
    if -0.50 <= u <= -0.48:
        return 1  # length 0.02
    elif -0.46 <= u <= -0.43:
        return 1  # length 0.03
    elif -0.40 <= u <= -0.37:
        return 1  # length 0.03
    elif -0.33 <= u <= -0.30:
        return 1  # length 0.03
    elif -0.25 <= u <= -0.21:
        return 1  # length 0.04
    elif -0.15 <= u <= -0.12:
        return 1  # length 0.03
    elif -0.05 <= u <= -0.01:
        return 1  # length 0.04
    elif  0.02 <= u <=  0.06:
        return 1  # length 0.04
    elif  0.10 <= u <=  0.15:
        return 1  # length 0.05
    elif  0.19 <= u <=  0.26:
        return 1  # length 0.07
    elif  0.30 <= u <=  0.36:
        return 1  # length 0.06
    elif  0.39 <= u <=  0.43:
        return 1  # length 0.04
    elif  0.48 <= u <=  0.5:
        return 1  # length 0.02
    else:
        return 0

#N number of round to generate the g,d
#M size of the discretization of u
def classify_gd_generation(N,M):

    y = np.linspace(-1/2, 1/2, M)
    

    I=np.zeros([2*N,M])

    for i in range(N):
        for j in range(M):
            if classify_P4(y[j])==1 and np.random.random()<0.2:
                I[i,j]=1
            if classify_P4(y[j])==0 and np.random.random()<0.2:
                I[i+N,j]=1

    I=I/I.sum(axis=1,keepdims=True)
            
    d=I.sum(axis=0)/N
    #g=np.zeros([M,M])
    #for i in range(M):
    #    for j in range(M):
    #        g[i,j]=np.sum(I[:,i]*I[:,j])/N
    g = np.dot(I.T, I) / N        
    #print (g/g2)
    return d,g,I

    
    

def psf(u,sigma):
    result=(2*np.pi*sigma**2)**(-1/4)*np.exp(-u*u/4/sigma/sigma)
    return result
    
def GD(N,M,L,sigma):
    d,g=gd_generation(N,M)
    Mv=int(1.5*M)
    
    
    u_array=np.linspace(-L/2,L/2,M)
    v_array=np.linspace(-L*0.75,L*0.75,Mv)
    
    
    G=np.zeros([Mv,Mv])
    for i in range(Mv):
        for j in range(Mv):
            for k in range(M):
                for l in range(M):
                    G[i,j]+=g[k,l]*psf(u_array[k]-v_array[i],sigma)**2*psf(u_array[l]-v_array[j],sigma)**2
    D=np.zeros([Mv,Mv])
    for i in range(Mv):
        for k in range(M):
            D[i,i]+=d[k]*psf(u_array[k]-v_array[i],sigma)**2
        
    return G,D

def psf_square(u,sigma):
    result=(2*np.pi*sigma**2)**(-1/4)*np.exp(-u*u/4/sigma/sigma)
    return result**2





def GD_integral_fast(N, M, M_measure, L_measure, sigma):
    #d, g = gd_generation(N, M)
    d, g, I =  classify_gd_generation(N,M)
    #Mv = int(1.5 * M)
    
    #L=max(l,sigma)
    u_array = np.linspace(-L / 2, L / 2, M)
    v_array = np.linspace(-L_measure/2, L_measure/2, M_measure)
    dis = L_measure / M_measure
    
    # Precompute psf_square integrals for all u-v combinations
    psf_integrals = np.zeros((M, M_measure))
    for k in range(M):
        for i in range(M_measure):
            psf_integrals[k, i], _ = integrate.quad(psf_square, 
                                                   u_array[k] - v_array[i] - dis / 2, 
                                                   u_array[k] - v_array[i] + dis / 2, 
                                                   args=(sigma,))
    
    # Compute G matrix using precomputed psf_integrals
    G = np.zeros((M_measure, M_measure))
    for i in range(M_measure):
        for j in range(M_measure):
            G[i, j] = np.sum(g * np.outer(psf_integrals[:, i], psf_integrals[:, j]))
    
    # Compute D matrix using precomputed psf_integrals
    D = np.zeros((M_measure, M_measure))
    for i in range(M_measure):
        D[i, i] = np.sum(d * psf_integrals[:, i])
    
    return G, D, I




def spectrum_array(N,M,L,M_measure, L_measure,sigma,S):
    #G,D=GD_fast(N, M, L, sigma)
    #G,D,I=GD_integral_fast(N, M, L, sigma)
    G,D,I=GD_integral_fast(N, M, M_measure, L_measure, sigma)
    val,vec=eigh(G,D)
    #print (D)
    (c,c)=np.shape(G)
    (n,)=np.shape(S)
    CT=np.zeros(n)

    for i in range(n):
        M1=G+(D-G)/S[i]
        M2=inv(M1)@G
        CT[i]=np.matrix.trace(M2)
    '''   
    beta=1/val-1
    for i in range(n):
        for j in range(c):
            CT[i]+=1/(1+beta[j]/S[i])
    '''
    return val,vec,CT,I,G,D

def p(u,L,sigma,I):
    (N,)=np.shape(I)
    y = np.linspace(-L/2, L/2, N)
    result=0
    for i in range(N):
        result+=I[i]*np.exp(-(u-y[i])**2/2/sigma**2)/np.sqrt(2*np.pi*sigma**2)
    return result
    
    
def Prob_array_discrete(I,M_measure,L_measure,L,sigma):
    #print (I)
    #(m,)=np.shape(I)
    x_array = np.linspace(-L_measure/2, L_measure/2, M_measure)
    l=(np.max(x_array)-np.min(x_array))/(M_measure-1)

    
    result=np.zeros(M_measure)
    for i in range(M_measure):
        y=x_array[i]
        result[i]=p(y,L,sigma,I)*l
    
    return result#/np.sum(result)        





    

def train_classifier_simple(training_data1_, training_data2_, fitting_order):

    # 1. Select last `fitting_order` rows (features)
    training_data1 = training_data1_[-fitting_order:, :]
    training_data2 = training_data2_[-fitting_order:, :]

    # 2. Stack samples and create labels
    X_raw = np.vstack((training_data1.T, training_data2.T))  # shape (n_samples, n_features)
    y = np.hstack((
        np.zeros(training_data1.shape[1], dtype=int),
        np.ones(training_data2.shape[1], dtype=int)
    ))

    # 3. Compute scaler: mean of absolute values of each feature
    scaler = np.mean(np.abs(X_raw), axis=0)
    scaler[scaler == 0] = 1.0   # avoid division by zero

    # 4. Scale features by dividing (no centering)
    X = X_raw / scaler

    # 5. Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    return clf, scaler


def classify_new_data_simple(clf, scaler, new_data, r, fitting_order):

    # 1. Project raw data
    proj = new_data @ r   # 1D array of length = total features
    # 2. Select last fitting_order features
    features = proj[-fitting_order:]
    # 3. Scale by precomputed scaler (same indices)
    #print (features,'before scaler')
    X_new = features / scaler[-fitting_order:]
    #print (X_new,'after scaler')
    # 4. Predict class (0 or 1)
    return clf.predict(X_new.reshape(1, -1))[0]



def sample_and_estimate_distribution_fast(P_test, sam_num):

    # Normalize the input distribution
    P = P_test / np.sum(P_test)
    sam_num_=int(sam_num)
    # Draw counts via a single multinomial sample
    #print (sam_num_)
    counts = np.random.multinomial(sam_num_, P)
    #print (counts)
    # Convert counts to empirical probabilities
    return counts / sam_num


def test_error(training_data1,training_data2,fitting_order,I, sam_num,N,M_measure,L_measure):
    clf, scaler = train_classifier_simple(training_data1, training_data2,fitting_order)
    count0=0
    count1=0
    for i in range(N):
        
        P_test=Prob_array_discrete(I[i, :],M_measure,L_measure,L,sigma)
        P_est = sample_and_estimate_distribution_fast(P_test, sam_num)
        predicted_class = classify_new_data_simple(clf, scaler, P_est,vec_array,fitting_order)
        #print("Predicted class 0 (sampled):", predicted_class)
        if predicted_class==0:
            count0+=1
        
        #print (((P_est@r)/(P_test@r))[-fitting_order:])       
        
        P_test=Prob_array_discrete(I[N+i, :],M_measure,L_measure,L,sigma)
        P_est = sample_and_estimate_distribution_fast(P_test, sam_num)
        predicted_class = classify_new_data_simple(clf, scaler, P_est,vec_array,fitting_order)
        #print("Predicted class 1 (sampled):", predicted_class)
        if predicted_class==1:
            count1+=1
    
        #print (((P_est@r)/(P_test@r))[-fitting_order:])       
    return count0, count1

def eigentask_precision(vec_array,order,M,L,sam_num,repeat,M_measure, L_measure,sigma):
    N=int(repeat/2)
    n=10
    d,g,I=classify_gd_generation(N,M)
    features_test=np.zeros([order,repeat])
    features_est=np.zeros([order,repeat])
    diff=np.zeros([order,repeat])
    for i in range(repeat):
        #if i%10==0:
            #print (i)
        P_test =Prob_array_discrete(I[i, :],M_measure,L_measure,L,sigma)
        #print (np.shape(P_test),np.shape(r))
        P_est = sample_and_estimate_distribution_fast(P_test, sam_num)
        features_test[:,i] = (P_test @ vec_array)[-order:]   
        features_est[:,i] = (P_est @ vec_array)[-order:]   
        
        diff[:,i]=(features_test[:,i]-features_est[:,i])/features_test[:,i]
        #print (features_test[:,i],features_est[:,i],diff[:,i])
    precision=np.zeros(order)
    for i in range(order):
        precision[i]=np.average(diff[i,:]**2)
    return precision

def run_single_simulation(i, vec_array, order, M, L, sam_num, repeat,M_measure, L_measure,sigma):
    #print(f"Running simulation {i}")
    precision_array = np.zeros((len(sam_num), order))
    for j, s in enumerate(sam_num):
        s_int = int(s)
        precision_array[j, :] = eigentask_precision(vec_array, order, M, L, s_int, repeat,M_measure, L_measure,sigma)
    return precision_array

def run_parallel_simulations(M_run, vec_array, order, M, L, sam_num, repeat,M_measure, L_measure,sigma):
    num_cpus = min(M_run, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [
            executor.submit(run_single_simulation, i, vec_array, order, M, L, sam_num, repeat,M_measure, L_measure,sigma)
            for i in range(M_run)
        ]
        results = [f.result() for f in futures]

    # Stack into array of shape [M_run, len(sam_num), order]
    return np.stack(results)

def plot_precision_summary(precision_all, sam_num, eigentask_indices=None, colors=None):
    print ('plot fun')
    """
    Plot solid mean lines with shaded min-max regions for selected eigentasks.

    Parameters:
        precision_all: ndarray of shape (M_run, K, order)
        sam_num: array of sample sizes of shape (K,)
        eigentask_indices: list of int indices of eigentasks to plot
        colors: optional list of colors
    """
    M_run, K, order = precision_all.shape
    if eigentask_indices is None:
        eigentask_indices = list(range(order))
    if colors is None:
        colors = [None] * len(eigentask_indices)  # matplotlib default cycle

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, i in enumerate(eigentask_indices):
        data = precision_all[:, :, i]  # shape (M_run, K)
        mean = np.mean(data, axis=0)
        min_ = np.min(data, axis=0)
        max_ = np.max(data, axis=0)

        ax.plot(sam_num, mean, label=f"r{i}", color=colors[idx], linewidth=2)
        ax.fill_between(sam_num, min_, max_, alpha=0.2, color=colors[idx])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Sample size", fontsize=14)
    ax.set_ylabel("Relative squared error", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precision_summary2(precision_all, sam_num, eigentask_indices=None, colors=None):
    print('plot fun')
    """
    Plot solid mean lines with shaded ±std regions for selected eigentasks.

    Parameters:
        precision_all: ndarray of shape (M_run, K, order)
        sam_num: array of sample sizes of shape (K,)
        eigentask_indices: list of int indices of eigentasks to plot
        colors: optional list of colors
    """
    M_run, K, order = precision_all.shape
    if eigentask_indices is None:
        eigentask_indices = list(range(order))
    if colors is None:
        colors = [None] * len(eigentask_indices)  # Use matplotlib default colors

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, i in enumerate(eigentask_indices):
        data = precision_all[:, :, i]  # shape (M_run, K)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        print (std)
        ax.plot(sam_num, mean, label=f"r{i}", color=colors[idx], linewidth=2)
        ax.fill_between(sam_num, mean - std, mean + std, alpha=0.2, color=colors[idx])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Sample size", fontsize=14)
    ax.set_ylabel("Relative squared error", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

N=400 #number of samples used in the generation of prior
M_measure=500 #discretization of the measured interval
M=200 #discretization of the intensity of the source
n=20


#S=np.array([1,2,3,4,5,6,7,8,9])
#S=np.append(S,np.logspace(1,10,n))
S=np.logspace(2,8,n)

L=10 #source size
L_measure=15 #measurement size
sigma=1
val_array=np.zeros([M_measure])
vec_array=np.zeros([M_measure,M_measure])
CT_array=np.zeros(n)

t0=time.time()
val_array,vec_array,CT_array,I,G,D=spectrum_array(N,M,L,M_measure, L_measure,sigma,S)
print (time.time()-t0)


fig, ax = plt.subplots()

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim([-1,21])
ax.xaxis.set_major_locator(MultipleLocator(5))

xarray=np.linspace(1,20,20)
ax.semilogy(xarray,(1/np.abs(val_array[-21:])-1)[::-1][-20:],linewidth=2.5,linestyle='-',label='bet2')

#ax.semilogy(range(21),val_array[-21:][::-1],linewidth=3)
#plt.savefig('eigenvalue_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots()

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#plt.xlim([-1,21])
#ax.xaxis.set_major_locator(MultipleLocator(5))

xarray=np.linspace(1,20,20)
ax.semilogx(S,CT_array,linewidth=2.5,linestyle='-',label='bet2')

#ax.semilogy(range(21),val_array[-21:][::-1],linewidth=3)
#plt.savefig('CT_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()



