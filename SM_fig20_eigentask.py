#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:46:03 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:24:04 2025

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
from concurrent.futures import ProcessPoolExecutor, as_completed


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


N=200
M_measure=500 #discretization of the measured interval
M=200 #discretization of the intensity of the source
n=100


#S=np.array([1,2,3,4,5,6,7,8,9])
#S=np.append(S,np.logspace(1,10,n))
S=np.logspace(2,10,n)

L=10 #source size
L_measure=15 #measurement size
sigma=1
val_array=np.zeros([M_measure])
vec_array=np.zeros([M_measure,M_measure])
CT_array=np.zeros(n)

t0=time.time()
val_array,vec_array,CT_array,I,G,D=spectrum_array(N,M,L,M_measure, L_measure,sigma,S)
print (time.time()-t0)


y = np.linspace(-1/2, 1/2, M_measure)


plt.figure()
ax=plt.subplot(1,1,1)


xmajorLocator   = MultipleLocator(0.5) #å°†xä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator   = MultipleLocator(1000000) #å°†yä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator = LogLocator(base=10)  # LogLocator for 10^4n
#xmajorFormatter = FormatStrFormatter('%1.1f') #è®¾ç½®xè½´æ ‡ç­¾æ–‡æœ¬çš„æ ¼å¼
#xminorLocator   = MultipleLocator(1) #å°†xè½´æ¬¡åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º5çš„å€æ•°
#ymajorLocator   = MultipleLocator(0.5)
ax.xaxis.set_major_locator(xmajorLocator)
#ax.xaxis.set_major_formatter(xmajorFormatter)
#ax.yaxis.set_major_locator(ymajorLocator)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#plt.xlim([5,D_array[-1]+10])
#plt.ylim([0,0.55])
#plt.xlim([0,D_array[-1]+1])
plt.ylim([-2,2])
plt.plot(y,vec_array[:,-1],linestyle='-',label='r     1',linewidth=2)
plt.plot(y,vec_array[:,-2],linestyle='--',label='r    2',linewidth=2)
plt.plot(y,vec_array[:,-3],linestyle='-.',label='r     3',linewidth=2)
plt.plot(y,vec_array[:,-4],linestyle='dotted',label='r     4',linewidth=2)
plt.legend(fontsize=17, loc='right', bbox_to_anchor=(1.4, 0.5), ncol=1)
foo_fig = plt.gcf() # 'get current figure'
#foo_fig.savefig('prior4_eigentask.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')

plt.show()
