#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 22:07:39 2025

@author: yunkaiwang
"""

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

    I=I#/I.sum(axis=1,keepdims=True)
            
    d=I.sum(axis=0)/N
    #g=np.zeros([M,M])
    #for i in range(M):
    #    for j in range(M):
    #        g[i,j]=np.sum(I[:,i]*I[:,j])/N
    g = np.dot(I.T, I) / N        
    #print (g/g2)
    return d,g,I

    
    








#prior distribution plot
N=10000
u=np.linspace(-1/2,1/2,N)
P=np.zeros(N)
for i in range(N):
    P[i]=classify_P4(u[i])


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
#plt.ylim([1e0,1e15])
plt.plot(u,P,linewidth=3)


foo_fig = plt.gcf() # 'get current figure'
#foo_fig.savefig('prior_P4.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')

plt.show()


M=50
N=5
d, g, I =  classify_gd_generation(N,M)



fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I[i].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 51, 10))
        axes[i].set_xticklabels(np.arange(0, 51, 10), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
fig.savefig('intensity_samples_case1.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I[-i-1].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 51, 10))
        axes[i].set_xticklabels(np.arange(0, 51, 10), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
#fig.savefig('intensity_samples_case2.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()


