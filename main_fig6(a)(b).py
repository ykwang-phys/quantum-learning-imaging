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
import matplotlib.ticker as ticker
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

    
#N number of round to generate the g,d
#M size of the discretization of u
def classify_gd_generation1(N,M):

    y = np.linspace(-1/2, -1/4, M)
    

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
    

#N number of round to generate the g,d
#M size of the discretization of u
def classify_gd_generation2(N,M):

    y = np.linspace(-1/4, 0, M)
    

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
    
#N number of round to generate the g,d
#M size of the discretization of u
def classify_gd_generation3(N,M):

    y = np.linspace(0, 1/4, M)
    

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
    
#N number of round to generate the g,d
#M size of the discretization of u
def classify_gd_generation4(N,M):

    y = np.linspace(1/4, 1/2, M)
    

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

L=10
sigma=1
alpha=0.1
source_size=alpha*sigma
M=4
K=1000
y0=np.array([-0.4, 0.1, 0.2,0.35])*L
    
    
source_position=np.zeros(M*K)

y = np.linspace(-0.5, 0.5, M*K )*L
for j  in range(M*K):
    for q in range(M):
        if abs(y[j] - y0[q]) < source_size/2:
            source_position[j]=1
            
#plt.figure(figsize=(24,4))
plt.figure()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.plot(y/L,source_position, linewidth=3)
#plt.title('source position', fontsize=25)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.5))
# make axis lines thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
#plt.tight_layout()
plt.ylim(-0.1,1.19)
plt.savefig('source_position.png', dpi=300, bbox_inches='tight')
plt.show()
    






#prior distribution plot
N=10000
u1=np.linspace(-1/2,-1/4,N)
P1=np.zeros(N)
for i in range(N):
    P1[i]=classify_P4(u1[i])
    
u2=np.linspace(-1/4,0,N)
P2=np.zeros(N)
for i in range(N):
    P2[i]=classify_P4(u2[i])   

u3=np.linspace(0,1/4,N)
P3=np.zeros(N)
for i in range(N):
    P3[i]=classify_P4(u3[i])
    
u4=np.linspace(1/4,1/2,N)
P4=np.zeros(N)
for i in range(N):
    P4[i]=classify_P4(u4[i])   

'''
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
foo_fig.savefig('prior_P4.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')

plt.show()
'''

plt.figure()
ax = plt.subplot(1,1,1)

# set up ticks (optional, but they'll be hidden anyway)
xmajorLocator = MultipleLocator(0.5)
ax.xaxis.set_major_locator(xmajorLocator)

# remove the axis lines, ticks, labels
ax.axis('off')

# your plot
plt.plot(u1, P1, linewidth=3)

# save without any axes
foo_fig = plt.gcf()
foo_fig.savefig('prior_P1.jpg', format='jpg', dpi=1000, bbox_inches='tight')

plt.show()

plt.figure()
ax = plt.subplot(1,1,1)

# set up ticks (optional, but they'll be hidden anyway)
xmajorLocator = MultipleLocator(0.5)
ax.xaxis.set_major_locator(xmajorLocator)

# remove the axis lines, ticks, labels
ax.axis('off')

# your plot
plt.plot(u2, P2, linewidth=3)

# save without any axes
foo_fig = plt.gcf()
foo_fig.savefig('prior_P2.jpg', format='jpg', dpi=1000, bbox_inches='tight')

plt.show()

plt.figure()
ax = plt.subplot(1,1,1)

# set up ticks (optional, but they'll be hidden anyway)
xmajorLocator = MultipleLocator(0.5)
ax.xaxis.set_major_locator(xmajorLocator)

# remove the axis lines, ticks, labels
ax.axis('off')

# your plot
plt.plot(u3, P3, linewidth=3)

# save without any axes
foo_fig = plt.gcf()
foo_fig.savefig('prior_P3.jpg', format='jpg', dpi=1000, bbox_inches='tight')

plt.show()

plt.figure()
ax = plt.subplot(1,1,1)

# set up ticks (optional, but they'll be hidden anyway)
xmajorLocator = MultipleLocator(0.5)
ax.xaxis.set_major_locator(xmajorLocator)

# remove the axis lines, ticks, labels
ax.axis('off')

# your plot
plt.plot(u4, P4, linewidth=3)

# save without any axes
foo_fig = plt.gcf()
foo_fig.savefig('prior_P4.jpg', format='jpg', dpi=1000, bbox_inches='tight')

plt.show()


M=20
N=5
d, g, I1 =  classify_gd_generation1(N,M)
d, g, I2 =  classify_gd_generation2(N,M)
d, g, I3 =  classify_gd_generation3(N,M)
d, g, I4 =  classify_gd_generation4(N,M)


fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I1[i].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
fig.savefig('intensity1_samples_case1.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I1[-i-1].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
fig.savefig('intensity1_samples_case2.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()






fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I2[i].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
#fig.savefig('intensity2_samples_case1.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I2[-i-1].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
#fig.savefig('intensity2_samples_case2.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()






fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I3[i].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
fig.savefig('intensity3_samples_case1.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I3[-i-1].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
#fig.savefig('intensity3_samples_case2.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()






fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I4[i].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
fig.savefig('intensity4_samples_case1.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(6, 1.1), sharex=True)

for i in range(N):
    row = I4[-i-1].reshape(1, M)
    axes[i].imshow(row, cmap='gray_r', aspect='equal', interpolation='none')
    axes[i].set_yticks([])  # No y-axis ticks
    axes[i].set_ylabel(f'$I_{{{i}}}$', fontsize=12, rotation=0, labelpad=20)
    if i < N-1:
        axes[i].set_xticks([])  # Hide x-ticks for top two plots
    else:
        axes[i].set_xticks(np.arange(0, 21, 5))
        axes[i].set_xticklabels(np.arange(0, 21, 5), fontsize=10)
        axes[i].set_xlabel('Index of $u$', fontsize=12)

plt.tight_layout(h_pad=0.3)
#fig.savefig('intensity4_samples_case2.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.show()

