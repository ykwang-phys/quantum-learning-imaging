#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 22:42:23 2025

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

def GD_integral(N,M,L,sigma):
    d,g=gd_generation(N,M)
    Mv=int(1.5*M)
    
    
    u_array=np.linspace(-L/2,L/2,M)
    v_array=np.linspace(-L*0.75,L*0.75,Mv)
    
    dis=L*1.5/Mv
    
    G=np.zeros([Mv,Mv])
    for i in range(Mv):
        for j in range(Mv):
            for k in range(M):
                for l in range(M):
                    temp1,err=integrate.quad(psf_square,u_array[k]-v_array[i]-dis/2,u_array[k]-v_array[i]+dis/2,args=(sigma,))
                    temp2,err=integrate.quad(psf_square,u_array[l]-v_array[j]-dis/2,u_array[l]-v_array[j]+dis/2,args=(sigma,))
                    G[i,j]+=g[k,l]*temp1*temp2
                    #G[i,j]+=g[k,l]*psf(u_array[k]-v_array[i],sigma)**2*psf(u_array[l]-v_array[j],sigma)**2
    D=np.zeros([Mv,Mv])
    for i in range(Mv):
        for k in range(M):
            temp=integrate.quad(psf_square,u_array[k]-v_array[i]-dis/2,u_array[k]-v_array[i]+dis/2,args=(sigma,))
            D[i,i]+=d[k]*temp
        
    return G,D



def GD_integral_fast(N, M, L, sigma):
    d, g = gd_generation(N, M)
    Mv = int(1.5 * M)
    
    u_array = np.linspace(-L / 2, L / 2, M)
    v_array = np.linspace(-L * 0.75, L * 0.75, Mv)
    dis = L * 1.5 / Mv
    
    # Precompute psf_square integrals for all u-v combinations
    psf_integrals = np.zeros((M, Mv))
    for k in range(M):
        for i in range(Mv):
            psf_integrals[k, i], _ = integrate.quad(psf_square, 
                                                   u_array[k] - v_array[i] - dis / 2, 
                                                   u_array[k] - v_array[i] + dis / 2, 
                                                   args=(sigma,))
    
    # Compute G matrix using precomputed psf_integrals
    G = np.zeros((Mv, Mv))
    for i in range(Mv):
        for j in range(Mv):
            G[i, j] = np.sum(g * np.outer(psf_integrals[:, i], psf_integrals[:, j]))
    
    # Compute D matrix using precomputed psf_integrals
    D = np.zeros((Mv, Mv))
    for i in range(Mv):
        D[i, i] = np.sum(d * psf_integrals[:, i])
    
    return G, D


def GD_fast(N, M, L, sigma):
    l=L/M
    d, g = gd_generation(N, M)
    Mv = int(1.5 * M)
    #print (d[0:4],g[0:4,0:4])
    u_array = np.linspace(-L/2, L/2, M)
    v_array = np.linspace(-L*0.75, L*0.75, Mv)
    
    # Create G matrix efficiently
    psf_u = psf(u_array[:, None] - v_array[None, :], sigma)
    psf_squared = psf_u ** 2
    
    G = np.einsum('kl,ki,lj->ij', g, psf_squared, psf_squared)
    
    # Create D matrix efficiently
    D = np.zeros([Mv, Mv])
    for i in range(Mv):
        D[i, i] = np.sum(d * psf_squared[:, i])
    
    return G*l**2, D*l

def spectrum(N,M,L,sigma,S):
    G,D=GD_fast(N, M, L, sigma)
    val,vec=eigh(G,D)
    
    M1=G+(D-G)/S
    M2=inv(M1)@G
    CT=np.matrix.trace(M2)

    return val,vec,CT


def spectrum_array(N,M,L,sigma,S):
    #G,D=GD_fast(N, M, L, sigma)
    G,D=GD_integral_fast(N, M, L, sigma)
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
    return val,vec,CT


N=200
M=200
n=100
Mv = int(1.5 * M)

#S=np.array([1,2,3,4,5,6,7,8,9])
#S=np.append(S,np.logspace(1,10,n))
S=np.logspace(0,14,n)

L=20
sigma=1
val_array1=np.zeros([Mv])
vec_array1=np.zeros([Mv,Mv])
CT_array1=np.zeros(n)
slope1=np.zeros(n-1)

t0=time.time()
val_array1,vec_array1,CT_array1=spectrum_array(N,M,L,sigma,S)
print (time.time()-t0)

for i in range(n-1):
    temp1=CT_array1[i+1]-CT_array1[i]
    temp2=np.log(S[i+1])-np.log(S[i])
    slope1[i]=temp1/temp2
    #print (i,temp1/temp2)

L=40
sigma=1
val_array2=np.zeros([Mv])
vec_array2=np.zeros([Mv,Mv])
CT_array2=np.zeros(n)
slope2=np.zeros(n-1)
t0=time.time()
val_array2,vec_array2,CT_array2=spectrum_array(N,M,L,sigma,S)
print (time.time()-t0)

for i in range(n-1):
    temp1=CT_array2[i+1]-CT_array2[i]
    temp2=np.log(S[i+1])-np.log(S[i])
    slope2[i]=temp1/temp2
    #print (i,temp1/temp2)


L=60
sigma=1
val_array3=np.zeros([Mv])
vec_array3=np.zeros([Mv,Mv])
CT_array3=np.zeros(n)
slope3=np.zeros(n-1)
t0=time.time()
val_array3,vec_array3,CT_array3=spectrum_array(N,M,L,sigma,S)
print (time.time()-t0)

for i in range(n-1):
    temp1=CT_array3[i+1]-CT_array3[i]
    temp2=np.log(S[i+1])-np.log(S[i])
    slope3[i]=temp1/temp2
    #print (i,temp1/temp2)
    
plt.figure()
ax=plt.subplot(1,1,1)


#xmajorLocator   = MultipleLocator(2) #å°†xä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator   = MultipleLocator(1000000) #å°†yä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator = LogLocator(base=10)  # LogLocator for 10^4n
#xmajorFormatter = FormatStrFormatter('%1.1f') #è®¾ç½®xè½´æ ‡ç­¾æ–‡æœ¬çš„æ ¼å¼
#xminorLocator   = MultipleLocator(1) #å°†xè½´æ¬¡åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º5çš„å€æ•°
#ymajorLocator   = MultipleLocator(0.5)
#ax.xaxis.set_major_locator(xmajorLocator)
#ax.xaxis.set_major_formatter(xmajorFormatter)
#ax.yaxis.set_major_locator(ymajorLocator)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim([0.5,S[-1]])
#plt.ylim([0,0.55])
#plt.xlim([0,D_array[-1]+1])
#plt.ylim([0,70])
label='log'
if label=='log':
    plt.semilogx(S,CT_array1,linewidth=2,linestyle='-',label='L=20')
    plt.semilogx(S,CT_array2,linewidth=2,linestyle='dotted',label='L=40')
    plt.semilogx(S,CT_array3,linewidth=2,linestyle='-.',label='L=60')
else:
    plt.plot(S,CT_array1,linewidth=2,linestyle='-',label='L=20')
    plt.plot(S,CT_array2,linewidth=2,linestyle='dotted',label='L=40')
    plt.plot(S,CT_array3,linewidth=2,linestyle='-.',label='L=60')    
plt.legend(fontsize=16,loc='lower right')


foo_fig = plt.gcf() # 'get current figure'
#foo_fig.savefig('general_source_S.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')

plt.show()


    
plt.figure()
ax=plt.subplot(1,1,1)


#xmajorLocator   = MultipleLocator(2) #å°†xä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator   = MultipleLocator(1000000) #å°†yä¸»åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º20çš„å€æ•°
#ymajorLocator = LogLocator(base=10)  # LogLocator for 10^4n
#xmajorFormatter = FormatStrFormatter('%1.1f') #è®¾ç½®xè½´æ ‡ç­¾æ–‡æœ¬çš„æ ¼å¼
#xminorLocator   = MultipleLocator(1) #å°†xè½´æ¬¡åˆ»åº¦æ ‡ç­¾è®¾ç½®ä¸º5çš„å€æ•°
#ymajorLocator   = MultipleLocator(0.5)
#ax.xaxis.set_major_locator(xmajorLocator)
#ax.xaxis.set_major_formatter(xmajorFormatter)
#ax.yaxis.set_major_locator(ymajorLocator)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim([0.5,S[-1]])
#plt.ylim([0,0.55])
#plt.xlim([0,D_array[-1]+1])
#plt.ylim([0,70])

plt.semilogx(S[-(n-1):],slope1,linewidth=2,linestyle='-',label='L=20')
plt.semilogx(S[-(n-1):],slope2,linewidth=2,linestyle='dotted',label='L=40')
plt.semilogx(S[-(n-1):],slope3,linewidth=2,linestyle='-.',label='L=60')    
plt.legend(fontsize=16,loc='upper right')


foo_fig = plt.gcf() # 'get current figure'
#foo_fig.savefig('slope.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')

plt.show()