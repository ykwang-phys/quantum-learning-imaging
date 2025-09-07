#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:33:09 2025

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
    eps = 1e-12
    D += eps * np.eye(D.shape[0])
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
S=10000
n=10
Mv = int(1.5 * M)
L1=100

sigma=np.logspace(0,1,n)
val_array=np.zeros([Mv,n])
vec_array=np.zeros([Mv,Mv,n])
CT_array1=np.zeros(n)

t0=time.time()
for i in range(n):
    val_array[:,i],vec_array[:,:,i],CT_array1[i]=spectrum(N,M,L1,sigma[i],S)
    print (i,time.time()-t0)
    
for i in range(n-1):
    print (np.log(CT_array1[i]/CT_array1[i+1])/np.log(sigma[i]/sigma[i+1]))    
    
L2=200
CT_array2=np.zeros(n)

t0=time.time()
for i in range(n):
    val_array[:,i],vec_array[:,:,i],CT_array2[i]=spectrum(N,M,L2,sigma[i],S)
    print (i,time.time()-t0)
    


for i in range(n-1):
    print (np.log(CT_array2[i]/CT_array2[i+1])/np.log(sigma[i]/sigma[i+1]))
    
L3=300
CT_array3=np.zeros(n)

t0=time.time()
for i in range(n):
    val_array[:,i],vec_array[:,:,i],CT_array3[i]=spectrum(N,M,L3,sigma[i],S)
    print (i,time.time()-t0)
    


for i in range(n-1):
    print (np.log(CT_array3[i]/CT_array3[i+1])/np.log(sigma[i]/sigma[i+1]))
    
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
#plt.xlim([0.5,S[-1]])
#plt.ylim([0,0.55])
#plt.xlim([0,D_array[-1]+1])
#plt.ylim([5,100])

plt.loglog(sigma,CT_array1,linestyle='-',linewidth=3,label='L1='+str(L1))
plt.loglog(sigma,CT_array2,linestyle='--',linewidth=3,label='L2='+str(L2)) 
plt.loglog(sigma,CT_array3,linestyle='-.',linewidth=3,label='L3='+str(L3))
plt.legend(fontsize=15)


foo_fig = plt.gcf() # 'get current figure'
#foo_fig.savefig('general_source_sigma.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')
plt.show()