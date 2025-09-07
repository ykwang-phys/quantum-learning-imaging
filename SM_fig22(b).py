#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 22:10:53 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 21:33:04 2025

@author: yunkaiwang
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 00:34:00 2025

@author: 27432
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 22:59:55 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 16:10:49 2025

@author: yunkaiwang
"""
import pickle
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
from functools import partial
from scipy.linalg import qr
from scipy.special import hermite
import math
from scipy.linalg import eig,eigh,inv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
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
    
def classify_P_multiple(u,q):
    if q==2:
        return classify_P4(u)
    elif q==1:
        return classify_P4(u)
    elif q==0:
        return classify_P4(u)
    elif q>=3:
        return classify_P4(u)
    
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


#W number of round to generate the g,d
#N size of the discretization of u
#M is the number of pieces we divide the figure into
#n order of derivation
def gd_generation(W,N,M,n,L):
    I = np.random.rand(W,N)
    I_normalized=I/I.sum(axis=1,keepdims=True)
    #print (np.shape(I_normalized))
    l=L/M
    
    y = np.linspace(-L/2, L/2, N)
    y0_temp = np.linspace(-L/2, L/2, 1+M)
    y0=np.zeros(M)
    for i in range(M):
        y0[i]=y0_temp[i]/2+y0_temp[i+1]/2
        
    
    m=N//M
    x=np.zeros([M,n,W])
    for i in range(M):
        #print (i)
        for j in range(n):
            print (j,n)
            for k in range(m):
                for w in range(W):

                        
                    #print (i,j,k,w,i*m+k,N)
                    #print (I_normalized[w,i*m+k])
                    #print (y0[i])
                    #print (y[i*m+k])

                    x[i,j,w]+=I_normalized[w,i*m+k]*((y[i*m+k]-y0[i])/l)**j

    d = np.zeros([n,M])
    for i in range(M):
        for j in range(n):
            for w in range(W):
                d[j,i]+=x[i,j,w]/W
            
    
    g=np.zeros([n*M,n*M])
    for i1 in range(M):
        for i2 in range(M):
            for j1 in range(n):
                for j2 in range(n):
                    for w in range(W):
                        g[j1*M+i1,j2*M+i2]+=x[i1,j1,w]*x[i2,j2,w]/W

    return d,g,x

#W is the number of round of generating images
#N is the discretization of the source
#M is the number of sources
#n is the included order of moments
#L is the size of the source
def gd_generation_fast_classify(W, N, M, n, L,classify_P):
    I = np.random.rand(W, N)
    
    y = np.linspace(-1/2, 1/2, N)
    

    I=np.zeros([2*W,N])

    for i in range(W):
        for j in range(N):
            if classify_P(y[j])==1 and np.random.random()<0.2:
                I[i,j]=1
            if classify_P(y[j])==0 and np.random.random()<0.2:
                I[i+W,j]=1

    
    # —— ensure no zero‐sum rows before normalization ——
    row_sums = I.sum(axis=1, keepdims=True)
    zero_rows = np.where(row_sums.flatten() == 0)[0]
    while zero_rows.size > 0:
        for idx in zero_rows:
            # regenerate that entire row
            new_row = np.zeros(N)
            for j in range(N):
                val = classify_P(y[j])
                # decide which half (first N rows or second N rows)
                if (idx < W and val == 1) or (idx >= W and val == 0):
                    if np.random.random() < 0.2:
                        new_row[j] = 1
            I[idx] = new_row
        # recompute sums and check again
        row_sums = I.sum(axis=1, keepdims=True)
        zero_rows = np.where(row_sums.flatten() == 0)[0]

    # normalize each row
    I_normalized = I / row_sums
    #I_normalized = I / I.sum(axis=1, keepdims=True)
    
    l = L / M
    y = np.linspace(-L/2, L/2, N)
    y0_temp = np.linspace(-L/2, L/2, M + 1)
    y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

    m = N // M
    x = np.zeros((M, n, 2*W))
    
    for i in range(M):
        y_diff = (y[i*m:(i+1)*m] - y0[i]) / l
        y_diff_matrix = np.array([y_diff**j for j in range(n)])
        x[i, :, :] = np.dot(y_diff_matrix, I_normalized[:, i*m:(i+1)*m].T)
    #print (x[:,:,0])    
    # Calculate d using matrix operations
    d = np.mean(x, axis=2).T
    
    # Reshape x and calculate g
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, 2*W)
    g = np.dot(x_reshaped, x_reshaped.T) / W/2

    return d, g,x



#W is the number of round of generating images
#N is the discretization of the source
#M is the number of sources
#n is the included order of moments
#L is the size of the source
def gd_generation_fast(W, N, M, n, L):
    I = np.random.rand(W, N)
    I_normalized = I / I.sum(axis=1, keepdims=True)
    
    l = L / M
    y = np.linspace(-L/2, L/2, N)
    y0_temp = np.linspace(-L/2, L/2, M + 1)
    y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

    m = N // M
    x = np.zeros((M, n, W))
    
    for i in range(M):
        y_diff = (y[i*m:(i+1)*m] - y0[i]) / l
        y_diff_matrix = np.array([y_diff**j for j in range(n)])
        x[i, :, :] = np.dot(y_diff_matrix, I_normalized[:, i*m:(i+1)*m].T)
    #print (x[:,:,0])    
    # Calculate d using matrix operations
    d = np.mean(x, axis=2).T
    
    # Reshape x and calculate g
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, W)
    g = np.dot(x_reshaped, x_reshaped.T) / W

    return d, g,x

def gd_generation_fast_edge(W, N, M, n, L):
    I = np.random.rand(W, N)
    #I_normalized = I / I.sum(axis=1, keepdims=True)
    
    l = L / M
    y = np.linspace(-L/2, L/2, N)
    y0_temp = np.linspace(-L/2, L/2, M + 1)
    y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

    m = N // M
    x = np.zeros((M, n, W))
    
    for i in range(M):
        y_diff = (y[i*m:(i+1)*m] - y0[i]) / l
        y_diff_matrix = np.array([y_diff**j for j in range(n)])
        for j in range(m):
            #print (np.shape(y_diff))
            if np.abs(y_diff[j])>0.2:
                I[:,i*m+j]*=np.exp(10*np.abs(y_diff[j]))
        
    I_normalized = I / I.sum(axis=1, keepdims=True)    
    print (I_normalized)
    for i in range(M):
        x[i, :, :] = np.dot(y_diff_matrix, I_normalized[:, i*m:(i+1)*m].T)
    #print (x[:,:,0])    
    # Calculate d using matrix operations
    d = np.mean(x, axis=2).T
    
    # Reshape x and calculate g
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, W)
    g = np.dot(x_reshaped, x_reshaped.T) / W

    return d, g,x


def gd_generation_linear(W, N, M, n, L):
    W=5
    
    x=np.linspace(0,1,W)
    for i in range(W):
    #I = np.random.rand(W, N)
        
        I1=np.linspace(1,x[i],N//2)
        I2=np.linspace(x[i],1,N//2)
        I_1=np.array([np.append(I1,I2)])
        if i==0:
            I=I_1
        else:
            I=np.append(I,I_1,axis=0)
    #print (I)
    print (np.shape(I))
    I_normalized = I / I.sum(axis=1, keepdims=True)
    
    l = L / M
    y = np.linspace(-L/2, L/2, N)
    y0_temp = np.linspace(-L/2, L/2, M + 1)
    y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

    m = N // M
    x = np.zeros((M, n, W))
    
    for i in range(M):
        y_diff = (y[i*m:(i+1)*m] - y0[i]) / l
        y_diff_matrix = np.array([y_diff**j for j in range(n)])
        x[i, :, :] = np.dot(y_diff_matrix, I_normalized[:, i*m:(i+1)*m].T)
    #print (x[:,:,0])    
    # Calculate d using matrix operations
    d = np.mean(x, axis=2).T
    
    # Reshape x and calculate g
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, W)
    g = np.dot(x_reshaped, x_reshaped.T) / W

    return d, g,x

def log_normal(u,mu,sigma):
    print (u,mu,sigma)
    result = np.zeros_like(u)
   
    # Apply the function only where u >= 0
    positive_mask = u > 0
    result[positive_mask] = np.exp(-(np.log(u[positive_mask]) - mu)**2 / (2 * sigma**2)) / (u[positive_mask] * sigma * np.sqrt(2 * np.pi))
   
    return result
    #return np.exp(-(np.log(u)-mu)**2/2/sigma**2)/u/sigma/np.sqrt(2*np.pi)

def gd_generation_log_normal(W, N, M, n):
    
    y = np.linspace(0, 30, N)
    
    shift=np.linspace(-5,20,W)
    sigma=np.linspace(10,30,W)
    I = np.zeros([W*W, N])
    for i1 in range(W):
        for i2 in range(W):
            I[i1*W+i2,:]+=log_normal(y,shift[i1],sigma[i2])

        
    #print (I)
    #print (np.shape(I))
    I_normalized = I / I.sum(axis=1, keepdims=True)
    print (I_normalized)

    
    m = N 
    x = np.zeros((1, n, W*W))
    l=30
    for i in range(1):
        y_diff = (y[i*m:(i+1)*m]) /l
        y_diff_matrix = np.array([y_diff**j for j in range(n)])
        x[i, :, :] = np.dot(y_diff_matrix, I_normalized[:, i*m:(i+1)*m].T)
    #print (x[:,:,0])    
    # Calculate d using matrix operations
    d = np.mean(x, axis=2).T
    print (d,'d')
    # Reshape x and calculate g
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, W*W)
    g = np.dot(x_reshaped, x_reshaped.T) / W/W

    return d, g,x


def beta_fun(u,alpha,beta):
    return u**(alpha-1)*(1-u)**(beta-1)


def gd_generation_beta(W, N, M, n):
    
    y = np.linspace(0+1e-8, 1-1e-8, N)
    
    alpha=np.linspace(2,3,W)
    beta=np.linspace(2,3,W)
    I = np.zeros([W*W, N])
    for i1 in range(W):
        for i2 in range(W):
            I[i1*W+i2,:]+=beta_fun(y,alpha[i1],beta[i2])

        
    #print (I)
    #print (np.shape(I))
    I_normalized = I / I.sum(axis=1, keepdims=True)
    print (I_normalized)


    x = np.zeros((1,n, W*W))

    y_diff = y-1/2
    y_diff_matrix = np.array([y_diff**j for j in range(n)])
    x[0, :, :] = np.dot(y_diff_matrix, I_normalized[:, :].T)
    
    #print (x[:,:,0])    
    # Calculate d using matrix operations
    d = np.mean(x, axis=2).T
    print (d,'d')
    # Reshape x and calculate g
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, W*W)
    g = np.dot(x_reshaped, x_reshaped.T) / W/W

    return d, g,x

def gd_generation_compact(W, N, M, n, L, alpha):
    
    
    
    I = np.random.rand(W, N)
    I_normalized = I / I.sum(axis=1, keepdims=True)
    print (I_normalized)
    l = L / M
    
    #y = np.linspace(-L/2, L/2, N)
    y0_temp = np.linspace(-L/2, L/2, M + 1)
    y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

    m = N // M
    
    size=l*alpha
    
    y=np.zeros(N)
    for i in range(M):
        y[i*m:i*m+m//2]=np.linspace(-size/2, -size/2+size*0.00001, m//2)+y0[i]
        y[i*m+m//2:i*m+m]=np.linspace(size/2-size*0.00001, size/2, m//2)+y0[i]
    y=y+size/2
    x = np.zeros((M, n, W))
    
    for i in range(M):
        y_diff = (y[i*m:(i+1)*m] - y0[i]) / size
        y_diff_matrix = np.array([y_diff**j for j in range(n)])
        x[i, :, :] = np.dot(y_diff_matrix, I_normalized[:, i*m:(i+1)*m].T)
    #print (x[:,:,0])    
    # Calculate d using matrix operations
    d = np.mean(x, axis=2).T
    
    # Reshape x and calculate g
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, W)
    g = np.dot(x_reshaped, x_reshaped.T) / W

    return d, g,x

def gd_generation_analytical(W,M,n):
    alpha=np.linspace(0.01,3,W)

    x=np.zeros([M,n,W])
    for i in range(W):
        for k in range(n):
            for j in range(M):
                x[j,k,i]=(k+1)**(-alpha[i]-j)
    
    x/=M
    
    #print (x[:,:,0])
    d = np.zeros([n,M])
    for i in range(M):
        for j in range(n):
            for w in range(W):
                d[j,i]+=x[i,j,w]/W
            
    
    g=np.zeros([n*M,n*M])
    for i1 in range(M):
        for i2 in range(M):
            for j1 in range(n):
                for j2 in range(n):
                    for w in range(W):
                        g[j1*M+i1,j2*M+i2]+=x[i1,j1,w]*x[i2,j2,w]/W
                        
    return d, g,x

def gd_generation_point(W,M,n):
    
    
    position1=np.linspace(1/4,1/2,W)
    position2=np.linspace(1/4,1/2,W)
    I=np.linspace(0,1,W)
    x=np.zeros([M,n,2*W])
    for i in range(W):
        for k in range(n):
            for j in range(M):
                if j==0:
                    x[j,k,i]=position1[i]**k*I[i]
                    x[j,k,i+W]=position1[i]**k*(-1)**k*(1-I[i])
                if j==1:
                    x[j,k,i]=position2[i]**k*(-1)**k*(1-I[i])
                    x[j,k,i+W]=position2[i]**k*I[i]
    #x/=M
    
    #print (x[:,:,0])
    d = np.zeros([n,M])
    for i in range(M):
        for j in range(n):
            for w in range(2*W):
                d[j,i]+=x[i,j,w]/W/2
            
    g=np.zeros([n*M,n*M])
    for i1 in range(M):
        for i2 in range(M):
            for j1 in range(n):
                for j2 in range(n):
                    for w in range(2*W):
                        g[j1*M+i1,j2*M+i2]+=x[i1,j1,w]*x[i2,j2,w]/W/2
                        
    return d,g,x

def project_onto_plane(vectors, w):
    V = np.column_stack(vectors)
    Q, R = qr(V, mode='economic')
    c = np.linalg.solve(R, Q.T @ w)
    projection = V @ c
    return projection

def projection_length(vectors, w):
    projection = project_onto_plane(vectors, w)
    length = np.linalg.norm(projection)
    return length

#product is an array previous[k,j] is the inner product w_k^T v_j
def inner_product(vectors):
    V = np.column_stack(vectors)
    Q, R = qr(V, mode='economic')
    product = Q.T @ V
    #print (Q.T@Q)
    return product


#product is an array previous[k,j] is the inner product w_k^T v_j
def inner_product_new(vectors,nmax,M,M0):
    M_M0_r=int(M/M0)
    vec_list=[]
    for i in range(nmax+1):
        #print (i)
        temp_list=vectors[0:i*M]
        l=len(vec_list)
        for j in range(l):
            temp_list.append(vec_list[j])
        for j in range(M0):
            temp_list.append(vectors[i*M+j*M_M0_r+int(M_M0_r/2)])
        V = np.column_stack(temp_list)
        Q, R = qr(V, mode='economic')
        for j in range(M0):
            vec_list.append(Q[:,-M0+j])
        #print (i,len(vec_list))
    #print (np.shape(np.column_stack(vec_list)),np.shape(np.column_stack(vectors)))
    #A=np.column_stack(vec_list)
    #print (np.shape(A))
    #(m,n)=np.shape(A)
    #for i in range(n):
        #for j in range(i,n):
            #print ('error inner product new',i,j,A[:,i]@A[:,j])
            #if np.abs(A[:,i]@A[:,j])>1e-13:
                #print ('error inner product new',i,j,A[:,i]@A[:,j])
    product = np.column_stack(vec_list).T @ np.column_stack(vectors)
    #print (Q.T@Q)
    return product

def inner_product_directSPADE(vectors,nmax,M):
    V = np.column_stack(vectors)
    product=np.zeros([(nmax+1)*M,(nmax+1)*M])
    for i in range(M):
        vectors_temp=[]
        for j in range(nmax+1):
            vectors_temp.append(vectors[i+M*j])
        V_temp = np.column_stack(vectors_temp)
        Q, R = qr(V_temp, mode='economic')
        product_temp = Q.T @ V
        for j in range(nmax+1):
            product[i+M*j,:]=product_temp[j,:]
    return product


def psi_fun(y, y0, sigma, n):
    Hn = hermite(n)
    result = 1 / (2 * np.pi * sigma**2)**(1/4) * (-1)**n * Hn((y - y0) / (2 * sigma)) * np.exp(-(y - y0)**2 / (4 * sigma**2)) / (2 * sigma)**n
    return result


#y0_ should be between [-L/2,L/2] and of size M
def psi_array_fun(L, Lmax, sigma, N, M, nmax,y0):
    result = np.zeros([N, M, nmax + 1])
    y = np.linspace(-Lmax/2, Lmax/2, N)
    #y0=L*y0_
    #y0_temp = np.linspace(-L/2, L/2, 1+M)
    #y0=np.zeros(M)
    #for i in range(M):
        #y0[i]=y0_temp[i]/2+y0_temp[i+1]/2
    for i in range(nmax + 1):
        for j in range(M):
            result[:, j, i] = psi_fun(y, y0[j], sigma, i)*np.sqrt(Lmax/N)
            #if i==0: #only if we didn't take derivative, it is normalized
                #print (np.linalg.norm(result[:,j,i])**2)
            #result[:,j,i]=result[:,j,i]/np.linalg.norm(result[:,j,i])
    return result


#b_j^l l<=n  j<=M
#M,nmax are parameter used for generating product
def a_coeff_fun(product,M,nmax,sigma):
    n=nmax+1
    result=np.zeros([n,M,n,M])
    for i1 in range(n):
        for i2 in range(n):
            for j1 in range(M):
                for j2 in range(M):
                    right=i1*M+j1
                    left=i2*M+j2
                    result[i1,j1,i2,j2]=product[left,right] * sigma**i1 /math.factorial(i1)
    return result

def a_coeff_new_fun(product,M,M0,nmax,sigma):
    n=nmax+1
    result=np.zeros([n,M,n,M0])
    for i1 in range(n):
        for i2 in range(n):
            for j1 in range(M):
                for j2 in range(M0):
                    right=i1*M+j1
                    left=i2*M0+j2
                    result[i1,j1,i2,j2]=product[left,right] * sigma**i1 /math.factorial(i1)
    return result

def C_coeff_new(n,M,M0,a):
    result=np.zeros([n,2*M0,n,M])
    for l in range(n):
        for p in range(n):
            for j in range(M0):
                for k in range(M):
                    for m in range(p+1):
                        s=p-m
                        #print (p,m,s,l)
                        result[l,2*j,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j,p,k]+=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]-=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        #Note in function a_coeff_fun we use right=i1*M+j1, which already includes a transpose for a
                        
                        #print (l,j,p,k,m,result[l,2*j,p,k])
                        #if l==0 and p==2:
                            #print (l,j,p,k,m,result[l,2*j,p,k])
    return result/2#/M


def C_coeff(n,M,a):
    result=np.zeros([n,2*M,n,M])
    for l in range(n):
        for p in range(n):
            for j in range(M):
                for k in range(M):
                    for m in range(p+1):
                        s=p-m
                        #print (p,m,s,l)
                        result[l,2*j,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j,p,k]+=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]-=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        #Note in function a_coeff_fun we use right=i1*M+j1, which already includes a transpose for a
                        
                        #print (l,j,p,k,m,result[l,2*j,p,k])
                        #if l==0 and p==2:
                            #print (l,j,p,k,m,result[l,2*j,p,k])
    return result/2/2#/M

def C_SPADE_coeff(n,M,a):
    result=np.zeros([n,2*M,n,M])
    for l in range(n):
        for p in range(n):
            for j in range(M):
                for k in range(M):
                    for m in range(p+1):
                        s=p-m
                        #print (p,m,s,l)
                        result[l,2*j,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j,p,k]+=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]-=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        #Note in function a_coeff_fun we use right=i1*M+j1, which already includes a transpose for a
                        
                        #print (l,j,p,k,m,result[l,2*j,p,k])
                        #if l==0 and p==2:
                            #print (l,j,p,k,m,result[l,2*j,p,k])
    return result/2/M/2

def C_SPADE_coeff_extra(n,M,a):
    result=np.zeros([M,n,M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s=p-m
                    result[j,p,q]+=a[m,q,0,j]*a[s,q,0,j].conjugate()
    return result/M/2

def C_new_coeff_extra(n,M,a):
    result=np.zeros([M,n,M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s=p-m
                    result[j,p,q]+=a[m,q,0,j]*a[s,q,0,j].conjugate()
    return result/2


def C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma):
    u_array=np.linspace(-L_measure/2,L_measure/2,N_measure)
    result=np.zeros([N_measure,n,M])
    for i in range(n):
        for k in range(M):
            for p in range(i+1):
                q=i-p
                
                #print (np.shape(result[:,i,k]),result[:,i,k].dtype)
                temp=psi_fun(u_array,y0[k],sigma,p)*np.conjugate(psi_fun(u_array,y0[k],sigma,q))/math.factorial(p)/math.factorial(q)*sigma**i
                
                #print (np.shape(temp),temp.dtype)
                #print (temp+result[:,i,k])
                result[:,i,k]+=temp.astype(np.float64)
                #result[:,i,k]+=psi_fun(u_array,y0[k],sigma,p)*np.conjugate(psi_fun(u_array,y0[k],sigma,q))/math.factorial(p)/math.factorial(q)*sigma**i
    return result*L_measure/N_measure  #normalization such that np.sum(result[:,0,0])=1


def integrate_psi(u,y0,sigma,p,q,i,k):
    result=psi_fun(u,y0,sigma,p)*np.conjugate(psi_fun(u,y0,sigma,q))/math.factorial(p)/math.factorial(q)*sigma**i
    return result

def C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma):
    
    u_temp = np.linspace(-L_measure/2,L_measure/2,N_measure+1)
    u_array = (u_temp[:-1] + u_temp[1:]) / 2
    #print (u_array,l)
    #u_array=np.linspace(-L_measure/2,L_measure/2,N_measure)
    l=L_measure/N_measure
    result=np.zeros([N_measure,n,M])
    for i in range(n):
        for k in range(M):
            for p in range(i+1):
                q=i-p
                temp=np.zeros(N_measure)
                for j in range(N_measure):
                    #print (j,u_array[j]-l/2,u_array[j]+l/2)
                    #temp1,err=integrate.quad(psi_fun,u_array[j]-l/2,u_array[j]+l/2,args=(y0[k],sigma,p))
                    #temp2,err=integrate.quad(psi_fun,u_array[j]-l/2,u_array[j]+l/2,args=(y0[k],sigma,q))
                    #temp[j]=temp1*np.conjugate(temp2)/math.factorial(p)/math.factorial(q)*sigma**i
                    temp[j],error=integrate.quad(integrate_psi,u_array[j]-l/2,u_array[j]+l/2,args=(y0[k],sigma,p,q,i,k))
                    
                    #print (result)
                #print (np.shape(result[:,i,k]),result[:,i,k].dtype)
                #temp=psi_fun(u_array,y0[k],sigma,p)*np.conjugate(psi_fun(u_array,y0[k],sigma,q))/math.factorial(p)/math.factorial(q)*sigma**i
                #temp1=integral.quad(psi_fun)
                #print (np.shape(temp),temp.dtype)
                #print (temp+result[:,i,k])
                result[:,i,k]+=temp.astype(np.float64)
                #result[:,i,k]+=psi_fun(u_array,y0[k],sigma,p)*np.conjugate(psi_fun(u_array,y0[k],sigma,q))/math.factorial(p)/math.factorial(q)*sigma**i
    return result#*L_measure/N_measure  #normalization such that np.sum(result[:,0,0])=1



def random_gd(n,M):
    
    g=np.random.random([n*M,n*M])
    g=g+g.T
    eigenvalues = np.linalg.eigvals(g)
    min_eigenvalue = np.min((eigenvalues))
    #epsilon = 1e-3  # Small positive number to ensure strict positive definiteness
    g += np.eye(n*M) * (np.abs(min_eigenvalue )+1e-6)
    g=g/np.max(g)
    
    max_eigenvalue = np.max(np.linalg.eigvals(g))
    g=g/max_eigenvalue/2

    d = np.random.random([n,M])
    return g,d

def GD_new(g,d,C,alpha,n,M,M0):

    D=np.zeros([n*2*M0,n*2*M0])
    G=np.zeros([n*2*M0,n*2*M0])
    for m in range(n):
        for k in range(M):
            #print (np.shape(D),np.shape(C[:,:,m,k].flatten()),np.shape(np.diag(C[:,:,m,k].flatten())))
            D+=alpha**m * d[m,k]*np.diag(C[:,:,m,k].flatten())
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    #print (m,k1,k2)
                    m2=m-m1
                    left=np.array([C[:,:,m1,k1].flatten()]).T
                    right=np.array([C[:,:,m2,k2].flatten()])
                    index1=m1*M+k1
                    index2=m2*M+k2
                    G+=alpha**m * g[index1,index2] * left@right
    return G,D

def GD(g,d,C,alpha,n,M,C_extra):

    D=np.zeros([n*2*M+M,n*2*M+M])
    G=np.zeros([n*2*M+M,n*2*M+M])
    for m in range(n):
        for k in range(M):
            #print (m,n,k,M,'label')
            #print (np.shape(D),np.shape(C[:,:,m,k].flatten()),np.shape(np.diag(C[:,:,m,k].flatten())))
            D+=alpha**m * d[m,k]*np.diag(np.append(C[:,:,m,k].flatten(),C_extra[:,m,k]))
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    #print (m,k1,k2)
                    m2=m-m1
                    left=np.array([np.append(C[:,:,m1,k1].flatten(),C_extra[:,m1,k1])]).T
                    right=np.array([np.append(C[:,:,m2,k2].flatten(),C_extra[:,m2,k2])])
                    index1=m1*M+k1
                    index2=m2*M+k2
                    G+=alpha**m * g[index1,index2] * left@right
    return G,D

def GD_directimaging(g,d,C,alpha,n,M,N_measure):
    D=np.zeros([N_measure,N_measure])
    G=np.zeros([N_measure,N_measure])
    for m in range(n):
        for k in range(M):
            #print (alpha)
            #print (d[m,k])
            #print (C[:,m,k])
            #print (m,n,k,M)
            D+=alpha**m * d[m,k] * np.diag(C[:,m,k])
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    m2=m-m1
                    left=np.array([C[:,m1,k1]]).T
                    right=np.array([C[:,m2,k2]])
                    index1=m1*M+k1
                    index2=m2*M+k2
                    G+=alpha**m * g[index1,index2] * left@right
    return G,D

def Dhalfinv_directimaging_fun(g,d,C,alpha,n,M,N_measure):
    result=np.zeros(N_measure)
    for m in range(n):
        for k in range(M):
            #print (np.shape(D),np.shape(C[:,:,m,k].flatten()),np.shape(np.diag(C[:,:,m,k].flatten())))
            result+=alpha**m * d[m,k]*C[:,m,k]
    for i in range(N_measure):
        if result[i]<0 and np.abs(result[i])<1e-14:
            result[i]+=1e-14
        elif result[i]<0 and np.abs(result[i])>1e-14:
            print ('Dhalfinv Error')
            
    #print (np.min(np.abs(result)),np.min(result))
    return np.diag(1/np.sqrt(result))

def Dhalfinv_fun(g,d,C,alpha,n,M,C_extra):
    result=np.zeros(n*2*M+M)
    for m in range(n):
        for k in range(M):
            #print (np.shape(D),np.shape(C[:,:,m,k].flatten()),np.shape(np.diag(C[:,:,m,k].flatten())))
            result+=alpha**m * d[m,k]*(np.append(C[:,:,m,k].flatten(),C_extra[:,m,k]))
    for i in range(n*2*M):
        if result[i]<0 and np.abs(result[i])<1e-14:
            result[i]+=1e-14
        elif result[i]<0 and np.abs(result[i])>1e-14:
            print ('Dhalfinv Error')
            
    #print (np.min(np.abs(result)),np.min(result))
    return np.diag(1/np.sqrt(result))


def Dhalfinv_fun_new(g,d,C,alpha,n,M,M0):
    result=np.zeros(n*2*M0)
    for m in range(n):
        for k in range(M):
            #print (np.shape(D),np.shape(C[:,:,m,k].flatten()),np.shape(np.diag(C[:,:,m,k].flatten())))
            result+=alpha**m * d[m,k]*C[:,:,m,k].flatten()
    for i in range(n*2*M0):
        if result[i]<0 and np.abs(result[i])<1e-14:
            result[i]+=1e-14
        elif result[i]<0 and np.abs(result[i])>1e-14:
            print ('Dhalfinv Error')
            
    #print (np.min(np.abs(result)),np.min(result))
    return np.diag(1/np.sqrt(result))

def main(G,D,Dhalfinv,S):

    M=Dhalfinv@G@Dhalfinv
    
    #l,v=eigh(M)
    #result=np.sort(np.real(np.real(l)))
    #print (result)
    

    l, v =eigh(M)
    

    sort_indices = np.argsort(np.abs(np.real(l)))
    

    sorted_l = l[sort_indices]
    sorted_v = v[:, sort_indices]
    
    M1=G+(D-G)/S
    M2=inv(M1)@G
    CT=np.matrix.trace(M2)
    return Dhalfinv,M,sorted_l,sorted_v,CT

def C_T(sorted_l,alpha,S):
    (K,)=np.shape(sorted_l)
    result=0
    beta_square=1/sorted_l-1
    for i in range(K):
        result+=1/(1+beta_square[-1-i]/S)
    return result


def Prob_direct_imaging(C1,alpha,x):
    (N_measure,n,M)=np.shape(C1)
    result=np.zeros(N_measure)
    for i in range(N_measure):
        for j in range(n):
            for k in range(M):
                result[i]+=C1[i,j,k]*x[k,j]*alpha**j
                #print (i,j,k,C1[i,j,k]*x[k,j]*alpha**j)
    return result

def Prob_array_direct_imaging(C1,alpha,x_array):
    (M,n,W_)=np.shape(x_array)
    #W=int(W_/2)
    (N_measure,n,M)=np.shape(C1)
    result=np.zeros([N_measure,W_])
    for  i in range(W_):
        result[:,i]=Prob_direct_imaging(C1,alpha,x_array[:,:,i])
    return result

def Prob_separate_SPADE(C2,C2_extra,alpha,x):
    (n,M_,n,M)=np.shape(C2)
    result=np.zeros(n*2*M+M)
    for j in range(n):
        for k in range(M*2):
            for m in range(M):
                for l in range(n):
                    result[k+j*M*2]+=C2[j,k,l,m]*x[m,l]*alpha**l
    for k in range(M):
        for m in range(M):
            for l in range(n):
                result[k+n*M*2]+=C2_extra[k,l,m]*x[m,l]*alpha**l
                
    return result

def Prob_array_separate_SPADE(C2,C2_extra,alpha,x_array):
    (M,n,W_)=np.shape(x_array)
    #W=int(W_/2)
    (n,M_,n,M)=np.shape(C2)
    result=np.zeros([n*2*M+M,W_])
    for  i in range(W_):
        result[:,i]=Prob_separate_SPADE(C2,C2_extra,alpha,x_array[:,:,i])
    return result

def Prob_orthogonal_SPADE(C3,C3_extra,alpha,x):
    (n,M_,n,M)=np.shape(C3)
    result=np.zeros(n*2*M+M)
    for j in range(n):
        for k in range(M*2):
            for m in range(M):
                for l in range(n):
                    result[k+j*M*2]+=C3[j,k,l,m]*x[m,l]*alpha**l
    for k in range(M):
        for m in range(M):
            for l in range(n):
                result[k+n*M*2]+=C3_extra[k,l,m]*x[m,l]*alpha**l
                
    return result

def Prob_array_orthogonal_SPADE(C3,C3_extra,alpha,x_array):
    (M,n,W_)=np.shape(x_array)
    #W=int(W_/2)
    (n,M_,n,M)=np.shape(C3)
    result=np.zeros([n*2*M+M,W_])
    for  i in range(W_):
        result[:,i]=Prob_separate_SPADE(C3,C3_extra,alpha,x_array[:,:,i])
    return result



def train_classifier_simple(P, r, fitting_order):
    temp_,W_=np.shape(P)
    #print ('here',temp_,W_)
    W=W_//2
    P1=P[:,:W]
    P1=P1.T
    P2=P[:,W:]
    P2=P2.T
    
    training_data1_=(P1@r).T
    training_data2_=(P2@r).T
    #print (np.shape(training_data1_))
    #print (training_data1_[-1,:])
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

#new data is the probabilitiy distribution array
def classify_new_data_simple(clf, scaler, new_data, r, fitting_order):
    #print (np.shape(new_data),np.shape(r),'here2')
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

#def sample_and_estimate_distribution_fast(P_test, sam_num):

    # Normalize the input distribution
    #if np.abs(np.sum(P_test[P_test < 0]))>1e-14:
        #print (P_test[P_test < 0],'Probability distribution smaller than 0')
    #P_test[P_test < 0] = 0
    
    #P = P_test / np.sum(P_test)
    
    # Draw counts via a single multinomial sample
    #counts = np.random.multinomial(sam_num, P)
    #print (counts)
    # Convert counts to empirical probabilities
    #return counts / sam_num

# maximum chunk size we can safely pass to np.random.multinomial
_MAX_CHUNK = np.iinfo(np.int32).max

def sample_and_estimate_distribution_fast(P_test, sam_num):
    """
    Draw sam_num total multinomial samples from P_test, by splitting into
    chunks of at most _MAX_CHUNK draws to avoid overflow.
    Returns the empirical distribution counts / sam_num.
    """
    # ensure P_test is non‐negative and sums to 1
    P = np.clip(P_test, 0, None)
    total = P.sum()
    if total == 0:
        raise ValueError("Empty distribution passed to sampling")
    P = P / total

    sam = int(sam_num)
    counts = np.zeros_like(P, dtype=np.int64)

    # how many full‐size chunks?
    full_chunks = sam // _MAX_CHUNK
    rem        = sam - full_chunks * _MAX_CHUNK

    # sample full chunks
    for _ in range(full_chunks):
        counts += np.random.multinomial(_MAX_CHUNK, P)

    # sample the remainder
    if rem > 0:
        counts += np.random.multinomial(rem, P)

    return counts / sam



def test_error(clf, scaler,fitting_order,P_test, sam_num, r):

    _,W_=np.shape(P_test)
    W=int(W_//2)
    count0=0
    count1=0
    for i in range(W):
        P_test_temp=P_test[:,i]
        P_est = sample_and_estimate_distribution_fast(P_test_temp, sam_num)
        predicted_class = classify_new_data_simple(clf, scaler, P_est,r,fitting_order)
        #print("Predicted class 0 (sampled):", predicted_class)
        if predicted_class==0:
            count0+=1
        
        #print (((P_est@r)/(P_test@r))[-fitting_order:])       
        
        P_test_temp=P_test[:,i+W]
        P_est = sample_and_estimate_distribution_fast(P_test_temp, sam_num)
        predicted_class = classify_new_data_simple(clf, scaler, P_est,r,fitting_order)
        #print("Predicted class 1 (sampled):", predicted_class)
        if predicted_class==1:
            count1+=1
    
        #print (((P_est@r)/(P_test@r))[-fitting_order:])       
    return count0, count1




def repeat_success_rate_evaluation_multiple(P, r, total_task, repeats, sam_nums, W_test,M,n,L,classify_P,C1,C2,C2_extra,C3,C3_extra,alpha,which_case,which_alpha):
    success_rates = {sam_num: np.zeros((repeats, total_task)) for sam_num in sam_nums}
    #print (np.shape(r),'here3')
    for i in range(total_task):
        T=time.time()
        fitting_order=i+1
        clf,scaler=train_classifier_simple(P, r, fitting_order)
        for s in range(repeats):
            for sam_num in sam_nums:
                #print(f"Testing with sam_num = {sam_num}")
            
            

                d_test,g_test,x_test=gd_generation_fast_classify(W_test, M*20, M, n, L,classify_P)
                if which_case=='DI':
                    P_test=Prob_array_direct_imaging(C1,alpha[which_alpha],x_test)
                elif which_case=='sSPADE':
                    P_test=Prob_array_separate_SPADE(C2,C2_extra,alpha[which_alpha],x_test)
                elif which_case=='oSPADE':
                    P_test=Prob_array_orthogonal_SPADE(C3,C3_extra,alpha[which_alpha],x_test)
                    
                count0, count1=test_error(clf, scaler,fitting_order,P_test, sam_num, r)
                #print (which_case,i,s,count0,count1)

                success_rates[sam_num][s, i] = (count0 + count1) / (2 * W_test)
        print (i,time.time()-T)
        T=time.time()
            
    return success_rates





# 1) Top‐level worker
def _eval_order_global(i,
                       P, r,
                       repeats, sam_nums, W_test, M, n, L,
                       classify_P, C1, C2, C2_extra, C3, C3_extra,
                       alpha, which_case, which_alpha):
    """
    Compute success‐rates for fitting_order = i+1.
    Returns (i, {sam_num: array_of_length_repeats})
    """
    fo = i + 1
    # train once for this fitting_order
    clf, scaler = train_classifier_simple(P, r, fo)

    rates = {sam: np.zeros(repeats) for sam in sam_nums}

    for s in range(repeats):
        # fresh test data
        _, _, x_test = gd_generation_fast_classify(
            W_test, M*20, M, n, L, classify_P
        )

        # build P_test
        if which_case == 'DI':
            P_test = Prob_array_direct_imaging(C1, alpha[which_alpha], x_test)
        elif which_case == 'sSPADE':
            P_test = Prob_array_separate_SPADE(C2, C2_extra,
                                               alpha[which_alpha], x_test)
        else:  # 'oSPADE'
            P_test = Prob_array_orthogonal_SPADE(C3, C3_extra,
                                                 alpha[which_alpha], x_test)

        # evaluate for each sample size
        for sam in sam_nums:
            c0, c1 = test_error(clf, scaler, fo, P_test, sam, r)
            rates[sam][s] = (c0 + c1) / (2 * W_test)

    return i, rates


# 2) Parallel wrapper
def parallel_success_rates(
    P, r, total_task, repeats, sam_nums, W_test,
    M, n, L, classify_P, C1, C2, C2_extra, C3, C3_extra,
    alpha, which_case, which_alpha
):
    success = {sam: np.zeros((repeats, total_task)) for sam in sam_nums}

    with ProcessPoolExecutor() as exe:
        futures = [
            exe.submit(
                _eval_order_global,
                i, P, r, repeats, sam_nums, W_test,
                M, n, L, classify_P,
                C1, C2, C2_extra, C3, C3_extra,
                alpha, which_case, which_alpha
            )
            for i in range(total_task)
        ]

        for fut in tqdm(as_completed(futures),
                        total=total_task,
                        desc="Parallel tasks",
                        unit="task"):
            i, rates_i = fut.result()
            for sam, arr in rates_i.items():
                success[sam][:, i] = arr

    return success


    
if __name__ == '__main__':
    #multiprocessing.set_start_method('fork')

    
    L =10
    Lmax=20
    sigma = 1
    N = 2000
    M = 4
    nmax = 150
    n=20
    W=50
    S=1000
    N_measure=60
    L_measure=L+10*sigma
    Q=21
    S=np.logspace(2,14,Q)
    alpha=0.1
    t0=time.time()
    
    
    y0=np.array([-0.4, 0.1, 0.2,0.35])*L
    #y0_temp = np.linspace(-L/2, L/2, M + 1)
    #y0 = (y0_temp[:-1] + y0_temp[1:]) / 2
    
    
    
    
    psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax,y0)
    vectors = []
    for j in range(nmax + 1):
        for i in range(M):
            vectors.append(psi_array[:, i, j])
    #g,d=random_gd(n,M)
    #d,g,x=gd_generation_fast(W, M*20, M, n, L)
    d,g,x=gd_generation_fast_classify(W, M*20, M, n, L,classify_P4)
    print ('start')
    
    #C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
    C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
    l1=np.zeros([N_measure,Q])
    r1=np.zeros([N_measure,N_measure,Q])
    M_array=np.zeros([N_measure,N_measure,Q])
    D_halfinv_array=np.zeros([N_measure,N_measure,Q])
    CT_array_DI1=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD_directimaging(g,d,C1,alpha,n,M,N_measure)
        Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha,n,M,N_measure)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l1[:,i]=temp
        r1[:,:,i]=D_halfinv_temp@v_temp
        #r1[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_DI1[i]=CT_temp
    
    t1=time.time()
    print ('direct imaging',t1-t0)   
    
    
    
    product2 = inner_product_directSPADE(vectors,nmax,M)
    a2= a_coeff_fun(product2,M,nmax,sigma)
    C2=C_SPADE_coeff(n, M, a2)
    C2_extra=C_SPADE_coeff_extra(n, M, a2)
    l2=np.zeros([2*n*M+M,Q])
    r2=np.zeros([2*n*M+M,2*n*M+M,Q])
    M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    CT_array_sSPADE1=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD(g,d,C2,alpha,n,M,C2_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C2,alpha,n,M,C2_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l2[:,i]=temp
        r2[:,:,i]=D_halfinv_temp@v_temp
        #r2[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_sSPADE1[i]=CT_temp
    
    #print (r[:,-1,-1])
    #print (r[:,-2,-1])
    print (np.sum(D))
    
    
    t2=time.time()
    print ('SPADE',t2-t1)     
        
        
    product3 = inner_product(vectors)
    a3= a_coeff_fun(product3,M,nmax,sigma)
    C3=C_coeff(n,M,a3)
    C3_extra=C_new_coeff_extra(n, M, a3)
    l3=np.zeros([2*n*M+M,Q])
    r3=np.zeros([2*n*M+M,2*n*M+M,Q])
    M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    CT_array_oSPADE1=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD(g,d,C3,alpha,n,M,C3_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C3,alpha,n,M,C3_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l3[:,i]=temp
        r3[:,:,i]=D_halfinv_temp@v_temp
        #r[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_oSPADE1[i]=CT_temp
    
    print (np.sum(D))
    t3=time.time()
    print ('new method',t3-t2)  
    
    
    
    
    alpha=1e-2

    #C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
    C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
    l1=np.zeros([N_measure,Q])
    r1=np.zeros([N_measure,N_measure,Q])
    M_array=np.zeros([N_measure,N_measure,Q])
    D_halfinv_array=np.zeros([N_measure,N_measure,Q])
    CT_array_DI2=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD_directimaging(g,d,C1,alpha,n,M,N_measure)
        Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha,n,M,N_measure)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l1[:,i]=temp
        r1[:,:,i]=D_halfinv_temp@v_temp
        #r1[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_DI2[i]=CT_temp
    
    t1=time.time()
    print ('direct imaging',t1-t0)   
    
    
    
    product2 = inner_product_directSPADE(vectors,nmax,M)
    a2= a_coeff_fun(product2,M,nmax,sigma)
    C2=C_SPADE_coeff(n, M, a2)
    C2_extra=C_SPADE_coeff_extra(n, M, a2)
    l2=np.zeros([2*n*M+M,Q])
    r2=np.zeros([2*n*M+M,2*n*M+M,Q])
    M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    CT_array_sSPADE2=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD(g,d,C2,alpha,n,M,C2_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C2,alpha,n,M,C2_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l2[:,i]=temp
        r2[:,:,i]=D_halfinv_temp@v_temp
        #r2[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_sSPADE2[i]=CT_temp
    
    #print (r[:,-1,-1])
    #print (r[:,-2,-1])
    print (np.sum(D))
    
    
    t2=time.time()
    print ('SPADE',t2-t1)     
        
        
    product3 = inner_product(vectors)
    a3= a_coeff_fun(product3,M,nmax,sigma)
    C3=C_coeff(n,M,a3)
    C3_extra=C_new_coeff_extra(n, M, a3)
    l3=np.zeros([2*n*M+M,Q])
    r3=np.zeros([2*n*M+M,2*n*M+M,Q])
    M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    CT_array_oSPADE2=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD(g,d,C3,alpha,n,M,C3_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C3,alpha,n,M,C3_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l3[:,i]=temp
        r3[:,:,i]=D_halfinv_temp@v_temp
        #r[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_oSPADE2[i]=CT_temp
    
    print (np.sum(D))
    t3=time.time()
    print ('new method',t3-t2)             


    alpha=1e-3
    #C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
    C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
    l1=np.zeros([N_measure,Q])
    r1=np.zeros([N_measure,N_measure,Q])
    M_array=np.zeros([N_measure,N_measure,Q])
    D_halfinv_array=np.zeros([N_measure,N_measure,Q])
    CT_array_DI3=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD_directimaging(g,d,C1,alpha,n,M,N_measure)
        Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha,n,M,N_measure)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l1[:,i]=temp
        r1[:,:,i]=D_halfinv_temp@v_temp
        #r1[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_DI3[i]=CT_temp
    
    t1=time.time()
    print ('direct imaging',t1-t0)   
    
    
    
    product2 = inner_product_directSPADE(vectors,nmax,M)
    a2= a_coeff_fun(product2,M,nmax,sigma)
    C2=C_SPADE_coeff(n, M, a2)
    C2_extra=C_SPADE_coeff_extra(n, M, a2)
    l2=np.zeros([2*n*M+M,Q])
    r2=np.zeros([2*n*M+M,2*n*M+M,Q])
    M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    CT_array_sSPADE3=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD(g,d,C2,alpha,n,M,C2_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C2,alpha,n,M,C2_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l2[:,i]=temp
        r2[:,:,i]=D_halfinv_temp@v_temp
        #r2[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_sSPADE3[i]=CT_temp
    
    #print (r[:,-1,-1])
    #print (r[:,-2,-1])
    print (np.sum(D))
    
    
    t2=time.time()
    print ('SPADE',t2-t1)     
        
        
    product3 = inner_product(vectors)
    a3= a_coeff_fun(product3,M,nmax,sigma)
    C3=C_coeff(n,M,a3)
    C3_extra=C_new_coeff_extra(n, M, a3)
    l3=np.zeros([2*n*M+M,Q])
    r3=np.zeros([2*n*M+M,2*n*M+M,Q])
    M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])
    CT_array_oSPADE3=np.zeros(Q)
    for i in range(Q):
    
        G,D=GD(g,d,C3,alpha,n,M,C3_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C3,alpha,n,M,C3_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
        l3[:,i]=temp
        r3[:,:,i]=D_halfinv_temp@v_temp
        #r[:,:,i]=v_temp
        M_array[:,:,i]=M_temp
        D_halfinv_array[:,:,i]=D_halfinv_temp
        CT_array_oSPADE3[i]=CT_temp
    
    print (np.sum(D))
    t3=time.time()
    print ('new method',t3-t2)  


    
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
    #plt.xlim([5,D_array[-1]+10])
    #plt.ylim([0,0.55])
    #plt.xlim([0,D_array[-1]+1])
    #plt.ylim([1e0,1e15])
    
    plt.semilogx(S,CT_array_DI1,linewidth=3,linestyle='-.',label='alpha=0.1')
    plt.semilogx(S,CT_array_DI2,linewidth=3,linestyle='--',label='alpha=0.01')
    plt.semilogx(S,CT_array_DI3,linewidth=3,linestyle='-',label='alpha=0.001')
    
    
    plt.legend(fontsize=20,loc='center left',bbox_to_anchor=(1.02, 0.5),ncol=1)
    #plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  
    
    
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('C_T_DI.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')


    
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
    #plt.xlim([5,D_array[-1]+10])
    #plt.ylim([0,0.55])
    #plt.xlim([0,D_array[-1]+1])
    #plt.ylim([1e0,1e15])
    
    plt.semilogx(S,CT_array_sSPADE1,linewidth=3,linestyle='-.',label='alpha=0.1')
    plt.semilogx(S,CT_array_sSPADE2,linewidth=3,linestyle='--',label='alpha=0.01')
    plt.semilogx(S,CT_array_sSPADE3,linewidth=3,linestyle='-',label='alpha=0.001')
    
    
    plt.legend(fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
    #plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  
    
    
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('C_T_sSPADE.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')


    
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
    #plt.xlim([5,D_array[-1]+10])
    #plt.ylim([0,0.55])
    #plt.xlim([0,D_array[-1]+1])
    #plt.ylim([1e0,1e15])
    
    plt.semilogx(S,CT_array_oSPADE1,linewidth=3,linestyle='-.',label='alpha=0.1')
    plt.semilogx(S,CT_array_oSPADE2,linewidth=3,linestyle='--',label='alpha=0.01')
    plt.semilogx(S,CT_array_oSPADE3,linewidth=3,linestyle='-',label='alpha=0.001')
    
    
    plt.legend(fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
    #plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  
    
    
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('C_T_oSPADE.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')
    
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
    #plt.xlim([5,D_array[-1]+10])
    #plt.ylim([0,0.55])
    #plt.xlim([0,D_array[-1]+1])
    #plt.ylim([1e0,1e15])
    plt.semilogx(S,CT_array_DI1,linewidth=3,linestyle='-',label='DI')
    plt.semilogx(S,CT_array_sSPADE1,linewidth=3,linestyle='--',label='separate SPADE')
    plt.semilogx(S,CT_array_oSPADE1,linewidth=3,linestyle='-.',label='orthogonalized SPADE')
    
    plt.legend(fontsize=20,loc='center left',bbox_to_anchor=(1.02, 0.5),ncol=1)
    
    

    
    
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('C_T_together.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')
    
    plt.show()

