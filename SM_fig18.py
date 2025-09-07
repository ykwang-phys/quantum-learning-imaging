#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 00:41:09 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 18:14:43 2025

@author: yunkaiwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 20:55:47 2025

@author: yunkaiwang
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.special import hermite
import math
from scipy.linalg import eig,eigh,inv
import time
from matplotlib.ticker import MultipleLocator
from scipy import integrate

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

def psi_array_fun(L, Lmax, sigma, N, M, nmax):
    result = np.zeros([N, M, nmax + 1])
    y = np.linspace(-Lmax/2, Lmax/2, N)
    y0_temp = np.linspace(-L/2, L/2, 1+M)
    y0=np.zeros(M)
    for i in range(M):
        y0[i]=y0_temp[i]/2+y0_temp[i+1]/2
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




'''

L =1
sigma = 1
Lmax=L*10

N = 20000
M = 1
nmax = 150
n=15
W=5
S=1000
N_measure=100

Q=3
alpha=np.logspace(-1,-0.3,Q)
t0=time.time()

y0_temp = np.linspace(-L/2, L/2, M + 1)
y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

L_measure=np.max(y0)-np.min(y0)+10*sigma


psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax)
vectors = []
for j in range(nmax + 1):
    for i in range(M):
        vectors.append(psi_array[:, i, j])
#g,d=random_gd(n,M)
d,g,x=gd_generation_fast(W, M*20, M, n, L)
print ('start')

#C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
l1=np.zeros([N_measure,Q])
r=np.zeros([N_measure,N_measure,Q])
M_array=np.zeros([N_measure,N_measure,Q])
D_halfinv_array=np.zeros([N_measure,N_measure,Q])

D_array=np.zeros([N_measure,N_measure,Q])
for i in range(Q):

    G,D=GD_directimaging(g,d,C1,alpha[i],n,M,N_measure)
    Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha[i],n,M,N_measure)
    D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S)
    D_array[:,:,i]=D
    l1[:,i]=temp
    r[:,:,i]=D_halfinv_temp@v_temp
    r[:,:,i]=v_temp
    M_array[:,:,i]=M_temp
    D_halfinv_array[:,:,i]=D_halfinv_temp
    
t1=time.time()
print ('direct imaging',t1-t0)   

val_g,vec_g=eigh(g)
C_matrix1=np.zeros([n,n,Q])
val_C_array1=np.zeros([n,Q])

val_guess1=np.zeros([n,Q])

prefactor1_actual=np.zeros([8,Q])
prefactor1_guess=np.zeros([8,Q])
for i in range(Q):
    C1_alpha=np.zeros([N_measure,n,M])
    
    
    for j in range(n):
        #print (np.shape(C1[:,j,0]*alpha[i]**j))
        C1_alpha[:,j,0]=C1[:,j,0]*alpha[i]**j  #we just discuss single compact source case
    
    C_matrix1[:,:,i]=C1_alpha[:,:,0].transpose()@np.linalg.inv(D_array[:,:,i])@C1_alpha[:,:,0]
    val_C_array1[:,i],vec_C1=eigh(C_matrix1[:,:,i])
    val_guess1[:,i]=val_C_array1[:,i]*val_g
    
    scaling_array1=alpha[i]**np.array([14,12,10,8,6,4,2,0])
    prefactor1_actual[:,i]=l1[-8:,i]/scaling_array1
    prefactor1_guess[:,i]=val_guess1[-8:,i]/scaling_array1
    #print (i,val_guess[:,i],l1[-n:,i])
    print(
        i,
        "\n", np.array2string(prefactor1_actual[:, i], formatter={'float_kind': lambda x: f"{x:.3g}"}),
        "\n", np.array2string(prefactor1_guess[:, i], formatter={'float_kind': lambda x: f"{x:.3g}"}),
        "\n", np.array2string(val_guess1[-8:, i] / l1[-8:, i], formatter={'float_kind': lambda x: f"{x:.3g}"})
    )

    


product2 = inner_product_directSPADE(vectors,nmax,M)
a2= a_coeff_fun(product2,M,nmax,sigma)
C2=C_SPADE_coeff(n, M, a2)
C2_extra=C_SPADE_coeff_extra(n, M, a2)
l2=np.zeros([2*n*M+M,Q])
r=np.zeros([2*n*M+M,2*n*M+M,Q])
M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])

D_array=np.zeros([2*n*M+M,2*n*M+M,Q])
for i in range(Q):

    G,D=GD(g,d,C2,alpha[i],n,M,C2_extra)
    D_array[:,:,i]=D
    Dhalfinv=Dhalfinv_fun(g,d,C2,alpha[i],n,M,C2_extra)
    D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S)
    l2[:,i]=temp
    r[:,:,i]=D_halfinv_temp@v_temp
    r[:,:,i]=v_temp
    M_array[:,:,i]=M_temp
    D_halfinv_array[:,:,i]=D_halfinv_temp


#print (r[:,-1,-1])
#print (r[:,-2,-1])
#print (np.sum(D))


t2=time.time()
#print ('SPADE',t2-t1)     
val_g,vec_g=eigh(g)

C_matrix2=np.zeros([n,n,Q])
val_C_array2=np.zeros([n,Q])

val_guess2=np.zeros([n,Q])

prefactor2_actual=np.zeros([8,Q])
prefactor2_guess=np.zeros([8,Q])
for i in range(Q):
    C2_alpha=np.zeros([2*n+1,n,M])
    
    for j in range(n):
        #print (np.shape(C1[:,j,0]*alpha[i]**j))
        C2_alpha[:(2*n),j,0]=C2[:,:,j,0].flatten()*alpha[i]**j  #we just discuss single compact source case
        #C2_alpha[n:(2*n),j,0]=C2[:,1,j,0]*alpha[i]**j
        C2_alpha[-1,j,0]=C2_extra[0,j,0]*alpha[i]**j
    C_matrix2[:,:,i]=C2_alpha[:,:,0].transpose()@np.linalg.inv(D_array[:,:,i])@C2_alpha[:,:,0]
    val_C_array2[:,i],vec_C2=eigh(C_matrix2[:,:,i])
    val_guess2[:,i]=val_C_array2[:,i]*val_g
    #print (i,val_guess[:,i],l1[-n:,i])
    
    scaling_array2=alpha[i]**np.array([8,6,6,4,4,2,2,0])
    prefactor2_actual[:,i]=l2[-8:,i]/scaling_array2
    prefactor2_guess[:,i]=val_guess2[-8:,i]/scaling_array2
    
    
    print(
        i,
        "\n", np.array2string(prefactor2_actual[:, i], formatter={'float_kind': lambda x: f"{x:.3g}"}),
        "\n", np.array2string(prefactor2_guess[:, i], formatter={'float_kind': lambda x: f"{x:.3g}"}),
        "\n", np.array2string(val_guess2[-8:, i] / l2[-8:, i], formatter={'float_kind': lambda x: f"{x:.3g}"})
    )

    
print (val_g)

matrix = val_C_array2[:,-1].reshape(1, -1)  # shape (1, n)

# Compute signed log
matrix_log_signed = np.sign(matrix) * np.log10(np.abs(matrix))

plt.figure(figsize=(8, 2))  # wider for a row

im = plt.imshow(matrix_log_signed, cmap='coolwarm', interpolation='nearest', aspect='auto')

cbar = plt.colorbar(im, shrink=0.8)
cbar.set_label("Signed log10 scale")

nrows, ncols = matrix.shape
for i in range(nrows):
    for j in range(ncols):
        val = matrix_log_signed[i, j]
        if np.isfinite(val):
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)

plt.xticks(range(ncols))
plt.yticks([])
plt.xlabel('Eigenvalue Index')
plt.title('Signed Logarithmic Eigenvalues (heatmap)')

plt.tight_layout()
plt.savefig("grid_d.png", dpi=300, bbox_inches='tight')
plt.show()


'''

'''

L =1
sigma = 1
Lmax=L*10

N = 200
M = 1
nmax = 150
n=15
W=5
S=1000
N_measure=100

P=1
Q=5
alpha=np.logspace(-2,-1,Q)
t0=time.time()

y0_temp = np.linspace(-L/2, L/2, M + 1)
y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

L_measure=np.max(y0)-np.min(y0)+10*sigma


psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax)
vectors = []
for j in range(nmax + 1):
    for i in range(M):
        vectors.append(psi_array[:, i, j])
#g,d=random_gd(n,M)

print ('start')


# pre-allocate with the right shapes:
val_g             = np.zeros((n,      P))


l1                = np.zeros((N_measure, Q,      P))
C_matrix1         = np.zeros((n,      n,      Q,      P))
val_C_array1      = np.zeros((n,      Q,      P))
val_guess1        = np.zeros((n,      Q,      P))
prefactor1_actual = np.zeros((8,      Q,      P))
prefactor1_guess  = np.zeros((8,      Q,      P))

l2               = np.zeros((2*n*M+M, Q,      P))
C_matrix2         = np.zeros((n,      n,      Q,      P))
val_C_array2      = np.zeros((n,      Q,      P))
val_guess2        = np.zeros((n,      Q,      P))
prefactor2_actual = np.zeros((8,      Q,      P))
prefactor2_guess  = np.zeros((8,      Q,      P))

    

for p in range(P):
    # --- 1) compute g/d/x and record the last-8 eigs of g ---
    d, g, x = gd_generation_fast(W, M*20, M, n, L)
    eigs_g, _ = eigh(g)
    val_g[:, p] = eigs_g
    for i in range(Q):




        # --- 2) build direct-imaging C1 and D, get the l1 spectrum ---
        C1 = C_directimaging_coeff_integral(n, M, N_measure, L_measure, y0, sigma)
        G, D = GD_directimaging(g, d, C1, alpha[i], n, M, N_measure)
        Dhalfinv, Mtemp, temp, vtemp, CT = main(G, D, Dhalfinv_directimaging_fun(g, d, C1, alpha[i], n, M, N_measure), S)

        # store the Fisher-matrix eigenvalues for this (i,p)
        l1[:, i, p] = temp

        # --- 3) form the C_matrix and get its eigenvalues ---
        #    here M=1 so C1[:,:,0] is (N_measure × n)
        C1_alpha = np.zeros((N_measure, n, M))
        for j in range(n):
            C1_alpha[:, j, 0] = C1[:, j, 0] * alpha[i]**j

        invD = np.linalg.inv(D)
        Cmat = C1_alpha[:, :, 0].T @ invD @ C1_alpha[:, :, 0]
        C_matrix1[:, :, i, p] = Cmat

        vals_C, vecs_C = eigh(Cmat)
        val_C_array1[:, i, p] = vals_C

        # --- 4) form the “guess” from the top-8 modes only ---
        guess8 = vals_C * val_g[:, p]
        val_guess1[:, i, p] = guess8

        # --- 5) prefactors ---
        scaling = alpha[i]**np.array([14,12,10,8,6,4,2,0])
        prefactor1_actual[:, i, p] = l1[-8:, i, p] / scaling
        prefactor1_guess[:,  i, p] = val_guess1[-8:, i, p] / scaling
        #print (p,i,'direct',np.round(prefactor1_guess[:,  i, p]/prefactor1_actual[:, i, p],3))
        
        
        
        
        
        product2 = inner_product_directSPADE(vectors,nmax,M)
        a2= a_coeff_fun(product2,M,nmax,sigma)
        C2=C_SPADE_coeff(n, M, a2)
        C2_extra=C_SPADE_coeff_extra(n, M, a2)

      
        G,D=GD(g,d,C2,alpha[i],n,M,C2_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C2,alpha[i],n,M,C2_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S)

        l2[:, i, p] = temp

        C2_alpha = np.zeros((2*n*M+M, n, M))

        for j in range(n):
            #print (np.shape(C1[:,j,0]*alpha[i]**j))
            C2_alpha[:(2*n),j,0]=C2[:,:,j,0].flatten()*alpha[i]**j  #we just discuss single compact source case
            #C2_alpha[n:(2*n),j,0]=C2[:,1,j,0]*alpha[i]**j
            C2_alpha[-1,j,0]=C2_extra[0,j,0]*alpha[i]**j
        Cmat=C2_alpha[:,:,0].transpose()@np.linalg.inv(D)@C2_alpha[:,:,0]
        C_matrix2[:,:,i,p]=Cmat


        vals_C, vecs_C = eigh(Cmat)
        val_C_array2[:, i, p] = vals_C

        # --- 4) form the “guess” from the top-8 modes only ---
        guess8 = vals_C * val_g[:, p]
        val_guess2[:, i, p] = guess8

        # --- 5) prefactors ---
        scaling = alpha[i]**np.array([8,6,6,4,4,2,2,0])
        prefactor2_actual[:, i, p] = l2[-8:, i, p] / scaling
        prefactor2_guess[:,  i, p] = val_guess2[-8:, i, p] / scaling
        print (p,i,'SPADE',np.round(prefactor2_guess[:,  i, p]/prefactor2_actual[:, i, p],3))        
        

# common settings
line_width = 2.5
font_size = 14
fig_size = (6,4)

plt.figure(figsize=fig_size)
plt.loglog(alpha, prefactor1_actual[-1,:,0], linestyle='-', color='red', linewidth=line_width, label='Actual -1')
plt.loglog(alpha, prefactor1_guess[-1,:,0], linestyle='--', color='red', linewidth=line_width, label='Guess -1')
plt.loglog(alpha, prefactor1_actual[-2,:,0], linestyle='-', color='blue', linewidth=line_width, label='Actual -2')
plt.loglog(alpha, prefactor1_guess[-2,:,0], linestyle='--', color='blue', linewidth=line_width, label='Guess -2')
plt.loglog(alpha, prefactor1_actual[-3,:,0], linestyle='-', color='orange', linewidth=line_width, label='Actual -3')
plt.loglog(alpha, prefactor1_guess[-3,:,0], linestyle='--', color='orange', linewidth=line_width, label='Guess -3')
#plt.loglog(alpha, prefactor1_actual[-4,:,0], linestyle='-', color='green', linewidth=line_width, label='Actual -4')
#plt.loglog(alpha, prefactor1_guess[-4,:,0], linestyle='--', color='green', linewidth=line_width, label='Guess -4')
plt.title('direct', fontsize=font_size)
plt.xlabel('', fontsize=font_size)
plt.ylabel('', fontsize=font_size)
plt.tick_params(axis='both', which='major', labelsize=font_size)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-2)
#plt.tight_layout()
plt.savefig('prefactor_alpha_direct.png', dpi=300)
plt.show()


plt.figure(figsize=fig_size)
plt.loglog(alpha, prefactor2_actual[-1,:,0], linestyle='-', color='red', linewidth=line_width, label='Actual -1')
plt.loglog(alpha, prefactor2_guess[-1,:,0], linestyle='--', color='red', linewidth=line_width, label='Guess -1')
plt.loglog(alpha, prefactor2_actual[-2,:,0], linestyle='-', color='blue', linewidth=line_width, label='Actual -2')
plt.loglog(alpha, prefactor2_guess[-2,:,0], linestyle='--', color='blue', linewidth=line_width, label='Guess -2')
plt.loglog(alpha, prefactor2_actual[-3,:,0], linestyle='-', color='orange', linewidth=line_width, label='Actual -3')
plt.loglog(alpha, prefactor2_guess[-3,:,0], linestyle='--', color='orange', linewidth=line_width, label='Guess -3')
#plt.loglog(alpha, prefactor2_actual[-4,:,0], linestyle='-', color='green', linewidth=line_width, label='Actual -4')
#plt.loglog(alpha, prefactor2_guess[-4,:,0], linestyle='--', color='green', linewidth=line_width, label='Guess -4')
plt.title('prefactor_alpha_SPADE', fontsize=font_size)
plt.xlabel('', fontsize=font_size)
plt.ylabel('', fontsize=font_size)
plt.tick_params(axis='both', which='major', labelsize=font_size)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-2)
#plt.tight_layout()
plt.savefig('prefactor_alpha_SPADE.png', dpi=300)
plt.show()


'''

L =1
sigma = 1
Lmax=L*10

N = 200
M = 1
nmax = 150
n=15
W=5
S=1000
N_measure=100



P=10
Q=1
alpha=np.logspace(-1,-1,Q)
t0=time.time()

y0_temp = np.linspace(-L/2, L/2, M + 1)
y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

L_measure=np.max(y0)-np.min(y0)+10*sigma


psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax)
vectors = []
for j in range(nmax + 1):
    for i in range(M):
        vectors.append(psi_array[:, i, j])
#g,d=random_gd(n,M)

print ('start')


# pre-allocate with the right shapes:
val_g             = np.zeros((n,      P))


l1                = np.zeros((N_measure, Q,      P))
C_matrix1         = np.zeros((n,      n,      Q,      P))
val_C_array1      = np.zeros((n,      Q,      P))
val_guess1        = np.zeros((n,      Q,      P))
prefactor1_actual = np.zeros((8,      Q,      P))
prefactor1_guess  = np.zeros((8,      Q,      P))

l2               = np.zeros((2*n*M+M, Q,      P))
C_matrix2         = np.zeros((n,      n,      Q,      P))
val_C_array2      = np.zeros((n,      Q,      P))
val_guess2        = np.zeros((n,      Q,      P))
prefactor2_actual = np.zeros((8,      Q,      P))
prefactor2_guess  = np.zeros((8,      Q,      P))

    

for p in range(P):
    # --- 1) compute g/d/x and record the last-8 eigs of g ---
    d, g, x = gd_generation_fast(W, M*20, M, n, L)
    eigs_g, _ = eigh(g)
    val_g[:, p] = eigs_g
    for i in range(Q):




        # --- 2) build direct-imaging C1 and D, get the l1 spectrum ---
        C1 = C_directimaging_coeff_integral(n, M, N_measure, L_measure, y0, sigma)
        G, D = GD_directimaging(g, d, C1, alpha[i], n, M, N_measure)
        Dhalfinv, Mtemp, temp, vtemp, CT = main(G, D, Dhalfinv_directimaging_fun(g, d, C1, alpha[i], n, M, N_measure), S)

        # store the Fisher-matrix eigenvalues for this (i,p)
        l1[:, i, p] = temp

        # --- 3) form the C_matrix and get its eigenvalues ---
        #    here M=1 so C1[:,:,0] is (N_measure × n)
        C1_alpha = np.zeros((N_measure, n, M))
        for j in range(n):
            C1_alpha[:, j, 0] = C1[:, j, 0] * alpha[i]**j

        invD = np.linalg.inv(D)
        Cmat = C1_alpha[:, :, 0].T @ invD @ C1_alpha[:, :, 0]
        C_matrix1[:, :, i, p] = Cmat

        vals_C, vecs_C = eigh(Cmat)
        val_C_array1[:, i, p] = vals_C

        # --- 4) form the “guess” from the top-8 modes only ---
        guess8 = vals_C * val_g[:, p]
        val_guess1[:, i, p] = guess8

        # --- 5) prefactors ---
        scaling = alpha[i]**np.array([14,12,10,8,6,4,2,0])
        prefactor1_actual[:, i, p] = l1[-8:, i, p] / scaling
        prefactor1_guess[:,  i, p] = val_guess1[-8:, i, p] / scaling
        #print (p,i,'direct',np.round(prefactor1_guess[:,  i, p]/prefactor1_actual[:, i, p],3))
        
        
        
        
        
        product2 = inner_product_directSPADE(vectors,nmax,M)
        a2= a_coeff_fun(product2,M,nmax,sigma)
        C2=C_SPADE_coeff(n, M, a2)
        C2_extra=C_SPADE_coeff_extra(n, M, a2)

      
        G,D=GD(g,d,C2,alpha[i],n,M,C2_extra)
        Dhalfinv=Dhalfinv_fun(g,d,C2,alpha[i],n,M,C2_extra)
        D_halfinv_temp,M_temp,temp,v_temp,CT_temp=main(G,D,Dhalfinv,S)

        l2[:, i, p] = temp

        C2_alpha = np.zeros((2*n*M+M, n, M))

        for j in range(n):
            #print (np.shape(C1[:,j,0]*alpha[i]**j))
            C2_alpha[:(2*n),j,0]=C2[:,:,j,0].flatten()*alpha[i]**j  #we just discuss single compact source case
            #C2_alpha[n:(2*n),j,0]=C2[:,1,j,0]*alpha[i]**j
            C2_alpha[-1,j,0]=C2_extra[0,j,0]*alpha[i]**j
        Cmat=C2_alpha[:,:,0].transpose()@np.linalg.inv(D)@C2_alpha[:,:,0]
        C_matrix2[:,:,i,p]=Cmat


        vals_C, vecs_C = eigh(Cmat)
        val_C_array2[:, i, p] = vals_C

        # --- 4) form the “guess” from the top-8 modes only ---
        guess8 = vals_C * val_g[:, p]
        val_guess2[:, i, p] = guess8

        # --- 5) prefactors ---
        scaling = alpha[i]**np.array([8,6,6,4,4,2,2,0])
        prefactor2_actual[:, i, p] = l2[-8:, i, p] / scaling
        prefactor2_guess[:,  i, p] = val_guess2[-8:, i, p] / scaling
        print (p,i,'SPADE',np.round(prefactor2_guess[:,  i, p]/prefactor2_actual[:, i, p],3))        
        
line_width = 2.5
font_size = 14
fig_size = (6,4)

# prefactor_direct
fig, ax = plt.subplots(figsize=fig_size)
ax.semilogy(range(P), prefactor1_actual[-1,0,:], linestyle='-', color='red', linewidth=line_width, label='Actual -1')
ax.semilogy(range(P), prefactor1_guess[-1,0,:], linestyle='--', color='red', linewidth=line_width, label='Guess -1')
ax.semilogy(range(P), prefactor1_actual[-2,0,:], linestyle='-', color='blue', linewidth=line_width, label='Actual -2')
ax.semilogy(range(P), prefactor1_guess[-2,0,:], linestyle='--', color='blue', linewidth=line_width, label='Guess -2')
ax.semilogy(range(P), prefactor1_actual[-3,0,:], linestyle='-', color='orange', linewidth=line_width, label='Actual -3')
ax.semilogy(range(P), prefactor1_guess[-3,0,:], linestyle='--', color='orange', linewidth=line_width, label='Guess -3')
ax.semilogy(range(P), prefactor1_actual[-4,0,:], linestyle='-', color='green', linewidth=line_width, label='Actual -4')
ax.semilogy(range(P), prefactor1_guess[-4,0,:], linestyle='--', color='green', linewidth=line_width, label='Guess -4')
ax.set_title('prefactor_direct', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.legend(fontsize=font_size-2, bbox_to_anchor=(1.05, 0.5), loc='center left')
#fig.savefig('prefactor_direct.png', dpi=300, bbox_inches='tight')
plt.show()


# prefactor_SPADE
fig, ax = plt.subplots(figsize=fig_size)
ax.semilogy(range(P), prefactor2_actual[-1,0,:], linestyle='-', color='red', linewidth=line_width, label='Actual -1')
ax.semilogy(range(P), prefactor2_guess[-1,0,:], linestyle='--', color='red', linewidth=line_width, label='Guess -1')
ax.semilogy(range(P), prefactor2_actual[-2,0,:], linestyle='-', color='blue', linewidth=line_width, label='Actual -2')
ax.semilogy(range(P), prefactor2_guess[-2,0,:], linestyle='--', color='blue', linewidth=line_width, label='Guess -2')
ax.semilogy(range(P), prefactor2_actual[-3,0,:], linestyle='-', color='orange', linewidth=line_width, label='Actual -3')
ax.semilogy(range(P), prefactor2_guess[-3,0,:], linestyle='--', color='orange', linewidth=line_width, label='Guess -3')
ax.semilogy(range(P), prefactor2_actual[-4,0,:], linestyle='-', color='green', linewidth=line_width, label='Actual -4')
ax.semilogy(range(P), prefactor2_guess[-4,0,:], linestyle='--', color='green', linewidth=line_width, label='Guess -4')
ax.set_title('prefactor_SPADE', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.legend(fontsize=font_size-2, bbox_to_anchor=(1.05, 0.5), loc='center left')
#fig.savefig('prefactor_SPADE.png', dpi=300, bbox_inches='tight')
plt.show()


# g_val
fig, ax = plt.subplots(figsize=fig_size)
ax.semilogy(range(P), val_g[-1,:], linestyle='-', color='red', linewidth=line_width, label='g -1')
ax.semilogy(range(P), val_g[-2,:], linestyle='-', color='blue', linewidth=line_width, label='g -2')
ax.semilogy(range(P), val_g[-3,:], linestyle='-', color='orange', linewidth=line_width, label='g -3')
ax.semilogy(range(P), val_g[-4,:], linestyle='-', color='green', linewidth=line_width, label='g -4')
ax.set_title('g_val', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.legend(fontsize=font_size-2, bbox_to_anchor=(1.05, 0.5), loc='center left')
#fig.savefig('g_val.png', dpi=300, bbox_inches='tight')
plt.show()


# C_val_direct
fig, ax = plt.subplots(figsize=fig_size)
ax.semilogy(range(P), val_C_array1[-1,0,:], linestyle='-', color='red', linewidth=line_width, label='C -1')
ax.semilogy(range(P), val_C_array1[-2,0,:], linestyle='-', color='blue', linewidth=line_width, label='C -2')
ax.semilogy(range(P), val_C_array1[-3,0,:], linestyle='-', color='orange', linewidth=line_width, label='C -3')
ax.semilogy(range(P), val_C_array1[-4,0,:], linestyle='-', color='green', linewidth=line_width, label='C -4')
ax.set_title('C_val_direct', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.legend(fontsize=font_size-2, bbox_to_anchor=(1.05, 0.5), loc='center left')
#fig.savefig('C_val_direct.png', dpi=300, bbox_inches='tight')
plt.show()


# C_val_SPADE
fig, ax = plt.subplots(figsize=fig_size)
ax.semilogy(range(P), val_C_array2[-1,0,:], linestyle='-', color='red', linewidth=line_width, label='C -1')
ax.semilogy(range(P), val_C_array2[-2,0,:], linestyle='-', color='blue', linewidth=line_width, label='C -2')
ax.semilogy(range(P), val_C_array2[-3,0,:], linestyle='-', color='orange', linewidth=line_width, label='C -3')
ax.semilogy(range(P), val_C_array2[-4,0,:], linestyle='-', color='green', linewidth=line_width, label='C -4')
ax.set_title('C_val_SPADE', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.legend(fontsize=font_size-2, bbox_to_anchor=(1.05, 0.5), loc='center left')
#fig.savefig('C_val_SPADE.png', dpi=300, bbox_inches='tight')
plt.show()
