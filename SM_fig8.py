
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig,eigh,inv
import matplotlib.pyplot as plt
import random
import time

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
    G,D=GD_fast(N, M, L, sigma)
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
S=np.logspace(2,10,n)

L=20
sigma=1
val_array1=np.zeros([Mv])
vec_array1=np.zeros([Mv,Mv])
CT_array1=np.zeros(n)

t0=time.time()
val_array1,vec_array1,CT_array1=spectrum_array(N,M,L,sigma,S)
print (time.time()-t0)

L=40
sigma=1
val_array2=np.zeros([Mv])
vec_array2=np.zeros([Mv,Mv])
CT_array2=np.zeros(n)

t0=time.time()
val_array2,vec_array2,CT_array2=spectrum_array(N,M,L,sigma,S)
print (time.time()-t0)

L=60
sigma=1
val_array3=np.zeros([Mv])
vec_array3=np.zeros([Mv,Mv])
CT_array3=np.zeros(n)

t0=time.time()
val_array3,vec_array3,CT_array3=spectrum_array(N,M,L,sigma,S)
print (time.time()-t0)


ax=plt.subplot(1,1,1)


#xmajorLocator   = MultipleLocator(2) #将x主刻度标签设置为20的倍数
ymajorLocator   = MultipleLocator(20) #将y主刻度标签设置为20的倍数
#ymajorLocator = LogLocator(base=10)  # LogLocator for 10^4n
#xmajorFormatter = FormatStrFormatter('%1.1f') #设置x轴标签文本的格式
#xminorLocator   = MultipleLocator(1) #将x轴次刻度标签设置为5的倍数
#ymajorLocator   = MultipleLocator(0.5)
#ax.xaxis.set_major_locator(xmajorLocator)
#ax.xaxis.set_major_formatter(xmajorFormatter)
ax.yaxis.set_major_locator(ymajorLocator)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#plt.xlim([0.5,S[-1]])
#plt.ylim([0,0.55])
#plt.xlim([0,D_array[-1]+1])
#plt.ylim([0,100])
label='linear'
if label=='log':
    plt.semilogx(S,CT_array1,linewidth=2,linestyle='-',label='L=20')
    plt.semilogx(S,CT_array2,linewidth=2,linestyle='dotted',label='L=40')
    plt.semilogx(S,CT_array3,linewidth=2,linestyle='-.',label='L=60')
else:
    plt.plot(S/1e10,CT_array1,linewidth=2,linestyle='-',label='L=20')
    plt.plot(S/1e10,CT_array2,linewidth=2,linestyle='dotted',label='L=40')
    plt.plot(S/1e10,CT_array3,linewidth=2,linestyle='-.',label='L=60')    
plt.legend(fontsize=16,loc='lower right')


foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('3_general_source_S.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')


