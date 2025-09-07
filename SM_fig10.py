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
    return result/2#/M

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
    return result/2/M

def C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma):
    u_array=np.linspace(-L_measure/2,L_measure/2,N_measure)
    result=np.zeros([N_measure,n,M])
    for i in range(n):
        for k in range(M):
            for p in range(i+1):
                q=i-p
                result[:,i,k]+=psi_fun(u_array,y0[k],sigma,p)*np.conjugate(psi_fun(u_array,y0[k],sigma,q))/math.factorial(p)/math.factorial(q)*sigma**i
    return result*L_measure/N_measure  #normalization such that np.sum(result[:,0,0])=1

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

def GD(g,d,C,alpha,n,M,C_extra):

    D=np.zeros([n*2*M+M,n*2*M+M])
    G=np.zeros([n*2*M+M,n*2*M+M])
    for m in range(n):
        for k in range(M):
            #print (np.shape(D),np.shape(C[:,:,m,k].flatten()),np.shape(np.diag(C[:,:,m,k].flatten())))
            #D+=alpha[k]**m * d[m,k]*np.diag(C[:,:,m,k].flatten())
            D+=alpha[k]**m * d[m,k]*np.diag(np.append(C[:,:,m,k].flatten(),C_extra[:,m,k]))
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
                    G+=alpha[k1]**m1 * alpha[k2]**m2 * g[index1,index2] * left@right
    return G,D



def GD_directimaging(g,d,C,alpha,n,M,N_measure):
    D=np.zeros([N_measure,N_measure])
    G=np.zeros([N_measure,N_measure])
    for m in range(n):
        for k in range(M):
            #print (alpha)
            #print (d[m,k])
            #print (C[:,m,k])
            D+=alpha[k]**m * d[m,k] * np.diag(C[:,m,k])
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    m2=m-m1
                    left=np.array([C[:,m1,k1]]).T
                    right=np.array([C[:,m2,k2]])
                    index1=m1*M+k1
                    index2=m2*M+k2
                    G+=alpha[k1]**m1 * alpha[k2]**m2 * g[index1,index2] * left@right
    return G,D

def Dhalfinv_directimaging_fun(g,d,C,alpha,n,M,N_measure):
    result=np.zeros(N_measure)
    for m in range(n):
        for k in range(M):
            #print (np.shape(D),np.shape(C[:,:,m,k].flatten()),np.shape(np.diag(C[:,:,m,k].flatten())))
            result+=alpha[k]**m * d[m,k]*C[:,m,k]
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
            result+=alpha[k]**m * d[m,k]*(np.append(C[:,:,m,k].flatten(),C_extra[:,m,k]))
    for i in range(n*2*M):
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



T0=time.time()
L =20 
Lmax=L*4  
sigma = 1 
N = 20000 
M =2 
nmax = 150 
n=20 
W=50 
N_measure=50  
L_measure=Lmax  
Q=20
alpha=[1e-3,1e-2]
B=M*3  
S=np.logspace(2,11,Q)

y0_temp = np.linspace(-L/2, L/2, M + 1)
y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax)
vectors = []
for j in range(nmax + 1):
    for i in range(M):
        vectors.append(psi_array[:, i, j])
#g,d=random_gd(n,M)
d,g,x=gd_generation_fast(W, M*20, M, n, L)


#C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
l1=np.zeros([N_measure,Q])
r=np.zeros([N_measure,N_measure,Q])
M_array=np.zeros([N_measure,N_measure,Q])
D_halfinv_array=np.zeros([N_measure,N_measure,Q])
CT_array1=np.zeros(Q)

G,D=GD_directimaging(g,d,C1,alpha,n,M,N_measure)
Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha,n,M,N_measure)
D_halfinv_temp,M_temp,temp1,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
for i in range(Q):
    #CT_temp=C_T(temp1[-B:],alpha,S[i])
    D_halfinv_temp,M_temp,temp1,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
    CT_array1[i]=CT_temp
T1=time.time()
print ('direct imaging',T1-T0)   

product = inner_product_directSPADE(vectors,nmax,M)
a2= a_coeff_fun(product,M,nmax,sigma)
C2=C_SPADE_coeff(n, M, a2)
C2_extra=C_SPADE_coeff_extra(n, M, a2)
l2=np.zeros([2*n*M,Q])
r=np.zeros([2*n*M,2*n*M,Q])
M_array=np.zeros([2*n*M,2*n*M,Q])
D_halfinv_array=np.zeros([2*n*M,2*n*M,Q])
CT_array2=np.zeros(Q)

G,D=GD(g,d,C2,alpha,n,M,C2_extra)
Dhalfinv=Dhalfinv_fun(g,d,C2,alpha,n,M,C2_extra)
D_halfinv_temp,M_temp,temp2,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
for i in range(Q):
    #CT_temp=C_T(temp2[-B:],alpha,S[i])
    D_halfinv_temp,M_temp,temp2,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
    CT_array2[i]=CT_temp
T2=time.time()
print ('SPADE',T2-T1)     
    
    
product = inner_product(vectors)
a3= a_coeff_fun(product,M,nmax,sigma)
C3=C_coeff(n,M,a3)
C3_extra=C_new_coeff_extra(n, M, a3)
l3=np.zeros([2*n*M,Q])
r=np.zeros([2*n*M,2*n*M,Q])
M_array=np.zeros([2*n*M,2*n*M,Q])
D_halfinv_array=np.zeros([2*n*M,2*n*M,Q])
CT_array3=np.zeros(Q)

G,D=GD(g,d,C3,alpha,n,M,C2_extra)
Dhalfinv=Dhalfinv_fun(g,d,C3,alpha,n,M,C2_extra)
D_halfinv_temp,M_temp,temp3,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
for i in range(Q):
    #CT_temp=C_T(temp3[-B:],alpha,S[i])
    D_halfinv_temp,M_temp,temp3,v_temp,CT_temp=main(G,D,Dhalfinv,S[i])
    CT_array3[i]=CT_temp
T3=time.time()
print ('new method',T3-T2)  


plt.figure()
ax=plt.subplot(1,1,1)


#xmajorLocator   = MultipleLocator(2) #将x主刻度标签设置为20的倍数
ymajorLocator   = MultipleLocator(1) #将y主刻度标签设置为20的倍数
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
#plt.xlim([5,D_array[-1]+10])
#plt.ylim([0,0.55])
#plt.xlim([1,1e9])
#plt.ylim([1e0,1e15])

plt.semilogx(S,CT_array1,linewidth=2,linestyle='-',label='direct')
plt.semilogx(S,CT_array2,linewidth=2,linestyle='dotted',label='SPADE')
plt.semilogx(S,CT_array3,linewidth=2,linestyle=(0,(1,6)),label='new')

plt.legend(fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
#plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  


foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('C_T_L'+str(L)+'alpha'+str(alpha[0])+'.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')
