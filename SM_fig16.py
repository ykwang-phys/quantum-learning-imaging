
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.special import hermite
import math
from scipy.linalg import eig,eigh,inv
import time
from matplotlib.ticker import MultipleLocator
from scipy import integrate
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
    
    # 使用 numpy.linalg.eigh 进行特征值分解
    l, v =eigh(M)
    
    # 获取排序后的索引（基于特征值的实部）
    sort_indices = np.argsort(np.abs(np.real(l)))
    
    # 使用这些索引来排序特征值和特征向量
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


#2024.11.08
#step_CT v3
W=50
Mmax=100

L =1
Lmax=10
sigma = 1
N = 20000
M = 1
nmax = 120
n=16
S=1000
N_measure=50
L_measure=Lmax
Q=40
alpha=1e-1
S=np.logspace(2,14,Q)

y0_temp = np.linspace(-L/2, L/2, M + 1)
y0 = (y0_temp[:-1] + y0_temp[1:]) / 2

psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax)
vectors = []
for j in range(nmax + 1):
    for i in range(M):
        vectors.append(psi_array[:, i, j])
#g,d=random_gd(n,M)
d,g,x=gd_generation_fast(W, Mmax, M, n, L)
print (g[0:4,0:4])
#d,g,x=gd_generation_fast_edge(W, M*20, M, n, L)
#d,g,x=gd_generation_analytical(W,M,n)
#d,g,x=gd_generation_point(W, M, n)
#d,g,x=gd_generation_linear(W, M*24, M, n, L)
#d[1:-1,:]*=10
#g*=100
#d,g,x=gd_generation_compact(W, M*8, M, n, L,alpha)  
#d,g,x=gd_generation_beta(W, M*60, M, n)
l,v=eigh(g)
print (l)
#C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
l1=np.zeros([N_measure,Q])
r=np.zeros([N_measure,N_measure,Q])
M_array=np.zeros([N_measure,N_measure,Q])
D_halfinv_array=np.zeros([N_measure,N_measure,Q])
CT_array1=np.zeros(Q)

G,D=GD_directimaging(g,d,C1,alpha,n,M,N_measure)
Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha,n,M,N_measure)

for i in range(Q):
    D_halfinv_temp,M_temp,temp1,v_temp,CT_temp1=main(G,D,Dhalfinv,S[i])
    CT_temp=C_T(temp1[-6:],alpha,S[i])
    #print (i,CT_temp1,CT_temp)
    CT_array1[i]=CT_temp1
print ('direct imaging')   


CT_array2=np.zeros(Q)

product2 = inner_product_directSPADE(vectors,nmax,M)
a2= a_coeff_fun(product2,M,nmax,sigma)
C2=C_SPADE_coeff(n, M, a2)
C2_extra=C_SPADE_coeff_extra(n, M, a2)
l2=np.zeros([2*n*M+M,Q])
r=np.zeros([2*n*M+M,2*n*M+M,Q])
M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])

G,D=GD(g,d,C2,alpha,n,M,C2_extra)
Dhalfinv=Dhalfinv_fun(g,d,C2,alpha,n,M,C2_extra)

    
    

for i in range(Q):
    D_halfinv_temp,M_temp,temp2,v_temp,CT_temp2=main(G,D,Dhalfinv,S[i])
    CT_temp=C_T(temp2[-6:],alpha,S[i])
    #print (i,CT_temp,CT_temp2)
    CT_array2[i]=CT_temp2
print ('SPADE')     
    


plt.figure()
ax=plt.subplot(1,1,1)


#xmajorLocator   = MultipleLocator(2) #将x主刻度标签设置为20的倍数
#ymajorLocator   = MultipleLocator(1000000) #将y主刻度标签设置为20的倍数
#ymajorLocator = LogLocator(base=10)  # LogLocator for 10^4n
#xmajorFormatter = FormatStrFormatter('%1.1f') #设置x轴标签文本的格式
#xminorLocator   = MultipleLocator(1) #将x轴次刻度标签设置为5的倍数
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

plt.semilogx(S,CT_array1,linewidth=2,linestyle='-',label='direct')
plt.semilogx(S,CT_array2,linewidth=2,linestyle='dotted',label='SPADE')


plt.legend(fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
#plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  


foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('gd_C_T_Mmax'+str(Mmax)+'W'+str(W)+'alpha'+str(alpha)+'.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')



alpha=1e-2


#C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
#C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
l1=np.zeros([N_measure,Q])
r=np.zeros([N_measure,N_measure,Q])
M_array=np.zeros([N_measure,N_measure,Q])
D_halfinv_array=np.zeros([N_measure,N_measure,Q])
CT_array1=np.zeros(Q)

G,D=GD_directimaging(g,d,C1,alpha,n,M,N_measure)
Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha,n,M,N_measure)

for i in range(Q):
    D_halfinv_temp,M_temp,temp1,v_temp,CT_temp1=main(G,D,Dhalfinv,S[i])
    CT_temp=C_T(temp1[-6:],alpha,S[i])
    #print (i,CT_temp1,CT_temp)
    CT_array1[i]=CT_temp1
print ('direct imaging')   


CT_array2=np.zeros(Q)

product2 = inner_product_directSPADE(vectors,nmax,M)
a2= a_coeff_fun(product2,M,nmax,sigma)
C2=C_SPADE_coeff(n, M, a2)
C2_extra=C_SPADE_coeff_extra(n, M, a2)
l2=np.zeros([2*n*M+M,Q])
r=np.zeros([2*n*M+M,2*n*M+M,Q])
M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])

G,D=GD(g,d,C2,alpha,n,M,C2_extra)
Dhalfinv=Dhalfinv_fun(g,d,C2,alpha,n,M,C2_extra)

    
    

for i in range(Q):
    D_halfinv_temp,M_temp,temp2,v_temp,CT_temp2=main(G,D,Dhalfinv,S[i])
    CT_temp=C_T(temp2[-6:],alpha,S[i])
    #print (i,CT_temp,CT_temp2)
    CT_array2[i]=CT_temp2
print ('SPADE')     
    


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
#plt.xlim([0,D_array[-1]+1])
#plt.ylim([1e0,1e15])

plt.semilogx(S,CT_array1,linewidth=2,linestyle='-',label='direct')
plt.semilogx(S,CT_array2,linewidth=2,linestyle='dotted',label='SPADE')


plt.legend(fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
#plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  


foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('gd_C_T_Mmax'+str(Mmax)+'W'+str(W)+'alpha'+str(alpha)+'.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')



alpha=1e-3


#C1=C_directimaging_coeff(n,M,N_measure,L_measure,y0,sigma)
#C1=C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma)
l1=np.zeros([N_measure,Q])
r=np.zeros([N_measure,N_measure,Q])
M_array=np.zeros([N_measure,N_measure,Q])
D_halfinv_array=np.zeros([N_measure,N_measure,Q])
CT_array1=np.zeros(Q)

G,D=GD_directimaging(g,d,C1,alpha,n,M,N_measure)
Dhalfinv=Dhalfinv_directimaging_fun(g,d,C1,alpha,n,M,N_measure)

for i in range(Q):
    D_halfinv_temp,M_temp,temp1,v_temp,CT_temp1=main(G,D,Dhalfinv,S[i])
    CT_temp=C_T(temp1[-6:],alpha,S[i])
    #print (i,CT_temp1,CT_temp)
    CT_array1[i]=CT_temp1
print ('direct imaging')   


CT_array2=np.zeros(Q)

product2 = inner_product_directSPADE(vectors,nmax,M)
a2= a_coeff_fun(product2,M,nmax,sigma)
C2=C_SPADE_coeff(n, M, a2)
C2_extra=C_SPADE_coeff_extra(n, M, a2)
l2=np.zeros([2*n*M+M,Q])
r=np.zeros([2*n*M+M,2*n*M+M,Q])
M_array=np.zeros([2*n*M+M,2*n*M+M,Q])
D_halfinv_array=np.zeros([2*n*M+M,2*n*M+M,Q])

G,D=GD(g,d,C2,alpha,n,M,C2_extra)
Dhalfinv=Dhalfinv_fun(g,d,C2,alpha,n,M,C2_extra)

    
    

for i in range(Q):
    D_halfinv_temp,M_temp,temp2,v_temp,CT_temp2=main(G,D,Dhalfinv,S[i])
    CT_temp=C_T(temp2[-6:],alpha,S[i])
    #print (i,CT_temp,CT_temp2)
    CT_array2[i]=CT_temp2
print ('SPADE')     
    



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
#plt.xlim([0,D_array[-1]+1])
#plt.ylim([1e0,1e15])

plt.semilogx(S,CT_array1,linewidth=2,linestyle='-',label='direct')
plt.semilogx(S,CT_array2,linewidth=2,linestyle='dotted',label='SPADE')


plt.legend(fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
#plt.legend(fontsize=17,loc='center left', bbox_to_anchor=(1, 0.5))  


foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('gd_C_T_Mmax'+str(Mmax)+'W'+str(W)+'alpha'+str(alpha)+'.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')


