from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy import integrate
def D(x_array,alpha,beta):
    (m,)=np.shape(x_array)
    result=np.zeros([m,m])
    for i in range(m):
        x=x_array[i]
        result[i,i]=np.sqrt(2/np.pi/(alpha**2+4*beta**2))*np.exp(-2*x**2/(alpha**2+4*beta**2))
    return result

def G(x_array,alpha,beta):
    (m,)=np.shape(x_array)
    result=np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            x=x_array[i]
            y=x_array[j]
            #temp1=np.exp(-(x**2+y**2)/2/beta**2+alpha**2*(x**2+y**2)/4/beta**2/(alpha**2+2*beta**2))
            #temp2=temp1/2/np.sqrt(2)/np.pi/alpha/beta**2/np.sqrt(2/alpha**2+1/beta**2)
            #temp3=temp2*2*np.cosh(alpha**2*x*y/2/beta**2/(alpha**2+2*beta**2))
            temp1=np.exp(-(x**2+y**2)/2/beta**2)/2/np.sqrt(2)/np.pi/alpha/beta**2/np.sqrt(2/alpha**2+1/beta**2)
            temp2=np.exp(alpha**2*(x-y)**2/4/beta**2/(alpha**2+2*beta**2))+np.exp(alpha**2*(x+y)**2/4/beta**2/(alpha**2+2*beta**2))
            temp3=temp1*temp2
            result[i,j]=temp3
    return result

def D_y_fun(y,alpha,beta):
    result=np.sqrt(2/np.pi/(alpha**2+4*beta**2))*np.exp(-2*y**2/(alpha**2+4*beta**2))
    return result
def G_y_fun(x,y,alpha,beta):
    #x=u[0]
    #y=u[1]
    temp1=np.exp(-(x**2+y**2)/2/beta**2)/2/np.sqrt(2)/np.pi/alpha/beta**2/np.sqrt(2/alpha**2+1/beta**2)
    temp2=np.exp(alpha**2*(x-y)**2/4/beta**2/(alpha**2+2*beta**2))+np.exp(alpha**2*(x+y)**2/4/beta**2/(alpha**2+2*beta**2))
    result=temp1*temp2
    return result
        
def G_Dinv_integral(x_array,alpha,beta):
    (m,)=np.shape(x_array)
    l=(np.max(x_array)-np.min(x_array))/(m-1)
    result=np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            x=x_array[i]
            y=x_array[j]
            #temp1=np.exp(-(x**2+y**2)/2/beta**2+alpha**2*(x**2+y**2)/4/beta**2/(alpha**2+2*beta**2))
            #temp2=temp1/2/np.sqrt(2)/np.pi/alpha/beta**2/np.sqrt(2/alpha**2+1/beta**2)
            #temp3=temp2*2*np.cosh(alpha**2*x*y/2/beta**2/(alpha**2+2*beta**2))
            #temp1=np.exp(-(x**2+y**2)/2/beta**2)/2/np.sqrt(2)/np.pi/alpha/beta**2/np.sqrt(2/alpha**2+1/beta**2)
            #temp2=np.exp(alpha**2*(x-y)**2/4/beta**2/(alpha**2+2*beta**2))+np.exp(alpha**2*(x+y)**2/4/beta**2/(alpha**2+2*beta**2))
            #temp3=temp1*temp2/np.sqrt(2/np.pi/(alpha**2+4*beta**2))*np.exp(2*y**2/(alpha**2+4*beta**2))
            D_temp,err=integrate.quad(D_y_fun,y-l/2,y+l/2,args=(alpha,beta))
            G_temp,err=integrate.dblquad(G_y_fun,y-l/2,y+l/2,x-l/2,x+l/2,args=(alpha,beta))
            
            result[i,j]=G_temp/D_temp
    return result

def G_Dinv(x_array,alpha,beta):
    (m,)=np.shape(x_array)
    result=np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            x=x_array[i]
            y=x_array[j]
            #temp1=np.exp(-(x**2+y**2)/2/beta**2+alpha**2*(x**2+y**2)/4/beta**2/(alpha**2+2*beta**2))
            #temp2=temp1/2/np.sqrt(2)/np.pi/alpha/beta**2/np.sqrt(2/alpha**2+1/beta**2)
            #temp3=temp2*2*np.cosh(alpha**2*x*y/2/beta**2/(alpha**2+2*beta**2))
            temp1=np.exp(-(x**2+y**2)/2/beta**2)/2/np.sqrt(2)/np.pi/alpha/beta**2/np.sqrt(2/alpha**2+1/beta**2)
            temp2=np.exp(alpha**2*(x-y)**2/4/beta**2/(alpha**2+2*beta**2))+np.exp(alpha**2*(x+y)**2/4/beta**2/(alpha**2+2*beta**2))
            temp3=temp1*temp2/np.sqrt(2/np.pi/(alpha**2+4*beta**2))*np.exp(2*y**2/(alpha**2+4*beta**2))
            result[i,j]=temp3
    return result

def C_T(x_array,alpha,beta,S):
    G_M=G(x_array,alpha,beta)
    D_M=D(x_array,alpha,beta)
    V_M=D_M-G_M
    #print (np.max(V_M/S))
    temp1=G_M+V_M/S
    #print (np.sum(np.abs(V_M/S)))
    temp2=linalg.inv(temp1)@G_M
    #eig, vec=linalg.eig(G_M)
    print (np.trace(linalg.pinv(G_M)@G_M))
    #print (eig)
    #print (linalg.pinv(G_M)@G_M)
    #print (np.max(temp2),np.min(temp2))
    result=np.trace(temp2)
    return result

n=20
M=40
beta=1e0
alpha=np.logspace(-3,-1,n)
S=100
x_array_temp=np.linspace(-5,5,M+1)
x_array = (x_array_temp[:-1] + x_array_temp[1:]) / 2
l_array=np.zeros([n,M])



for i in range(n):
    G_M=G(x_array,alpha[i],beta)
    D_M=D(x_array,alpha[i],beta)
    G_Dinv_M=G_Dinv_integral(x_array,alpha[i],beta)
    V_M=D_M-G_M
    val,vec=linalg.eig(G_Dinv_M)
    l=1/val-1
    l_array[i,:]=np.sort(np.abs(l))
    


threshold=1e15
beta1_fit=np.zeros([2,n])
k1=(l_array[:,1]/alpha**(-4))[-2]
k2=(l_array[:,2]/alpha**(-8))[-2]
for i in range(n):
    if l_array[i,1]>threshold:
        beta1_fit[0,i]=k1*alpha[i]**(-4)
    else:
        beta1_fit[0,i]=l_array[i,1]
        
    if l_array[i,2]>threshold:
        beta1_fit[1,i]=k2*alpha[i]**(-8)
    else:
        beta1_fit[1,i]=l_array[i,2]
    

        
        


ax=plt.subplot(1,1,1)


#xmajorLocator   = MultipleLocator(2) #将x主刻度标签设置为20的倍数
#ymajorLocator   = MultipleLocator(20000) #将y主刻度标签设置为20的倍数
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
#plt.ylim([1e-7,1e15])
#plt.loglog(alpha,l_array[:,0],label='bet0')
plt.loglog(alpha,beta1_fit[0,:],linestyle='-',label='bet1')
plt.loglog(alpha,beta1_fit[1,:],linestyle='--',label='bet2')
#plt.plot(D_array,np.array([1/(M-1)]*n),label='asymptotic',linestyle=':')
plt.legend(fontsize=16,loc='upper right')#, bbox_to_anchor=(0, 0.1))



foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('direct.jpg', format='jpg', dpi=1000,bbox_inches = 'tight')