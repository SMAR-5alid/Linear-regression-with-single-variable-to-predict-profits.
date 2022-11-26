import matplotlib as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read date from file
path = 'D:\Eng_samar\materials\AI\Python\data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])


# Plot the data
data.plot(kind='scatter' , x='Population', y='Profit',figsize=(5,5))


#add new column 
data.insert(0, 'ones', 1)


#Seperate data  
cols=data.shape[1]
X=data.iloc[ : , 0:cols-1]
y=data.iloc[ : , cols-1:cols]


# convert the data into matrices
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))


# cost function 
def computecost (X,y,theta):
    z=np.power(((X * theta.T)-y),2)
    return np.sum(z) / (2* len(X)) 


# Implement cost function
print(" Cost Function : " , computecost(X, y, theta))


# gradient descent
def gradientdescent (X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape)) 
    parameters= int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    
    for i in range (iters):
        error=(X*theta.T) -y
        
        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j] - ((alpha/len(X)) *np.sum(term))
            
        theta=temp
        cost[i]=computecost(X, y, theta)
        
    return theta , cost


# Implement gradient descent function
alpha=0.01
iters=1000
g,cost=gradientdescent(X, y, theta, alpha, iters)
print('g= ', g)
print('cost= ',cost)
print('compute cost = ' , computecost(X, y, g))


# Best fit line
x=np.linspace(data.Population.min(),data.Population.max(),100)
print ("X=  " , x)


# Hypothesis
f=g[0,0] + g[0,1]*x 


# Draw Best fit lin 
fig,ax=plt.subplots()
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Tranning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# Draw entropy graph
fig,ax=plt.subplots()
ax.plot(np.arange(iters) , cost ,'g')
ax.legend(loc=2)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Errors vs. Iterations ')

