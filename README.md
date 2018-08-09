# My_first_project
K Mean clustering 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1) #generating random numbers on the basis of time and date
#print(np.random.seed(1))

# define the parameters of Gaussian
mu = 0.0
sigma = 1.0

numDims = 1
numSamples = 2000

X = np.random.normal(mu,sigma,[numSamples,numDims])
print(X)

Mean   = np.mean(X)
StdDev = np.std(X)
Median = np.median(X)

print("empirical mean    = ", Mean)
print("empirical std dev = ", StdDev)
print("empirical median = ", Median)
#how to save the csv files
my_dataframe = pd.DataFrame(X)
my_dataframe.to_csv("Mydata.csv")
np.random.seed(1)
numDims=2

mu = np.tile(0.0,numDims) # used numDims insted of 2 bcoz always used general code if numdims change mu change
#cov = np.identity(2)
cov = [[1,0.6],[0.6,1]]
print(mu)
print(cov)
# generate samples from 2D normal distribution
X = np.random.multivariate_normal(mu,cov,numSamples) #we dont use list here bcoz mu and cov are 2D so python undestnd it is 2D matrix
print("matrix size =" , np.shape(X))
# visualize the data
fig, ax = plt.subplots()
ax.plot(X[:,0],X[:,1],'.')
ax.set_xlabel("Dim1")
ax.set_ylabel("Dim2")
ax.set_title('two dimensions plot')
ax.axis('equal')
# compute the mean and standard deviation of the generated samples
mean = X.mean(0) #mean (0) of all coloum
#mean_row = X.mean(1) #mean (0) of all rows

cov=np.cov(X.T) # X.T is transpose of X
print(mean)
print(cov.shape)
from sklearn.cluster import KMeans
X= np.array([[1,2],[2,5],[3,4],[3,5],[6,9],[10,8],[2.7,9],[10,7],[6,6],[4.5,5]])
plt.plot(X[:,0],X[:,1],'*r')
plt.show()
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels    = kmeans.labels_

print(centroids)
print (labels)
colors = ["g.","y."]

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1],colors[labels[i]], markersize = 20)
    plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels    = kmeans.labels_

print(centroids)
print (labels)

colors = ["g.","y.",".r"]

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 20)
    plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()
