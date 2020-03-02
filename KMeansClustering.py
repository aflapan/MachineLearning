import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

class KmeansCluster(object):
    def __init__(self, X, k = None, ClusterLabels = None):
        self.X = X
        self.ClusterLabels = ClusterLabels
        self.k = k
    
    def Train(self, epsilon = 1e-10, maxIter = 10000):
        """Trains k-Means Clustering via iterative algorithm
        :Params: k positive integer. Number of clusters"""
        (n, p) = self.X.shape
        RandomIndex = np.random.choice(range(0, n-1), self.k)
        InitialClusterMeans = self.X[RandomIndex, ]
        InitialClusterLabels = self.GetClusterLabels(InitialClusterMeans)
        error = 1 #initialize loss function error
        numIter = 1 #initialize Iteration number
        while error > epsilon and numIter < maxIter:
            Loss = self.LossFunction(InitialClusterMeans)
            Means = self.ComputeClusterMeans()
            Labels = self.GetClusterLabels(Means)
            UpdatedLoss = self.LossFunction(Means)
            error = abs(Loss - UpdatedLoss)
            numIter += 1
        self.ClusterLabels = Labels
        return Labels
            
            
    
    def GetClusterLabels(self, ClusterMeans):
        """Computes Cluster Labels based on closest distance
        to ClusterMeans."""
        (n, p) = self.X.shape
        self.ClusterLabels = np.zeros((n,))
        for i in range(n):
            sample = self.X[i,]
            Distances = np.sum(ClusterMeans*(ClusterMeans - 2*sample), axis = 1)
            self.ClusterLabels[i] = np.argmin(Distances)
        return self.ClusterLabels
    
    def ComputeClusterMeans(self):
        (n, p) = self.X.shape
        ClusterMeans = np.zeros(shape = (self.k,p))
        for C in range(self.k):
            Cluster = [self.X[i,] for i in range(n) if self.ClusterLabels[i] == C]
            ClusterMeans[C,] = np.mean(Cluster, axis = 0)
        return ClusterMeans
    
    def LossFunction(self, ClusterMeans):
        """Computes the K means cluster loss function
        for the provided cluster means and cluster labels."""
        (n, p) = self.X.shape
        Loss = np.zeros((self.k,))
        for C in range(self.k):
            Cluster = [self.X[i,] for i in range(n) if self.ClusterLabels[i] == C]
            #compute sum-of-squares on Class C
            Loss[C] = np.sum(Cluster - ClusterMeans[C, ])**2 
        return np.sum(Loss)
            
    
    

                    
#--- Test batch perceptron on Iris Data ---
Iris = datasets.load_iris()
X = Iris.data
y = Iris.target


plt.scatter(X[:, 0], X[:, 1], c =y)
plt.show()

IrisKMeans = KmeansCluster(X, k = 3)
print(IrisKMeans.Train())

plt.scatter(X[:, 0], X[:, 1], c = IrisKMeans.ClusterLabels)
plt.show()
