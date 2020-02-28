import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, X_train, y_train, gamma, vector = None, bias = None):
        self.X_train = X_train
        self.y_train = y_train
        self.gamma = gamma
        self.vector = vector
        self.bias = bias
        
    def Train(self, max_iter = 10000):
        """Trains soft SVM using stochastic gradient descent"""
        (n,p) = self.X_train.shape
        vector = np.zeros((p,)) #Initialize support vector and bias
        mean_vector = np.zeros((p,))
        bias = 0
        mean_bias = 0
        for i in range(1, max_iter + 1):
            sample_id = np.random.choice(range(0,n-1), 1)
            x_sample = np.reshape(self.X_train[sample_id, ], (p,))
            y_sample = y_train[sample_id]
            hinge_loss_term = y_sample * (np.dot(x_sample, vector) + bias)
            #Compute derivatives depending on hinge_loss_term
            if  hinge_loss_term < 1:
                deriv_bias = -y_sample
                deriv_vector = self.gamma*vector - y_sample * x_sample
            else:
                deriv_bias = 0
                deriv_vector = self.gamma*vector
            # do gradient descent updates
            vector = vector - deriv_vector/n
            mean_vector = ((i-1)*mean_vector + vector)/i
            bias = bias - deriv_bias/n
            mean_bias = ((i-1) * mean_bias + bias)/i
        self.vector, self.bias = mean_vector, mean_bias
        return mean_vector, mean_bias
    
    def Predict(self, X):
        return np.sign(np.matmul(X, self.vector) + self.bias)
            
        
    
    
    
#--- Test SVM on Iris Data ---
Iris = datasets.load_iris()
X = Iris.data
y = Iris.target


New_y = [] # going to make this a two-class linearly-separable data
for e in y:
    if e == 0:
        New_y.append(-1)
    else:
        New_y.append(1)
        
X_train, X_test, y_train, y_test = train_test_split(X, New_y, test_size = 0.2,
                                                    random_state = 12345)

n, p = X_train.shape
y_train, y_test = np.array(y_train), np.array(y_test)

test_SVM = SVM(X_train = X_train, y_train = y_train, gamma = 1)
test_SVM.Train()
y_pred = test_SVM.Predict(X_test)
print("Accuracy of Support Vector Machines on test data is",
      sum(y_pred == y_test)/len(y_test)*100,"%")