import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

class BatchPerceptron:
    def __init__(self, X_train, y_train, weights = None, bias = None):
        """Initializes the batch perceptron algorithm class.
        Assumes that the there are only two classes, with corresponding
        labels +1 and -1."""
        self.X_train = X_train
        self.y_train = y_train
        self.weights = weights
        self.bias = bias
        
    def Train(self, MaxIter = 10000):
        """Trains the batch perceptron algorithm. Guaranteed 
        to converge for the linearly seperable case. Returns
        discriminant vector w for seperable case. If maximum number
        of iteration reached, returns vector w and warning stating that 
        data may not be seperable."""
        (n, p) = self.X_train.shape
        X_train = np.append(self.X_train, np.ones((n,1)), axis = 1)
        y_train = self.y_train
        if len(np.unique(y_train)) > 2:
            return """Warning: more than two classes represented 
                in training labels. Please make sure there are only two
                    classes of +1 and -1 labels."""
        w = np.zeros(p+1)
        IterCount = 0 
        Sign = True # a flag for if any of the training data are labelled wrongly.
                     # flips to false once everything is perfectly labelled. 
        while Sign and IterCount <= MaxIter:
            for i in range(n):
                if np.sign(np.matmul(X_train[i,:] , np.transpose(w))) != y_train[i]:
                    w = w + y_train[i] * X_train[i,:]
                    Sign = any(np.sign(np.matmul(X_train, w)) != y_train) # set sign flag
                    IterCount += 1
        if IterCount >= MaxIter:
             print("""Warning: maximum number of iterations reached, 
                  data may not be linearly seperable.""")
        self.bias = w[-1]
        self.weights = w[:p]
        return self.weights, self.bias
    
    def Predict(self, X):
        bias = self.bias
        weights = self.weights
        return np.sign(np.matmul(X, np.transpose(weights)) + bias)
    
        

                    
#--- Test batch perceptron on Iris Data ---
Iris = datasets.load_iris()
X = Iris.data
y = Iris.target

New_y = []
for e in y:
    if e == 0:
        New_y.append(-1)
    else:
        New_y.append(1)
        
X_train, X_test, y_train, y_test = train_test_split(X, New_y, test_size = 0.2,
                                                    random_state = 12345)


plt.scatter(X_train[:, 0], X_train[:, 1], c =y_train)
plt.show()

BP = BatchPerceptron(X_train, y_train)
BP.Train(1000)
y_test_pred = BP.Predict(X_test)
print("The accuracy of our Batch Perceptron on the test data is ", 
      sum(y_test_pred == y_test)/len(y_test)*100,'%')