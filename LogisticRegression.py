import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, X_train, y_train, weights = None, bias = None):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = weights
        self.bias = bias
    
    def Predict(self, X):
        LinearTerms = np.matmul(X, self.weights) + self.bias
        return self.Sigmoid(LinearTerms)
    
    def Loss(self, weights, bias):
        """Computes the evaluation of the loss function given
        X_train, y_train, weights, and bias."""
        (n,p) = self.X_train.shape
        LinearTerms = np.matmul(self.X_train, weights) + bias
        y_pred = self.Sigmoid(LinearTerms)
        NewLoss = np.log(1 + np.exp(-self.y_train * LinearTerms))
        return sum(NewLoss)/n
        
        
    def Train(self, StepSize = 0.1, epsilon = 0.0001, MaxIter = 10000):
        """ Runs gradient descent with given step size
        on the logistic regression loss function
        with respect to the weight vector and intercept."""
        (n,p) = self.X_train.shape
        weights = np.zeros((p,))
        bias = 0
        IterCount = 0
        error = 1 #initialize difference of loss function
        OldLoss = self.Loss(weights, bias)
        while error >= epsilon and IterCount < MaxIter:
            IterCount += 1
            Gradient = self.ComputeGradientLoss(weights, bias)
            NewWeights = weights - StepSize* Gradient[0].sum(axis = 0)/n
            NewBias = bias - StepSize * sum(Gradient[1])/n
            NewLoss = self.Loss(NewWeights, NewBias)
            error = abs(OldLoss - NewLoss) # Difference of Loss functions due to updating weights.
            weights, bias = NewWeights, NewBias
            OldLoss = NewLoss
        self.weights = weights
        self.bias = bias
        return weights, bias
        
    
    def ComputeGradientLoss(self, weights, bias):
        """Computes the gradient of the logistic regression loss
        function with respect to the weights and bias."""
        LinearTerm = np.matmul(self.X_train, weights)+bias
        ExponentialEvaluation = np.exp(-self.y_train * LinearTerm)
        
        FirstDeriv = self.LossDeriv(ExponentialEvaluation)
        SecondDeriv = self.ExponentialDeriv(LinearTerm)
        
        DerivWeights = self.X_train
        DerivBias = 1
        
        BiasGradient = FirstDeriv * SecondDeriv * DerivBias
        WeightGradient = np.transpose(FirstDeriv * SecondDeriv) * np.transpose(DerivWeights)
        return np.transpose(WeightGradient), BiasGradient    
    
    def LossDeriv(self, s):
        return 1/(1+s)
    
    def ExponentialDeriv(self, t):
        """Assumes input is a numpy array of the same shape as y_train"""
        try:
            return -y_train*np.exp(-y_train*t)
        except:
            raise ValueError("""y_train and input of ExponentialDeriv function
                             not of the same shape.""")
    
    def Sigmoid(self, t):
        """Sigmoid function. Assumes t is a numpy array."""
        return 1/(1+np.exp(-t))
    
    
#--- Test logistic regression on Iris Data ---
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

test_weights = np.zeros((4,))
test_bias = 0

LR = LogisticRegression(X_train, y_train)
LR.Train(epsilon = 1e-5)
print("Accuracy of logistic regression on test data is",
      sum(np.sign(LR.Predict(X_test)-1/2) == y_test)/len(y_test)*100,"%")
