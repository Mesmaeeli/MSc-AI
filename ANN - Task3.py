
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class NeuralNetwrok:
    def __init__(self,):
        
        self.activation_functions = {'tanh':     {'func' :(lambda x: np.tanh(x)),
                                                  'deriv':(lambda x: 1-np.tanh(x)**2)},
                                     'relu':     {'func' :(lambda x: x*(x > 0)),
                                                  'deriv':(lambda x: 1 * (x>0))},
                                     'sigmoid':  {'func' :(lambda x: 1/(1+np.exp(-x))),
                                                  'deriv':(lambda x: x * (1 - x))}}
        
        self.loss_functions       = {'rmse':     {'func' :(lambda x,y: np.sqrt(((x - y) ** 2).mean())),
                                                  'deriv':(lambda x,y: y-x)},
                                     'cross_ent':{'func' :(lambda x: x*(x > 0)),
                                                  'deriv':(lambda x: 1 * (x>0))},
                                     'sigmoid':  {'func' :(lambda x: 1/(1+np.exp(-x))),
                                                  'deriv':(lambda x: x * (1 - x))}}

        self.deltas   = []
        self.neurons  = [] 
        self.arch     = []
        self.weights  = []
        self.biases   = []
        self.actives  = []
        self.n_layers = -1
        self.input    = None
        self.target   = None
        self.history  = []
        self.acc      = []
        self.output   = []
        
    def add_layer(self, input_node, activation_func):
        
        self.arch.append(input_node)
        
        if len(self.arch) > 1:
            self.weights.append(np.random.rand(self.arch[-2], self.arch[-1]) - 0.5)
            
        self.actives.append(activation_func)
        
        self.n_layers += 1
            
    def forward_propagation(self):
        
        func = self.activation_functions[self.actives[0]]['func']
        result = np.dot(self.input, self.weights[0])
        self.neurons.append(func(result))
        
        for i in range(1,self.n_layers):
            func = self.activation_functions[self.actives[i]]['func']
            result = np.dot(self.neurons[i-1], self.weights[i])
            self.neurons.append(func(result))
            
            
    def back_propagation(self):
        
        loss_func  = self.loss_functions[self.loss]['func']
        loss_deriv = self.loss_functions[self.loss]['deriv']
        
        func  = self.activation_functions[self.actives[-1]]['func']
        deriv = self.activation_functions[self.actives[-1]]['deriv']
        
        loss_value = loss_func(self.neurons[-1], self.target)
        
        self.history.append(loss_value)
        
        self.neurons.insert(0,self.input)
        
        result = loss_deriv(self.target,self.neurons[-1]) * deriv(self.neurons[-1])
        self.deltas.append(result)
        
        for i in range(2,self.n_layers+1):
            result = np.dot(self.deltas[i-2],self.weights[-i+1].T) * deriv(self.neurons[-i])
            self.deltas.append(result)
        
        self.deltas.reverse()
        
        for i in range(self.n_layers):
            self.weights[i] -= self.lr_rate*(np.dot(self.neurons[i].T, self.deltas[i]))
            

        
        
    def fit(self, x_train, y_train, loss_func = 'rmse', epochs = 1000, learning_rate = 0.001, optimizer = None):
        
        self.input  = x_train
        self.target = y_train
        
        self.loss = loss_func
        
        self.lr_rate = learning_rate
        
        for step in range(epochs+1):
            
            self.forward_propagation()
            self.back_propagation()
        

            self.output = self.neurons[-1]
            
            # self.output[np.where(self.output > 0.5)] = 1
            # self.output[np.where(self.output < 0.5)] = 0
            accracy = sum(self.neurons[-1] == self.target) / len(self.target)
            self.acc.append(accracy)
            
            self.deltas.clear()
            self.neurons.clear()
            
            print('epoch: {},    loss: {}   , acc: {}'.format(step, self.history[step], self.acc[step]))
            
    def predict(self,Xtest):
        
        result = Xtest
        for i in self.weights:
            result = np.dot(result, i)
            
        return result
            


# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler

# X, y = make_blobs( n_samples=1000, n_features=5, centers=((1, 1,1), (5, 5,5)), cluster_std = 3)

# X = StandardScaler().fit_transform(X)
# y = np.reshape(y,(-1,1))

# from sklearn.datasets import make_moons
# X, y = make_moons(500, noise=0.10)


# y = np.reshape(y,(-1,1))



from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_train = x_train.astype('float64')
x_train /= 255.0
x_train = np.around(x_train, decimals= 4)
x_train = x_train.squeeze()
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_train = y_train.squeeze()
# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28*28)
x_test = x_test.astype('float64')
x_test /= 255.0
x_test = np.around(x_test, decimals= 4)
y_test = np_utils.to_categorical(y_test)





M = NeuralNetwrok()

M.add_layer(28*28,'sigmoid')
M.add_layer(200,'sigmoid')
M.add_layer(50,'sigmoid')
M.add_layer(10,'sigmoid')

M.fit(x_train[:1000],y_train[:1000])

plt.plot(M.history)