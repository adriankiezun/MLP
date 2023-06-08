import pandas as pd
import numpy as np
import copy as cp
import warnings
warnings.filterwarnings('ignore')

class MLP:

    def __init__(self, X, y, hidden_layer_sizes, activation="sigmoid",
                  learning_rate=0.01, epochs=100, random_state=None,
                  weights=[], bias=[], activation_output="linear", random_w_min = -1, random_w_max = 1,
                  cv_X = None, cv_y = None, print_evey_n_epoch = 100, momentum = 0,
                    epsilon = 1e-8, beta = 0.9, batch_size = None, optimizer = None,
                    beta1_adam =0.9, beta2_adam = 0.999, cost_function_value = None):
        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = cp.deepcopy(activation)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights = cp.deepcopy(weights)
        self.bias = cp.deepcopy(bias)
        self.output_size = None
        self.activation_output = cp.deepcopy(activation_output)
        self.random_w_min = random_w_min
        self.random_w_max = random_w_max
        self.cv_X = cp.deepcopy(cv_X)
        self.cv_y = cp.deepcopy(cv_y)
        self.random_w_bool = False
        self.declared_weights = cp.deepcopy(weights)
        self.declared_bias = cp.deepcopy(bias)
        self.print_evey_n_epoch = cp.deepcopy(print_evey_n_epoch)
        self.batch_size = cp.deepcopy(batch_size)
        self.momentum = cp.deepcopy(momentum)
        self.epsilon = cp.deepcopy(epsilon)
        self.beta = cp.deepcopy(beta)
        self.optimizer = cp.deepcopy(optimizer)
        self.beta1_adam = cp.deepcopy(beta1_adam)
        self.beta2_adam = cp.deepcopy(beta2_adam)
        self.rmsprop_w = None
        self.rmsprop_b = None
        self.adam_m_w = None
        self.adam_m_b = None
        self.adam_v_w = None
        self.adam_v_b = None
        self.t = 0
        self.prev_delta_w = None
        self.prev_delta_b = None
        self.min_cost_function_train = None
        self.min_cost_function_val = None
        self.cost_function_value = cp.deepcopy(cost_function_value)
        self.best_weights = None
        self.best_bias = None
        

    def activation_function(self, x):
        if self.activation == "sigmoid":
            return 1/(1+np.exp(-x))
        if self.activation == "relu":
            return np.maximum(0,x)
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "linear":
            return x
        
    def activation_function_derivative(self, x):
        if self.activation == "sigmoid":
            return (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
        if self.activation == "relu":
            return np.where(x<=0,0,1)
        if self.activation == "tanh":
            return 1 - np.tanh(x)**2
        if self.activation == "linear":
            return 1
        
    def activation_output_function(self, x):
        if self.activation_output == "sigmoid":
            return 1/(1+np.exp(-x))
        if self.activation_output == "linear":
            return x
        if self.activation_output == "softmax":
            exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def activation_output_function_derivative(self, x):
        if self.activation_output == "sigmoid":
            return (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
        if self.activation_output == "linear":
            return 1
        if self.activation_output == "softmax":
            exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
            softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return softmax_probs * (1 - softmax_probs)
        
    def loss_function(self, y, ypred):
        if self.activation_output == "sigmoid":
            return -np.sum(y*np.log(ypred))
        if self.activation_output == "linear":
            return np.sum(np.square(y - ypred)) / y.shape[0]
        if self.activation_output == "softmax":
            return np.sum(np.array([-np.sum(y*np.log(ypred))])).round(4)
        
    def loss_function_derivative(self, y, ypred):
        if self.activation_output == "sigmoid":
            return y-ypred
        if self.activation_output == "linear":
            return 2*(y-ypred)
        if self.activation_output == "softmax":
            return y-ypred
        
    def loss_function_print(self, y, ypred):
        if self.activation_output == "linear":
            return str(np.array([np.sum(np.square(y - ypred)) / y.shape[0]]))[2:-2]
        if self.activation_output == "sigmoid":
            return str(np.array([-np.sum(y*np.log(ypred))]))[2:-2]
        if self.activation_output == "softmax":
            return np.sum(np.array([-np.sum(y*np.log(ypred))])).round(4)
        
    def OneHot(self, y):
        n = y.iloc[:,0].max() + 1
        return np.eye(n)[y]

    def feedforward(self, X):
        z = []
        a = []
        for i in range(len(self.weights)):
            if i == 0:
                z.append(np.dot(X, self.weights[i]) + self.bias[i])
                a.append(self.activation_function(z[i]))
            elif i == len(self.weights)-1:
                z.append(np.dot(a[i-1], self.weights[i]) + self.bias[i])
                a.append(self.activation_output_function(z[i]))
            else:
                z.append(np.dot(a[i-1], self.weights[i]) + self.bias[i])
                a.append(self.activation_function(z[i]))
        return z, a
    
    def backpropagation(self, z, a, y, x):
        delta_w = [0] * len(self.weights)
        delta_w_final = [0] * len(self.weights)
        delta_b = [0] * len(self.bias)
        for i in range(len(self.weights) - 1, -1, -1):
            if i == (len(self.weights) - 1):
                delta_w[i] = self.loss_function_derivative(y, a[i]) * self.activation_output_function_derivative(z[i])
                delta_w_final[i] = np.dot(a[i - 1].T, delta_w[i])
                delta_b[i] = self.loss_function_derivative(y, a[i]) * self.activation_output_function_derivative(z[i])
            elif i == 0:
                delta_w[i] = np.dot(delta_w[i + 1], self.weights[i + 1].T) * self.activation_function_derivative(z[i])
                delta_w_final[i] = np.dot(x.T, delta_w[i])
                delta_b[i] = np.dot(delta_b[i + 1], self.weights[i + 1].T) * self.activation_function_derivative(z[i])
            else:
                delta_w[i] = np.dot(delta_w[i + 1], self.weights[i + 1].T) * self.activation_function_derivative(z[i])
                delta_w_final[i] = np.dot(a[i - 1].T, delta_w[i])
                delta_b[i] = np.dot(delta_b[i + 1], self.weights[i + 1].T) * self.activation_function_derivative(z[i])
        return delta_w_final, delta_b        

    def fit(self):
        if self.activation_output == "softmax":
            self.output_size = cp.deepcopy(len(np.unique(self.y)))
            self.y = cp.deepcopy(pd.DataFrame(self.OneHot(self.y).reshape(self.y.shape[0], self.output_size)))
            if self.cv_X is not None:
                self.cv_y = cp.deepcopy(pd.DataFrame(self.OneHot(self.cv_y).reshape(self.cv_y.shape[0], self.output_size)))
        else:
            self.output_size = 1
        if len(self.weights) == 0 or len(self.bias) == 0:
            if self.random_state != None:
                np.random.seed(self.random_state)
            else:
                self.random_state = int(np.random.randint(0, 10000000, size=(1, 1)))
                np.random.seed(self.random_state)
            self.random_w_bool = True
            self.weights = [0] * (len(self.hidden_layer_sizes) + 1)
            self.bias = [0] * (len(self.hidden_layer_sizes) + 1)
            for i in range(len(self.hidden_layer_sizes) + 1):
                if i == 0:
                    self.weights[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(self.X.shape[1],self.hidden_layer_sizes[i])).astype(np.float64)
                    self.bias[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(1, self.hidden_layer_sizes[i])).astype(np.float64)
                elif i == len(self.hidden_layer_sizes):
                    self.weights[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(self.hidden_layer_sizes[i - 1], self.output_size)).astype(np.float64)
                    self.bias[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(1, self.output_size)).astype(np.float64)
                else:
                    self.weights[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i])).astype(np.float64)
                    self.bias[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(1, self.hidden_layer_sizes[i])).astype(np.float64)
        elif len(self.weights) != len(self.bias):
            raise ValueError("Number of weights and bias must be equal")
        else:
            pass
        self.prev_delta_w = cp.deepcopy([np.zeros_like(w) for w in self.weights])
        self.prev_delta_b = cp.deepcopy([np.zeros_like(b) for b in self.bias])
        self.rmsprop_w = cp.deepcopy([np.zeros_like(w) for w in self.weights])
        self.rmsprop_b = cp.deepcopy([np.zeros_like(b) for b in self.bias])
        self.adam_m_w = cp.deepcopy([np.zeros_like(w) for w in self.weights])
        self.adam_m_b = cp.deepcopy([np.zeros_like(b) for b in self.bias])
        self.adam_v_w = cp.deepcopy([np.zeros_like(w) for w in self.weights])
        self.adam_v_b = cp.deepcopy([np.zeros_like(b) for b in self.bias])
        self.t = 0
        for i in range(self.epochs):
            delta_w = [np.zeros_like(w) for w in self.weights]
            delta_b = [np.zeros_like(b) for b in self.bias]
            if self.batch_size == None:
                for j in range(0, self.X.shape[0]):
                    z, a = self.feedforward(np.array([self.X.iloc[j, :].values]))
                    delta_w_temp, delta_b_temp = self.backpropagation(z, a, np.array([self.y.iloc[j, :].values]), np.array([self.X.iloc[j, :].values]))
                    for l in range(len(delta_w)):
                        delta_w[l] = delta_w[l] + delta_w_temp[l]
                        delta_b[l] = delta_b[l] + delta_b_temp[l]
                for l in range(len(delta_w)):
                    delta_w[l] /= self.X.shape[0]
                    delta_b[l] /= self.X.shape[0]
                for k in range(len(self.weights)):
                    if self.optimizer is None:
                        self.weights[k] += self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                        self.bias[k] += self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                        self.prev_delta_w[k] = self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                        self.prev_delta_b[k] = self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                    elif self.optimizer == "rmsprop":
                        self.rmsprop_w[k] = self.beta * self.rmsprop_w[k] + (1 - self.beta) * delta_w[k] ** 2
                        self.rmsprop_b[k] = self.beta * self.rmsprop_b[k] + (1 - self.beta) * delta_b[k] ** 2
                        self.weights[k] += self.learning_rate * delta_w[k] / (np.sqrt(self.rmsprop_w[k]) + self.epsilon)
                        self.bias[k] += self.learning_rate * delta_b[k] / (np.sqrt(self.rmsprop_b[k]) + self.epsilon)
                    elif self.optimizer == "adam":
                        self.t += 1
                        self.adam_m_w[k] = self.beta1_adam * self.adam_m_w[k] + (1 - self.beta1_adam) * delta_w[k]
                        self.adam_m_b[k] = self.beta1_adam * self.adam_m_b[k] + (1 - self.beta1_adam) * delta_b[k]
                        self.adam_v_w[k] = self.beta2_adam * self.adam_v_w[k] + (1 - self.beta2_adam) * delta_w[k] ** 2
                        self.adam_v_b[k] = self.beta2_adam * self.adam_v_b[k] + (1 - self.beta2_adam) * delta_b[k] ** 2
                        adam_m_w_hat = self.adam_m_w[k] / (1 - self.beta1_adam ** self.t)
                        adam_m_b_hat = self.adam_m_b[k] / (1 - self.beta1_adam ** self.t)
                        adam_v_w_hat = self.adam_v_w[k] / (1 - self.beta2_adam ** self.t)
                        adam_v_b_hat = self.adam_v_b[k] / (1 - self.beta2_adam ** self.t)
                        self.weights[k] += self.learning_rate * adam_m_w_hat / (np.sqrt(adam_v_w_hat) + self.epsilon)
                        self.bias[k] += self.learning_rate * adam_m_b_hat / (np.sqrt(adam_v_b_hat) + self.epsilon)
                if self.min_cost_function_train is None or self.min_cost_function_train > self.loss_function(self.y, self.predict(self.X)):
                    self.min_cost_function_train = self.loss_function(self.y, self.predict(self.X))
                if self.cv_X is not None and (self.min_cost_function_val is None or self.min_cost_function_val > self.loss_function(self.y, self.predict(self.X))):
                    self.min_cost_function_val = self.loss_function(self.cv_y, self.predict(self.cv_X))
                if self.cost_function_value is not None:
                    if self.cost_function_value > self.loss_function(self.y, self.predict(self.X)):
                        self.cost_function_value = self.loss_function(self.y, self.predict(self.X))
                        self.best_weights = cp.deepcopy(self.weights)
                        self.best_bias = cp.deepcopy(self.bias)
                if (i+1) % self.print_evey_n_epoch == 0 or i == 0:
                    if self.cv_X is None:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} === Min: {self.min_cost_function_train:^14}")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} === Min: {self.min_cost_function_train:^14}")
                    else:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} === Min: {self.min_cost_function_train:^14} === Val MSE: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} === Min: {self.min_cost_function_val:^14} ===")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} === Min: {self.min_cost_function_train:^14} === Val CrossEntropy: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} === Min: {self.min_cost_function_val:^14} ===")
            else:
                if i == 0:
                    self.X = self.X.sample(frac=1, random_state=self.random_state)
                    self.y = self.y.sample(frac=1, random_state=self.random_state)
                else:
                    self.X = self.X.sample(frac=1, random_state=(self.random_state + i))
                    self.y = self.y.sample(frac=1, random_state=(self.random_state + i))
                number_of_batches = int(self.X.shape[0] / self.batch_size)
                for n in range(0, number_of_batches):
                    for b in range(0, self.batch_size):
                        z, a = self.feedforward(np.array([self.X.iloc[n * self.batch_size + b, :].values]))
                        delta_w_temp, delta_b_temp = self.backpropagation(z, a, np.array([self.y.iloc[n * self.batch_size + b, :].values]), np.array([self.X.iloc[n * self.batch_size + b, :].values]))
                        for r in range(len(delta_w)):
                            delta_w[r] = delta_w[r] + delta_w_temp[r]
                            delta_b[r] = delta_b[r] + delta_b_temp[r]
                    for l in range(len(delta_w)):
                        delta_w[l] /= self.batch_size
                        delta_b[l] /= self.batch_size
                    for k in range(len(self.weights)):
                        if self.optimizer is None:
                            self.weights[k] += self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                            self.bias[k] += self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                            self.prev_delta_w[k] = self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                            self.prev_delta_b[k] = self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                        elif self.optimizer == "rmsprop":
                            self.rmsprop_w[k] = self.beta * self.rmsprop_w[k] + (1 - self.beta) * delta_w[k] ** 2
                            self.rmsprop_b[k] = self.beta * self.rmsprop_b[k] + (1 - self.beta) * delta_b[k] ** 2
                            self.weights[k] += self.learning_rate * delta_w[k] / (np.sqrt(self.rmsprop_w[k]) + self.epsilon)
                            self.bias[k] += self.learning_rate * delta_b[k] / (np.sqrt(self.rmsprop_b[k]) + self.epsilon)
                        elif self.optimizer == "adam":
                            self.t += 1
                            self.adam_m_w[k] = self.beta1_adam * self.adam_m_w[k] + (1 - self.beta1_adam) * delta_w[k]
                            self.adam_m_b[k] = self.beta1_adam * self.adam_m_b[k] + (1 - self.beta1_adam) * delta_b[k]
                            self.adam_v_w[k] = self.beta2_adam * self.adam_v_w[k] + (1 - self.beta2_adam) * delta_w[k] ** 2
                            self.adam_v_b[k] = self.beta2_adam * self.adam_v_b[k] + (1 - self.beta2_adam) * delta_b[k] ** 2
                            adam_m_w_hat = self.adam_m_w[k] / (1 - self.beta1_adam ** self.t)
                            adam_m_b_hat = self.adam_m_b[k] / (1 - self.beta1_adam ** self.t)
                            adam_v_w_hat = self.adam_v_w[k] / (1 - self.beta2_adam ** self.t)
                            adam_v_b_hat = self.adam_v_b[k] / (1 - self.beta2_adam ** self.t)
                            self.weights[k] += self.learning_rate * adam_m_w_hat / (np.sqrt(adam_v_w_hat) + self.epsilon)
                            self.bias[k] += self.learning_rate * adam_m_b_hat / (np.sqrt(adam_v_b_hat) + self.epsilon)
                if self.min_cost_function_train is None or self.min_cost_function_train > self.loss_function(self.y, self.predict(self.X)):
                    self.min_cost_function_train = self.loss_function(self.y, self.predict(self.X))
                if self.cv_X is not None and (self.min_cost_function_val is None or self.min_cost_function_val > self.loss_function(self.y, self.predict(self.X))):
                    self.min_cost_function_val = self.loss_function(self.cv_y, self.predict(self.cv_X))
                if self.cost_function_value is not None:
                    if self.cost_function_value > self.loss_function(self.y, self.predict(self.X)):
                        self.cost_function_value = self.loss_function(self.y, self.predict(self.X))
                        self.best_weights = cp.deepcopy(self.weights)
                        self.best_bias = cp.deepcopy(self.bias)
                if ((i+1) % self.print_evey_n_epoch == 0 or i == 0) and n == (number_of_batches - 1):
                    if self.cv_X is None:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14}")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14}")
                    else:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14} === Val MSE: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} Min: {self.min_cost_function_val:^14} ===")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14} === Val CrossEntropy: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} Min: {self.min_cost_function_val:^14} ===")
    
    def continue_fit(self, epochs):
        for i in range(epochs):
            delta_w = [np.zeros_like(w) for w in self.weights]
            delta_b = [np.zeros_like(b) for b in self.bias]
            if self.batch_size == None:
                for j in range(0, self.X.shape[0]):
                    z, a = self.feedforward(np.array([self.X.iloc[j, :].values]))
                    delta_w_temp, delta_b_temp = self.backpropagation(z, a, np.array([self.y.iloc[j, :].values]), np.array([self.X.iloc[j, :].values]))
                    for r in range(len(delta_w)):
                        delta_w[r] = delta_w[r] + delta_w_temp[r]
                        delta_b[r] = delta_b[r] + delta_b_temp[r]
                for l in range(len(delta_w)):
                    delta_w[l] /= self.X.shape[0]
                    delta_b[l] /= self.X.shape[0]
                for k in range(len(self.weights)):
                    if self.optimizer is None:
                        self.weights[k] += self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                        self.bias[k] += self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                        self.prev_delta_w[k] = self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                        self.prev_delta_b[k] = self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                    elif self.optimizer == "rmsprop":
                        self.rmsprop_w[k] = self.beta * self.rmsprop_w[k] + (1 - self.beta) * delta_w[k] ** 2
                        self.rmsprop_b[k] = self.beta * self.rmsprop_b[k] + (1 - self.beta) * delta_b[k] ** 2
                        self.weights[k] += self.learning_rate * delta_w[k] / (np.sqrt(self.rmsprop_w[k]) + self.epsilon)
                        self.bias[k] += self.learning_rate * delta_b[k] / (np.sqrt(self.rmsprop_b[k]) + self.epsilon)
                    elif self.optimizer == "adam":
                        self.t += 1
                        self.adam_m_w[k] = self.beta1_adam * self.adam_m_w[k] + (1 - self.beta1_adam) * delta_w[k]
                        self.adam_m_b[k] = self.beta1_adam * self.adam_m_b[k] + (1 - self.beta1_adam) * delta_b[k]
                        self.adam_v_w[k] = self.beta2_adam * self.adam_v_w[k] + (1 - self.beta2_adam) * delta_w[k] ** 2
                        self.adam_v_b[k] = self.beta2_adam * self.adam_v_b[k] + (1 - self.beta2_adam) * delta_b[k] ** 2
                        adam_m_w_hat = self.adam_m_w[k] / (1 - self.beta1_adam ** self.t)
                        adam_m_b_hat = self.adam_m_b[k] / (1 - self.beta1_adam ** self.t)
                        adam_v_w_hat = self.adam_v_w[k] / (1 - self.beta2_adam ** self.t)
                        adam_v_b_hat = self.adam_v_b[k] / (1 - self.beta2_adam ** self.t)
                        self.weights[k] += self.learning_rate * adam_m_w_hat / (np.sqrt(adam_v_w_hat) + self.epsilon)
                        self.bias[k] += self.learning_rate * adam_m_b_hat / (np.sqrt(adam_v_b_hat) + self.epsilon)
                if self.min_cost_function_train is None or self.min_cost_function_train > self.loss_function(self.y, self.predict(self.X)):
                    self.min_cost_function_train = self.loss_function(self.y, self.predict(self.X))
                if self.cv_X is not None and (self.min_cost_function_val is None or self.min_cost_function_val > self.loss_function(self.y, self.predict(self.X))):
                    self.min_cost_function_val = self.loss_function(self.cv_y, self.predict(self.cv_X))
                if self.cost_function_value is not None:
                    if self.cost_function_value > self.loss_function(self.y, self.predict(self.X)):
                        self.cost_function_value = self.loss_function(self.y, self.predict(self.X))
                        self.best_weights = cp.deepcopy(self.weights)
                        self.best_bias = cp.deepcopy(self.bias)
                if (i+1) % self.print_evey_n_epoch == 0 or i == 0:
                    if self.cv_X is None:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14}")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14}")
                    else:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14} === Val MSE: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} Min: {self.min_cost_function_val:^14} ===")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14} === Val CrossEntropy: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} Min: {self.min_cost_function_val:^14} ===")
            else:
                if i == 0:
                    self.X = self.X.sample(frac=1, random_state=self.random_state)
                    self.y = self.y.sample(frac=1, random_state=self.random_state)
                else:
                    self.X = self.X.sample(frac=1, random_state=(self.random_state + i))
                    self.y = self.y.sample(frac=1, random_state=(self.random_state + i))
                number_of_batches = int(self.X.shape[0] / self.batch_size)
                for n in range(0, number_of_batches):
                    for b in range(0, self.batch_size):
                        z, a = self.feedforward(np.array([self.X.iloc[n * self.batch_size + b, :].values]))
                        delta_w_temp, delta_b_temp = self.backpropagation(z, a, np.array([self.y.iloc[n * self.batch_size + b, :].values]), np.array([self.X.iloc[n * self.batch_size + b, :].values]))
                        for r in range(len(delta_w)):
                            delta_w[r] = delta_w[r] + delta_w_temp[r]
                            delta_b[r] = delta_b[r] + delta_b_temp[r]
                    for l in range(len(delta_w)):
                        delta_w[l] /= self.batch_size
                        delta_b[l] /= self.batch_size
                    for k in range(len(self.weights)):
                        if self.optimizer is None:
                            self.weights[k] += self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                            self.bias[k] += self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                            self.prev_delta_w[k] = self.learning_rate * delta_w[k] + self.momentum * self.prev_delta_w[k]
                            self.prev_delta_b[k] = self.learning_rate * delta_b[k] + self.momentum * self.prev_delta_b[k]
                        elif self.optimizer == "rmsprop":
                            self.rmsprop_w[k] = self.beta * self.rmsprop_w[k] + (1 - self.beta) * delta_w[k] ** 2
                            self.rmsprop_b[k] = self.beta * self.rmsprop_b[k] + (1 - self.beta) * delta_b[k] ** 2
                            self.weights[k] += self.learning_rate * delta_w[k] / (np.sqrt(self.rmsprop_w[k]) + self.epsilon)
                            self.bias[k] += self.learning_rate * delta_b[k] / (np.sqrt(self.rmsprop_b[k]) + self.epsilon)
                        elif self.optimizer == "adam":
                            self.t += 1
                            self.adam_m_w[k] = self.beta1_adam * self.adam_m_w[k] + (1 - self.beta1_adam) * delta_w[k]
                            self.adam_m_b[k] = self.beta1_adam * self.adam_m_b[k] + (1 - self.beta1_adam) * delta_b[k]
                            self.adam_v_w[k] = self.beta2_adam * self.adam_v_w[k] + (1 - self.beta2_adam) * delta_w[k] ** 2
                            self.adam_v_b[k] = self.beta2_adam * self.adam_v_b[k] + (1 - self.beta2_adam) * delta_b[k] ** 2
                            adam_m_w_hat = self.adam_m_w[k] / (1 - self.beta1_adam ** self.t)
                            adam_m_b_hat = self.adam_m_b[k] / (1 - self.beta1_adam ** self.t)
                            adam_v_w_hat = self.adam_v_w[k] / (1 - self.beta2_adam ** self.t)
                            adam_v_b_hat = self.adam_v_b[k] / (1 - self.beta2_adam ** self.t)
                            self.weights[k] += self.learning_rate * adam_m_w_hat / (np.sqrt(adam_v_w_hat) + self.epsilon)
                            self.bias[k] += self.learning_rate * adam_m_b_hat / (np.sqrt(adam_v_b_hat) + self.epsilon)
                if self.min_cost_function_train is None or self.min_cost_function_train > self.loss_function(self.y, self.predict(self.X)):
                    self.min_cost_function_train = self.loss_function(self.y, self.predict(self.X))
                if self.cv_X is not None and (self.min_cost_function_val is None or self.min_cost_function_val > self.loss_function(self.y, self.predict(self.X))):
                    self.min_cost_function_val = self.loss_function(self.cv_y, self.predict(self.cv_X))
                if self.cost_function_value is not None:
                    if self.cost_function_value > self.loss_function(self.y, self.predict(self.X)):
                        self.cost_function_value = self.loss_function(self.y, self.predict(self.X))
                        self.best_weights = cp.deepcopy(self.weights)
                        self.best_bias = cp.deepcopy(self.bias)
                if ((i+1) % self.print_evey_n_epoch == 0 or i == 0) and n == (number_of_batches - 1):
                    if self.cv_X is None:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14}")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14}")
                    else:
                        if self.activation_output == "linear":
                            print(f"=== Epoch: {i + 1:^7} === Train MSE: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14} === Val MSE: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} Min: {self.min_cost_function_val:^14} ===")
                        else:
                            print(f"=== Epoch: {i + 1:^7} === Train CrossEntropy: {self.loss_function_print(self.y, self.predict(self.X)):^14} Min: {self.min_cost_function_train:^14} === Val CrossEntropy: {self.loss_function_print(self.cv_y, self.predict(self.cv_X)):^14} Min: {self.min_cost_function_val:^14} ===")


    def predict(self, X):
        z, a = self.feedforward(X)
        return a[-1]
    
    def predict_best(self, X):
        z = []
        a = []
        for i in range(len(self.best_weights)):
            if i == 0:
                z.append(np.dot(X, self.best_weights[i]) + self.best_bias[i])
                a.append(self.activation_function(z[i]))
            elif i == len(self.best_weights)-1:
                z.append(np.dot(a[i-1], self.best_weights[i]) + self.best_bias[i])
                a.append(self.activation_output_function(z[i]))
            else:
                z.append(np.dot(a[i-1], self.best_weights[i]) + self.best_bias[i])
                a.append(self.activation_function(z[i]))
        return a[-1]
    
    def initial_weights(self):
        if self.random_w_bool == True:
            np.random.seed(self.random_state)
            wts = [0] * (len(self.hidden_layer_sizes) + 1)
            bs = [0] * (len(self.hidden_layer_sizes) + 1)
            for i in range(len(self.hidden_layer_sizes) + 1):
                if i == 0:
                    wts[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(self.X.shape[1],self.hidden_layer_sizes[i])).astype(np.float64)
                    bs[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(1, self.hidden_layer_sizes[i])).astype(np.float64)
                elif i == len(self.hidden_layer_sizes):
                    wts[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(self.hidden_layer_sizes[i - 1], self.output_size)).astype(np.float64)
                    bs[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(1, self.output_size)).astype(np.float64)
                else:
                    wts[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i])).astype(np.float64)
                    bs[i] = np.random.uniform(low=self.random_w_min, high=self.random_w_max, size=(1, self.hidden_layer_sizes[i])).astype(np.float64)
            return [wts, bs]
        else:
            return [[self.declared_weights],[self.declared_bias]]

    
    def get_params(self, deep=True):
        return {"hidden_layer_sizes": self.hidden_layer_sizes,
                "output_size": self.output_size,
                "optimizer": self.optimizer,
                 "activation": self.activation,
                   "activation_output": self.activation_output,
                     "epochs": self.epochs,
                     "batch_size": self.batch_size,
                       "learning_rate": self.learning_rate,
                         "random_state": self.random_state,
                           "random_w_min": self.random_w_min,
                             "random_w_max": self.random_w_max,
                             "initial_weights": self.initial_weights()[0],
                             "initial_bias": self.initial_weights()[1]}
    
