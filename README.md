# MLP
Multilayer Perceptron built from scratch in Python using NumPy.
Efficient and easy to use.

Features:
- Stochastic gradient descent
- Batch gradient descent
- Mini batch gradient descent
- Sigmoid/ReLU/tanh/linear activation functions
- Sigmoid/linear/softmax activation function in the output layer
- MSE/cross-entropy cost function
- ADAM/RMSprop/Momentum optimizers
- Examining nn's performance on validation set during the training process
- Declaring own weights and bias
- Possible to specify a range of the initial weights
- Possible to resume the training process

Example:
classifier = MLP(X=X_train, y=y_train, hidden_layer_sizes=[10,10], activation="tanh",
                  activation_output="softmax", learning_rate=0.0005, epochs=1500, random_w_min = -5, random_w_max = 5,
                  print_evey_n_epoch=100, optimizer="adam", beta1_adam=0.9, beta2_adam=0.999, epsilon=1e-8, batch_size=32)

classifier.fit()

classifier.predict(X_test)

Comments:

- The class label shouldn't be one hot encoded.
- Prediciton is made in one hot format.
- Weights and bias should be specified in a list as follows:
  - weights: [array([[ 0.03273557,  0.02457152, -0.02957698, -0.02303797, -0.06907668],
         [-0.08002465,  0.03324651, -0.0635965 ,  0.06544793, -0.01471366]]), 
         array([[-0.05257053, -0.0834931 ],
         [ 0.07451358, -0.0149692 ],
         [ 0.07215782, -0.06897695],
         [-0.01981866,  0.02585352],
         [-0.04474592,  0.034605  ]])]      (NOTE: weights between the input layer and the first hidden layer should transposed,
                                            each numpy array correspond to each level of weights, 
                                            example for nn with 1 hidden layer with 5 neurons, 2 inputs and 2 outputs)
  - bias: [array([[ 0.02885715,  0.01938363, -0.00163914,  0.07433262,  0.00135008]]),
  array([[0.06622996, 0.06626889]])]        (NOTE: bias should transposed, each numpy array correspond to each level of bias,
                                            example for nn with 1 hidden layer with 5 neurons, 2 inputs and 2 outputs)
                                            
