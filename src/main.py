import torch
import numpy as np
import random
import model
from torch.autograd import Variable
# testing
# Test data
# Setting weights and bias
w = torch.tensor([[3.0], 
                  [4.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)
 
# Defining our forward function for prediction
def forward(x):
    # using .mm module for matrix multiplication 
    y_pred = torch.mm(x, w) + b
    return y_pred
 
# define a tensor 'X' with multiple rows
X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0], 
                  [5.0, 6.0]])
 
# Making predictions for Multi-Dimensional tensor "X"
y_pred = forward(X)
print("Predictions for 'X': ", y_pred)
  
# create dummy data for training
x_test = np.array([[0.0, 0.0]])
for i in range(20):
    x_test = np.append(x_test, [[random.uniform(0.0, 10.0), random.uniform(0.0, 10.0)]], axis=0)
print(x_test)
x_train = torch.from_numpy(np.float32(x_test))
y_train = forward(x_train)
print(y_train)

# train the model 
learningRate = 0.001 
epochs = 100

# build the model object
lr_model = model.LR(2, 1)
# optimizer 
optimizer = torch.optim.SGD(lr_model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    inputs = Variable(x_train)
    labels = Variable(y_train)

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()
    # get output from the model, given the inputs
    outputs = lr_model(inputs)

    # get loss for the predicted output
    criterion = torch.nn.MSELoss() 
    loss = criterion(outputs, labels)
    # get gradients w.r.t to parameters
    loss.backward()
    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# make predictions for multiple input samples of 'X'
y_pred  = lr_model(X)
print("Predictions for 'X': ", y_pred)
