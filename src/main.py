import torch
import numpy as np
import random
import model
from torch.autograd import Variable
from data_preprocess import get_data
import random

# get the train data
train_data_start = 0 #random.randint(0, 8_000_000)
train_data_len = 1_000_000
print("Getting", train_data_len, "train data from", train_data_start, "~")
(train_inputs, train_labels) = get_data("data/test_file.txt", train_data_start, train_data_len, False)

# get the test data
test_data_start = 0
test_data_len = 100
print("Getting", test_data_len, "train data from", test_data_start, "~")
(test_inputs, test_labels) = get_data("data/test_file.txt", test_data_start, test_data_len, True)

# converting inputs and labels to Variable
train_inputs = Variable(train_inputs)
train_labels = Variable(train_labels)

# train the model
learningRate = 0.00001
epochs = 10
batch_size = 10000

# build the model object
lr_model = model.LR(8, 1)
# optimizer 
optimizer = torch.optim.SGD(lr_model.parameters(), lr=learningRate)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
    permutation = torch.randperm(train_inputs.size()[0])
    for i in range(0, train_inputs.size()[0], batch_size):
        # clear gradient buffers 
        optimizer.zero_grad()
        # get the batch 
        indices = permutation[i: i + batch_size]
        batch_inputs, batch_labels = train_inputs[indices], train_labels[indices]
        # get output from the model, given the inputs
        outputs = lr_model(batch_inputs)
        # get loss for the predicted output
        #print(outputs)
        loss = criterion(outputs, batch_labels)
        # get gradients w.r.t to parameters
        loss.backward()
        # update parameters
        optimizer.step()
        print('epoch {}, loss {}, batch {}/{}'.format(epoch, loss.item(), i, train_inputs.size()[0]))

print("test!")
for i in range(len(test_inputs)):
    print("input {} output {} label {}".format(test_inputs[i], lr_model(test_inputs[i]).item(), test_labels[i].item()))

