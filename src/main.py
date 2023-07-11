import torch
import numpy as np
import random
import model
from torch.autograd import Variable
from data_preprocess import get_data
import random

# set the seed
torch.manual_seed(0)

# get the train data
train_data_start = 0 #random.randint(0, 8_000_000)
train_data_len = 1_000_000
print("Getting", train_data_len, "train data from", train_data_start, "~")
(train_inputs, train_labels) = get_data("data/train_file.txt", train_data_start, train_data_len, False)

# get the test data
test_data_start = 0
test_data_len = 100
print("Getting", test_data_len, "test data from", test_data_start, "~")
(test_inputs, test_labels) = get_data("data/test_file.txt", test_data_start, test_data_len, True)

# converting inputs and labels to Variable
train_inputs = Variable(train_inputs)
train_labels = Variable(train_labels)

# train parameters
learningRate = 0.00001
epochs = 3
batch_size = 1024

# build the model object
lr_model = model.quality_model()
# optimizer 
optimizer = torch.optim.SGD(lr_model.parameters(), lr=learningRate)
criterion = torch.nn.MSELoss()

# train the model
for epoch in range(epochs):
    val_outputs = lr_model(test_inputs)
    val_loss = criterion(val_outputs, test_labels)
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
        print('epoch {}, loss {}, val_loss {} batch {}/{}'.format(epoch, loss.item(), val_loss, i, train_inputs.size()[0]))

print("test!")
for i in range(len(test_inputs)):
    print("input {} output {} label {}".format(test_inputs[i], lr_model(test_inputs)[i].item(), test_labels[i].item()))

