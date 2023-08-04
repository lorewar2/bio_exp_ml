import torch
import numpy as np
import random
import model
from dataset import QualityDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_preprocess import get_data
import random
import math

PATH = "./result/model/chr1_1bil_model.pt"

def main():
    train_dataset = QualityDataset ("data/train_file.txt", "data/train_file.idx")
    train_loader = DataLoader (
        dataset = train_dataset,
        batch_size = 16,
        shuffle = False,
        drop_last = True
    )
    for batch_idx, (x, y) in enumerate(train_loader):
        print(batch_idx)
        print(x[0])
    # set the seed
    torch.manual_seed(0)
    random.seed(2)
    #train_model()
    #evaluate_model()
    #test()
    return

def test():
    lr_model = model.quality_model()
    checkpoint = torch.load(PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])
    for name, param in lr_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    return
# this function will evalute the model and aggregate the results (output of the model for wrong and right)
def evaluate_model():
    # arrays to save the result
    error_counts = [0] * 93
    all_counts = [0] * 93
    # get the data to test
    (eval_inputs, eval_labels) = get_data("data/chr21_ml_file.txt", 0, 100_000_000, False)

    # load the model
    lr_model = model.quality_model()
    checkpoint = torch.load(PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])

    # run the data
    with torch.no_grad():
        lr_model.eval()
        pred = lr_model(eval_inputs)
        for i in range(len(eval_inputs)):
            position = int(-10 * math.log(1 - pred[i].item(), 10))
            all_counts[position] += 1
            if eval_labels[i].item() < 0.9:
                error_counts[position] += 1
    print(all_counts)
    print(error_counts)

# this function will train the model using the train data
def train_model():
    # get the train data (1_000_000 normal)
    (train_inputs, train_labels) = get_data("data/train_file.txt", 0, 10_000_000, False)

    # get the test data (random test data)
    (test_inputs, test_labels) = get_data("data/test_file.txt", random.randint(0, 1_000_000), 100, True)

    # converting inputs and labels to Variable
    train_inputs = Variable(train_inputs)
    train_labels = Variable(train_labels)

    # train parameters
    learningRate = 0.0001
    epochs = 10
    batch_size = 1024

    # build the model object
    lr_model = model.quality_model()

    # optimizer 
    optimizer = torch.optim.SGD(lr_model.parameters(), lr=learningRate)
    criterion = torch.nn.MSELoss()
    #torch.save({'epoch': 0, 'model_state_dict': lr_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': 9999}, PATH)
    # load the previous saved trained model
    checkpoint = torch.load(PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

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
            loss = criterion(outputs, batch_labels)
            # get gradients w.r.t to parameters
            loss.backward()
            # update parameters
            optimizer.step()
            print('epoch {}, loss {}, val_loss {} batch {}/{}'.format(epoch, loss.item(), val_loss, i, train_inputs.size()[0]))

    # save the trained model
    torch.save({'epoch': epoch, 'model_state_dict': lr_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, PATH)

    print("test!")
    for i in range(len(test_inputs)):
        print("input {} pacbio_qual {} output {}  label {}".format(test_inputs[i][12:], test_inputs[i][12].item(), lr_model(test_inputs)[i].item(), test_labels[i].item()))

if __name__ == "__main__":
    main()
   


