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
    # set the seed
    torch.manual_seed(0)
    random.seed(2)
    train_model()
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
    batch_size = 1024
    # get the data to test
    eval_dataset = QualityDataset ("data/train_file.txt", "data/train_file.idx")
    eval_loader = DataLoader (
        dataset = eval_dataset,
        batch_size = batch_size,
        num_workers = 4,
        shuffle = True,
        drop_last = True
    )
    # load the model
    lr_model = model.quality_model()
    checkpoint = torch.load(PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])

    # run the data
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_labels) in enumerate(eval_loader):
            lr_model.eval()
            pred = lr_model(batch_inputs)
            for i in range(len(batch_inputs)):
                position = int(-10 * math.log(1 - pred[i].item(), 10))
                all_counts[position] += 1
                if batch_labels[i].item() < 0.9:
                    error_counts[position] += 1
    print(all_counts)
    print(error_counts)

# this function will train the model using the train data
def train_model():
    # train parameters
    learningRate = 0.00001
    epochs = 1
    batch_size = 1024    
    # data loading
    train_dataset = QualityDataset ("data/train_file.txt", "data/train_file.idx")
    train_loader = DataLoader (
        dataset = train_dataset,
        batch_size = batch_size,
        num_workers = 0,
        shuffle = True,
        drop_last = True
    )
    # build the model object
    lr_model = model.quality_model()

    # optimizer 
    optimizer = torch.optim.SGD(lr_model.parameters(), lr=learningRate)
    criterion = torch.nn.MSELoss()
    torch.save({'epoch': 0, 'model_state_dict': lr_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': 9999}, PATH)
    # load the previous saved trained model
    checkpoint = torch.load(PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    num_batches = len(train_loader)
    # train loop
    for epoch in range(epochs):
        for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
            # clear gradient buffers 
            optimizer.zero_grad()
            # get output from the model, given the inputs
            outputs = lr_model(batch_inputs)
            # get loss for the predicted output
            loss = criterion(outputs, batch_labels)
            # get gradients w.r.t to parameters
            loss.backward()
            # update parameters
            optimizer.step()
            print('epoch {}, loss {}, batch {}/{}'.format(epoch, loss.item(), batch_idx, num_batches))
    # save the trained model
    torch.save({'epoch': epoch, 'model_state_dict': lr_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, PATH)
    #for i in range(len(test_inputs)):
        #print("input {} pacbio_qual {} output {}  label {}".format(test_inputs[i][12:], test_inputs[i][12].item(), lr_model(test_inputs)[i].item(), test_labels[i].item()))

if __name__ == "__main__":
    main()
   


