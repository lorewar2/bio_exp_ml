import torch
import numpy as np
import random
import model
from dataset import QualityDataset
from torch.utils.data import DataLoader
import random
import math

PATH = "./result/model/chr1_1bil_model.pt"

def main():
    # set the seed
    torch.manual_seed(0)
    random.seed(2)
    #train_model()
    #evaluate_model()
    #view_result()
    return

def view_result():
    # show the model parameters
    lr_model = model.quality_model()
    checkpoint = torch.load(PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])
    for name, param in lr_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    # get and view 100 error data and 100 correct data
    required_number = 5
    batch_size = 1024
    correct_tensor_len = 0
    correct_tensor = torch.empty((required_number, 71), dtype = torch.float32)
    error_tensor_len = 0
    error_tensor = torch.empty((required_number, 71), dtype = torch.float32)
    eval_dataset = QualityDataset ("data/train_file.txt", "data/train_file.idx")
    eval_loader = DataLoader (
        dataset = eval_dataset,
        batch_size = batch_size,
        num_workers = 4,
        shuffle = True,
        drop_last = True
    )
    eval_len = len(eval_loader)
    with torch.no_grad():
        lr_model.eval()
        for batch_idx, (batch_inputs, batch_labels) in enumerate(eval_loader):
            pred = lr_model(batch_inputs)
            for i in range(len(batch_inputs)):
                if (batch_labels[i].item() < 0.9) and (error_tensor_len < required_number):
                    pacbio_qual = batch_inputs[i][0][64].item()
                    if pacbio_qual > 0.001:
                        error_tensor[error_tensor_len] = torch.concat((batch_inputs[i], batch_labels[i], pred[i]), dim = 1)
                        error_tensor_len += 1
                elif (correct_tensor_len < required_number):
                    correct_tensor[correct_tensor_len] = torch.concat((batch_inputs[i], batch_labels[i], pred[i]), dim = 1)
                    correct_tensor_len += 1
            print("Processing {}/{} found correct {} error {}".format(batch_idx, eval_len, correct_tensor_len, error_tensor_len))
            if (correct_tensor_len >= required_number) and (error_tensor_len >= required_number):
                break
    lr_model.train()
    # display the data nicely
    print("errors")
    for error in error_tensor:
        print_result_tensor(error)
    print("correct")
    for correct in correct_tensor:
        print_result_tensor(correct)
    #print(error_tensor)
    #print(correct_tensor)
    return

def print_result_tensor(obtained_tensor):
    # get the three base context from first 64 bits
    three_base_context_64bit = 0
    for idx, value in enumerate(obtained_tensor[0:64]):
        if value > 0.5:
            three_base_context_64bit = idx
    first_base = three_base_context_64bit % 4
    second_base = int(three_base_context_64bit / 4) % 4
    third_base = int(three_base_context_64bit / 16) % 4
    three_base_context = [int_to_base(first_base), int_to_base(second_base), int_to_base(third_base)]
    pacbio_qual = obtained_tensor[64].item()
    parallel_nodes = obtained_tensor[65:69]
    correct_rate = obtained_tensor[70].item()
    calc_qual = int(-10 * math.log(1 - correct_rate, 10))
    print("three_base_context {} parallel_nodes {} pacbio_qual {} correct_rate {} calculated_qual {} ".format(three_base_context, parallel_nodes, pacbio_qual, correct_rate, calc_qual))
    return

def int_to_base(number):
    base = 'P'
    if number == 0:
        base = 'A'
    elif number == 1:
        base = 'C'
    elif number == 2:
        base = 'G'
    elif number == 3:
        base = 'T'
    return base

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
        shuffle = False,
        drop_last = True
    )
    eval_len = len(eval_loader)
    # load the model
    lr_model = model.quality_model()
    checkpoint = torch.load(PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])

    # run the data
    with torch.no_grad():
        lr_model.eval()
        for batch_idx, (batch_inputs, batch_labels) in enumerate(eval_loader):
            pred = lr_model(batch_inputs)
            for i in range(len(batch_inputs)):
                pacbio_qual = batch_inputs[i][0][64].item()
                position = int(-10 * math.log(1 - pred[i].item(), 10))
                all_counts[position] += 1
                if batch_labels[i].item() < 0.9 and pacbio_qual > 0.001:
                    error_counts[position] += 1
            print("Evaluating {}/{}".format(batch_idx, eval_len))
    print(all_counts)
    print(error_counts)
    lr_model.train()

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

    # define custom weights
    custom_weight = torch.rand(lr_model.linear.weight.shape)
    # calling base count
    custom_weight[0][65] = torch.tensor(1.0)
    # other base count
    custom_weight[0][66] = torch.tensor(-1.0)
    custom_weight[0][67] = torch.tensor(-1.0)
    custom_weight[0][68] = torch.tensor(-1.0)
    # pacbio qual
    custom_weight[0][64] = torch.tensor(1.0)
    # put the weights in the model
    lr_model.linear.weight = torch.nn.Parameter(custom_weight)

    # optimizer 
    optimizer = torch.optim.SGD(lr_model.parameters(), lr=learningRate)
    criterion = torch.nn.MSELoss()

    # torch.save({'epoch': 0, 'model_state_dict': lr_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': 9999}, PATH)
    # # load the previous saved trained model
    # checkpoint = torch.load(PATH)
    # lr_model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
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

if __name__ == "__main__":
    main()
   


