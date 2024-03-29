import torch
import numpy as np
import random
import model
from dataset import QualityDataset
from torch.utils.data import DataLoader
import random
import math
import util

DATA_PATH = "/data1/hifi_consensus/processed_data/chr2_deep_filtered.txt"
RAW_PATH = "/data1/hifi_consensus/processed_data/chr2_deep.txt"
FILTER_PATH =  "/data1/hifi_consensus/processed_data/filters"
MODEL_PATH = "./result/model/multi_layered_model_new.pt"
CONTEXT_COUNT = 3
EXTRA_COUNT = 20

def main():
    # set the seed
    torch.manual_seed(1)
    random.seed(3)
    np.random.seed(2)
    #util.check_line_sizes_in_file(DATA_PATH)
    #util.filter_data_using_confident_germline_indel_depth("chr20", RAW_PATH, "/data1/hifi_consensus/processed_data/filters", DATA_PATH)
    #evaluate_model()
    #util.output_the_base_corrections(DATA_PATH, "result")
    #util.filter_data_using_confident_germline_indel_depth("chr18", RAW_PATH, "/data1/hifi_consensus/processed_data/filters", DATA_PATH)
    #util.clean_the_data (DATA_PATH, "/data1/hifi_consensus/processed_data/chr18_ip_pw_cleaned.txt")
    #util.print_pacbio_scores(DATA_PATH)
    #train_model()
    #view_result()
    #evaluate_model()
    #util.filter_data_deep_consensus("chr2", RAW_PATH, FILTER_PATH, DATA_PATH)
    util.print_deep_scores(DATA_PATH)
    return

# this function will train the model using the train data
def train_model():
    tensor_length = pow(5, CONTEXT_COUNT) + EXTRA_COUNT
    # train parameters
    learningRate = 0.000001
    epochs = 10
    batch_size = 1024 * 16
    # data loading
    train_dataset = QualityDataset (DATA_PATH, False, CONTEXT_COUNT)
    train_loader = DataLoader (
        dataset = train_dataset,
        batch_size = batch_size,
        num_workers = 64,
        shuffle = False,
        drop_last = True
    )
    # build the model object
    lr_model = model.quality_model_1_layer(CONTEXT_COUNT, EXTRA_COUNT)

    # define custom weights
    custom_weight = torch.rand(lr_model.linear.weight.shape)
    #first_layer_size = tensor_length
    first_layer_size = int(tensor_length / 2)
    # calling base count
    for i in range(0, first_layer_size):
        for j in range(0, tensor_length):
            custom_weight[i][j] = torch.tensor(0.0)
        custom_weight[i][tensor_length - 4] = torch.tensor(+1.0437)
        # other base count
        custom_weight[i][tensor_length - 3] = torch.tensor(-0.2337)
        custom_weight[i][tensor_length - 2] = torch.tensor(-0.9995)
        custom_weight[i][tensor_length - 1] = torch.tensor(-1.0)
        # pacbio qual
        custom_weight[i][tensor_length - 5] = torch.tensor(+1.0)
    # put the weights in the model
    lr_model.linear.weight = torch.nn.Parameter(custom_weight)

    # optimizer
    optimizer = torch.optim.SGD(lr_model.parameters(), lr = learningRate)
    criterion = torch.nn.MSELoss()

    # # load the previous saved trained model
    checkpoint = torch.load(MODEL_PATH)
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
            less = False
            if loss.item() < 0.0001:
                less = True
            print('epoch {}, loss {}, batch {}/{} {}'.format(epoch, str(loss.item()).zfill(22), batch_idx, num_batches, less))
        # save the trained model after each epoch
        torch.save({'epoch': epoch, 'model_state_dict': lr_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, MODEL_PATH)
        # reshuffle the data for the next epoch
        # train_dataset.reshuffle()

# this function will evalute the model and aggregate the results (output of the model for wrong and right)
def evaluate_model():
    tensor_length = pow(5, CONTEXT_COUNT) + EXTRA_COUNT
    # arrays to save the result
    error_counts = [0] * 94
    all_counts = [0] * 94
    fixed_polished_counts = [0] * 4
    messed_polished_counts = [0] * 4
    batch_size = 1024
    # get the data to test
    eval_dataset = QualityDataset (DATA_PATH, False, CONTEXT_COUNT)
    eval_loader = DataLoader (
        dataset = eval_dataset,
        batch_size = batch_size,
        num_workers = 64,
        shuffle = False,
        drop_last = True
    )
    eval_len = len(eval_loader)
    # load the model
    lr_model = model.quality_model_1_layer(CONTEXT_COUNT, EXTRA_COUNT)
    checkpoint = torch.load(MODEL_PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])
    # run the data
    with torch.no_grad():
        lr_model.eval()
        for batch_idx, (batch_inputs, batch_labels, batch_polished) in enumerate(eval_loader):
            pred = lr_model(batch_inputs)
            for i in range(len(batch_inputs)):
                pacbio_qual = batch_inputs[i][0][tensor_length - 5].item()
                position = int(-10 * math.log((1.0 - pred[i].item()) + 0.000000000001, 10))
                if position > 92:
                    position = 93
                all_counts[position] += 1
                if batch_labels[i].item() < 0.5 and pacbio_qual > 0.001:
                    if batch_polished[i] == 0:
                        error_counts[position] += 1
                    else:
                        fixed_polished_counts[batch_polished[i]] += 1
                elif batch_labels[i].item() > 0.5:
                    messed_polished_counts[batch_polished[i]] += 1
            break
            print("Evaluating {}/{}".format(batch_idx, eval_len))
    print(all_counts)
    print(error_counts)
    print(fixed_polished_counts)
    print(messed_polished_counts)
    lr_model.train()

def view_result():
    # show the model parameters
    lr_model = model.quality_model_1_layer(CONTEXT_COUNT, EXTRA_COUNT)
    tensor_length = pow(5, CONTEXT_COUNT) + EXTRA_COUNT
    checkpoint = torch.load(MODEL_PATH)
    lr_model.load_state_dict(checkpoint['model_state_dict'])
    for name, param in lr_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    # get and view 100 error data and 100 correct data
    required_number = 10
    batch_size = 1024
    correct_tensor_len = 0
    correct_tensor = torch.empty((required_number, tensor_length + 2), dtype = torch.float32)
    error_tensor_len = 0
    error_tensor = torch.empty((required_number, tensor_length + 2), dtype = torch.float32)
    eval_dataset = QualityDataset (DATA_PATH, False, CONTEXT_COUNT)
    eval_loader = DataLoader (
        dataset = eval_dataset,
        batch_size = batch_size,
        num_workers = 4,
        shuffle = False,
        drop_last = True
    )
    eval_len = len(eval_loader)
    with torch.no_grad():
        lr_model.eval()
        for batch_idx, (batch_inputs, batch_labels) in enumerate(eval_loader):
            pred = lr_model(batch_inputs)
            for i in range(len(batch_inputs)):
                if (batch_labels[i].item() < 0.9) and (error_tensor_len < required_number):
                    pacbio_qual = batch_inputs[i][0][tensor_length - 5].item()
                    if pacbio_qual > 0.001:
                        error_tensor[error_tensor_len] = torch.concat((batch_inputs[i], batch_labels[i], pred[i]), dim = 1)
                        error_tensor_len += 1
                elif (batch_labels[i].item() > 0.9) and (correct_tensor_len < required_number):
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
    return


def print_result_tensor(obtained_tensor):
    tensor_length = pow(5, CONTEXT_COUNT) + EXTRA_COUNT
    # get the three base context from first 64 bits
    for idx, value in enumerate(obtained_tensor[0: tensor_length - EXTRA_COUNT]):
        if value > 0.5:
            converted_number = idx
    base_context = util.convert_bits_to_bases(converted_number, CONTEXT_COUNT)
    pacbio_qual = obtained_tensor[tensor_length - 5].item()
    parallel_nodes = obtained_tensor[tensor_length - 4: tensor_length]
    error_rate = obtained_tensor[tensor_length + 1].item()
    calc_qual = int(-10 * math.log(error_rate, 10))
    print("three_base_context {} parallel_nodes {} pacbio_qual {} error_rate {} calculated_qual {} ".format(base_context, parallel_nodes, pacbio_qual, error_rate, calc_qual))
    return

if __name__ == "__main__":
    main()