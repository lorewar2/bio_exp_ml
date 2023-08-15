
import torch
import random
import numpy as np
import scipy.special

def print_topology_cut_scores():
    # arrays to save the result
    error_counts = [0] * 93
    all_counts = [0] * 93
    path = "./data/train_file.txt"
    file = open(path, "r")
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        if len(split_txt) != 11:
            continue
        result = split_txt[0]
        position = int(split_txt[6])
        all_counts[position] += 1
        if result == "true":
            error_counts[position] += 1
        if index % 100000 == 0:
            print("Running line {}".format(index))
    print(error_counts)
    print(all_counts)
    return

def calculate_topology_score(calling_base, base_A_count, base_C_count, base_G_count, base_T_count, num_of_reads, prob):
    ln_prob_base_A = np.log(0.25)
    ln_prob_base_C = np.log(0.25)
    ln_prob_base_G = np.log(0.25)
    ln_prob_base_T = np.log(0.25)
    
    ln_prob_data_given_A = np.log(calculate_binomial(num_of_reads, base_A_count, prob))
    ln_prob_data_given_C = np.log(calculate_binomial(num_of_reads, base_C_count, prob))
    ln_prob_data_given_G = np.log(calculate_binomial(num_of_reads, base_G_count, prob))
    ln_prob_data_given_T = np.log(calculate_binomial(num_of_reads, base_T_count, prob))

    ln_sum_of_probabilities = ln_prob_data_given_A + ln_prob_base_A
    ln_sum_of_probabilities = np.logaddexp(ln_sum_of_probabilities, ln_prob_data_given_C + ln_prob_base_C)
    ln_sum_of_probabilities = np.logaddexp(ln_sum_of_probabilities, ln_prob_data_given_G + ln_prob_base_G)
    ln_sum_of_probabilities = np.logaddexp(ln_sum_of_probabilities, ln_prob_data_given_T + ln_prob_base_T)

    if calling_base == "A":
        correct_rate = np.exp(ln_prob_data_given_A + ln_prob_base_A - ln_sum_of_probabilities)
    elif calling_base == "C":
        correct_rate = np.exp(ln_prob_data_given_C + ln_prob_base_C - ln_sum_of_probabilities)
    elif calling_base == "G":
        correct_rate = np.exp(ln_prob_data_given_G + ln_prob_base_G - ln_sum_of_probabilities)
    elif calling_base == "T":
        correct_rate = np.exp(ln_prob_data_given_T + ln_prob_base_T - ln_sum_of_probabilities)

    error_rate = 1.0 - correct_rate
    quality_score = (-10.00) * np.log10(error_rate + 0.000000000000000000001)
    #print(quality_score)
    return quality_score

def calculate_binomial(n, k, prob):
    binomial_coeff = scipy.special.binom(n, k)
    success = np.power(prob, k)
    failure = np.power(1.00 - prob, n - k)
    return (binomial_coeff * success * failure)

def pipeline_calculate_topology_score_with_probability():
    # arrays to save the result
    error_counts = [0] * 300
    all_counts = [0] * 300
    path = "./data/train_file.txt"
    file = open(path, "r")
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        if len(split_txt) != 11:
            continue
        result = split_txt[0]
        parallel_vec_s = [split_txt[7], split_txt[8], split_txt[9], split_txt[10]]
        char_remov = ["]", "[", ",", "\n"]
        for char in char_remov:
            for index_s in range(len(parallel_vec_s)):
                temp = parallel_vec_s[index_s].replace(char, "")
                parallel_vec_s[index_s] = temp
        parallel_vec_f = []
        for parallel in parallel_vec_s:
            parallel_vec_f.append(float(parallel))
        recalculated_score = int(calculate_topology_score(split_txt[1][1], parallel_vec_f[0], parallel_vec_f[1], parallel_vec_f[2], parallel_vec_f[3], int(split_txt[6]), 0.95))
        all_counts[recalculated_score] += 1
        if result == "true":
            error_counts[recalculated_score] += 1
        if index % 100000 == 0:
            print("Running line {}".format(index))
    print(error_counts)
    print(all_counts)
    return

pipeline_calculate_topology_score_with_probability()

def old_data_loader(path, start, length, get_random):
    file = open(path, "r")
    label_tensor = torch.empty((length, 1), dtype = torch.float32)
    input_tensor = torch.empty((length, 17), dtype = torch.float32)
    index = 0
    tensor_pos = 0
    for line in file:
        
        # only get the specified section in file
        if index < start:
            if line != "\n":
                index += 1
            continue
        # if random is required break when tensor is full
        elif get_random:
            if tensor_pos >= length:
                break
        # if random is not required then break when after the specified length
        elif index > start and index >= start + length:
            break
        # continue if random choice is true
        if get_random and random.choice([True, True, True, True, False]):
            continue
        if line != "\n":
            split_txt = line.split(" ")
            if len(split_txt) != 11:
                index += 1
                continue
            # get three base context in one hot encoded
            encoded_bases = one_hot_encoding_bases(split_txt[1][0]) + one_hot_encoding_bases(split_txt[1][1]) + one_hot_encoding_bases(split_txt[1][2])
            # get quality in float
            quality = float(split_txt[3]) / 100
            # get the num of parallel bases in float
            parallel_vec_s = [split_txt[7], split_txt[8], split_txt[9], split_txt[10]]
            char_remov = ["]", "[", ",", "\n"]
            for char in char_remov:
                for index_s in range(len(parallel_vec_s)):
                    temp = parallel_vec_s[index_s].replace(char, "")
                    parallel_vec_s[index_s] = temp
            parallel_vec_f = []
            for parallel in parallel_vec_s:
                parallel_vec_f.append(float(parallel))
            # rearrange so that the calling base num first and rest in decending order
            sorted_vec = rearrange_sort_parallel_bases(parallel_vec_f, split_txt[1][1])
            # make and append to the input tensor,
            input_tensor[tensor_pos] = torch.tensor([encoded_bases + [quality] + sorted_vec])
            # append to result tensor,
            result = split_txt[0]
            if result == "false":
                label_tensor[tensor_pos] = torch.tensor([[1.0]])
                # continue if we want errors and its not a error
            else:
                label_tensor[tensor_pos] = torch.tensor([[0.00]])
            
            if index % 10000 == 0 and get_random == False:
                print("Going through line: {} getting data point: {}/{}".format(index, index - start, length))
            elif tensor_pos % 100 == 0 and get_random == True:
                print("Going through line: {} getting data point: {}/{}".format(index, tensor_pos, length))
            index += 1
            tensor_pos += 1
    file.close()
    return (input_tensor, label_tensor)

def rearrange_sort_parallel_bases(parallel_vec, base):
    selected_base = parallel_vec[0]
    if base == "A":
        selected_base = parallel_vec[0]
        del parallel_vec[0]
    elif base == "C":
        selected_base = parallel_vec[1]
        del parallel_vec[1]
    elif base == "G":
        selected_base = parallel_vec[2]
        del parallel_vec[2]
    elif base == "T":
        selected_base = parallel_vec[3]
        del parallel_vec[3]
    parallel_vec.sort(reverse = True)
    parallel_vec = [selected_base] + parallel_vec
    return parallel_vec

def one_hot_encoding_bases(base):
    one_hot_base = [0.0] * 4
    if base == "A":
        one_hot_base[0] = 1.0
    elif base == "C":
        one_hot_base[1] = 1.0
    elif base == "G":
        one_hot_base[2] = 1.0
    elif base == "T":
        one_hot_base[3] = 1.0
    return one_hot_base