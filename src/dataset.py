import torch
import os
import numpy as np
import util

class QualityDataset(torch.utils.data.Dataset):
    def __init__(self, file_loc, shuffle_all, base_context_count):
        self.file_loc = file_loc
        self.shuffle_all = shuffle_all
        self.base_context_count = base_context_count
        self.tensor_length = pow(5, base_context_count) + 17
        # get len and save it
        with open(file_loc) as f:
            f.seek(0, 2)
            offset = f.tell()
            self.len = int((offset - 106) / 106) - 1
        # load all data
        if self.shuffle_all:
            # make a shuffled index
            self.index_array = np.arange(0, self.len)
            np.random.shuffle(self.index_array)
            print("initializing complete")

    def __len__(self):
        return self.len

    def reshuffle(self):
        if self.shuffle_all:
            np.random.shuffle(self.index_array)
            print("reshuffling complete")
        return

    def __getitem__(self, index):
        if self.shuffle_all == True:
            input_tensor, label_tensor = self.retrieve_item_from_disk(self.index_array[index])
        else:
            input_tensor, label_tensor = self.retrieve_item_from_disk(index)
        return input_tensor, label_tensor

    def retrieve_item_from_disk(self, index):
        # search the index file to file the location # index offset is 106
        retrieved_line = ""
        with open(self.file_loc) as f1:
            f1.seek(index * 106)
            retrieved_line = f1.readline()
        split_txt = retrieved_line.split(" ")
        # case of corrupted data $dont use this$ 
        if len(split_txt) != 18:
            return torch.zeros(1, self.tensor_length), torch.tensor([[0.00]])
        # get the required base context
        if self.base_context_count == 3:
            base_context = [split_txt[6][2], split_txt[6][3], split_txt[6][4]]
        elif self.base_context_count == 5:
            base_context = [split_txt[6][1], split_txt[6][2], split_txt[6][3], split_txt[6][4], split_txt[6][5]]
        else:
            base_context = [split_txt[6][0], split_txt[6][1], split_txt[6][2], split_txt[6][3], split_txt[6][4], split_txt[6][5], split_txt[6][6]]
        # get the number from the base context
        converted_number = util.convert_bases_to_bits(base_context, self.base_context_count)
        hot_encoded = [0.0] * pow(5, self.base_context_count)
        hot_encoded[converted_number] = 1.0

        # get the read length and position information
        read_position = int(split_txt[4])
        read_len = int(split_txt[5])

        base_in_last_100 = 0.0
        base_in_last_10 = 0.0
        base_in_last_3 = 0.0

        if (read_position >= read_len - 3) or (read_position <= 3):
            base_in_last_3 = 1.0
        if (read_position >= read_len - 10) or (read_position <= 10):
            base_in_last_10 = 1.0
        if (read_position >= read_len - 100) or (read_position <= 100):
            base_in_last_100 = 1.0

        # get quality in float
        quality = float(split_txt[2]) / 100
        # get the num of parallel bases in float
        parallel_vec_s = [split_txt[14], split_txt[15], split_txt[16], split_txt[17]]
        char_remov = ["]", "[", ",", "\n"]
        for char in char_remov:
            for index_s in range(len(parallel_vec_s)):
                temp = parallel_vec_s[index_s].replace(char, "")
                parallel_vec_s[index_s] = temp
        parallel_vec_f = []
        for parallel in parallel_vec_s:
            parallel_vec_f.append(float(parallel))

        # check the state of the poa

        # retrieve sn

        # get the required sn details

        # rearrange so that the calling base num first and rest in decending order
        sorted_vec = self.rearrange_sort_parallel_bases(parallel_vec_f, split_txt[8])

        # make and append to the input tensor,
        input_tensor = torch.tensor([hot_encoded + [base_in_last_100, base_in_last_10, base_in_last_3, quality] + sorted_vec])

        # append to result tensor,
        if split_txt[8] == split_txt[1][1]:
            label_tensor = torch.tensor([[0.00]])
        else:
            label_tensor = torch.tensor([[1.00]])
        return input_tensor, label_tensor

    def clean_string_get_array(self, string):
        # to do
        return
    
    def rearrange_sort_parallel_bases(self, parallel_vec, base):
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
        else:
            selected_base = parallel_vec[0]
            del parallel_vec[0]
        parallel_vec.sort(reverse = True)
        parallel_vec = [selected_base] + parallel_vec
        return parallel_vec