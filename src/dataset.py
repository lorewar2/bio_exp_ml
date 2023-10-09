import torch
import os

class QualityDataset(torch.utils.data.Dataset):
    def __init__(self, file_loc, load_all):
        self.file_loc = file_loc
        self.load_all = load_all
        # get len and save it
        with open(file_loc) as f:
            f.seek(0, 2)
            offset = f.tell()
            self.len = int((offset - 36) / 36) - 1
        # load all data
        if load_all:
            self.label_tensor = torch.empty((self.len, 1), dtype = torch.float32)
            self.input_tensor = torch.empty((self.len, 69), dtype = torch.float32)
            for index in range(0, self.len):
                temp_input_tensor, temp_label_tensor = self.retrieve_item_from_disk(index)
                self.input_tensor[index] = temp_input_tensor
                self.label_tensor[index] = temp_label_tensor

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.load_all == True:
            input_tensor = self.input_tensor[index]
            label_tensor = self.label_tensor[index]
        else:
            input_tensor, label_tensor = self.retrieve_item_from_disk(index)
        return input_tensor, label_tensor

    def retrieve_item_from_disk(self, index):
        # search the index file to file the location # index offset is 22
        location = 0
        retrieved_line = ""
        with open(self.file_loc) as f1:
            f1.seek(index * 36)
            retrieved_line = f1.readline()
        split_txt = retrieved_line.split(" ")
        # case of corrupted data $dont use this$ 
        if len(split_txt) != 9:
            return torch.zeros(1, 69), torch.tensor([[0.00]])
        # get three base context in one hot encoded
        encoded_bases = self.one_hot_encoding_64bit(split_txt[1][0], split_txt[3], split_txt[1][2])
        # get quality in float
        quality = float(split_txt[2]) / 100
        # get the num of parallel bases in float
        parallel_vec_s = [split_txt[5], split_txt[6], split_txt[7], split_txt[8]]
        char_remov = ["]", "[", ",", "\n"]
        for char in char_remov:
            for index_s in range(len(parallel_vec_s)):
                temp = parallel_vec_s[index_s].replace(char, "")
                parallel_vec_s[index_s] = temp
        parallel_vec_f = []
        for parallel in parallel_vec_s:
            parallel_vec_f.append(float(parallel))
        # rearrange so that the calling base num first and rest in decending order
        sorted_vec = self.rearrange_sort_parallel_bases(parallel_vec_f, split_txt[3])
        # make and append to the input tensor,
        input_tensor = torch.tensor([encoded_bases + [quality] + sorted_vec])
        # append to result tensor,
        if split_txt[3] == split_txt[1][1]:
            label_tensor = torch.tensor([[0.00]])
            # continue if we want errors and its not a error
        else:
            label_tensor = torch.tensor([[1.00]])
        return input_tensor, label_tensor

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

    def one_hot_encoding_bases(self, base):
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

    def get_base_to_int(self, base):
        if base == "A":
            base_int = 0
        elif base == "C":
            base_int = 1
        elif base == "G":
            base_int = 2
        elif base == "T":
            base_int = 3
        else:
            base_int = 0
        return base_int

    def one_hot_encoding_64bit(self, base0, base1, base2):
        encoded_bases = [0.0] * 64
        bases = [self.get_base_to_int(base0), self.get_base_to_int(base1), self.get_base_to_int(base2)]
        result = 0
        for idx, base in enumerate(bases):
            multiplier = idx * 4
            result += base * multiplier
        encoded_bases[int(result)] = 1.0
        return encoded_bases