import torch
import os

class QualityDataset(torch.utils.data.Dataset):
    def __init__(self, file_loc, index_loc):
        self.file_loc = file_loc
        self.index_loc = index_loc
        # get len and save it
        with open(self.index_loc) as f:
            f.seek(0, 2)
            offset = f.tell()
            self.len = int((offset - 44) / 44)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        # search the index file to file the location # index offset is 44
        location = 0
        retrieved_line = ""
        with open(self.index_loc) as f1:
            f1.seek(index * 44)
            line = f1.readline()
            line_split = line.split(" ")
            location = int(line_split[1])
        with open(self.file_loc) as f2:
            f2.seek(location)
            retrieved_line = f2.readline()
            #print(retrieved_line)
        # process the retrieved line
        split_txt = retrieved_line.split(" ")
        # get three base context in one hot encoded
        encoded_bases = self.one_hot_encoding_bases(split_txt[1][0]) + self.one_hot_encoding_bases(split_txt[1][1]) + self.one_hot_encoding_bases(split_txt[1][2])
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
        sorted_vec = self.rearrange_sort_parallel_bases(parallel_vec_f, split_txt[1][1])
        # make and append to the input tensor,
        input_tensor = torch.tensor([encoded_bases + [quality] + sorted_vec])
        # append to result tensor,
        result = split_txt[0]
        if result == "false":
            label_tensor = torch.tensor([[1.0]])
            # continue if we want errors and its not a error
        else:
            label_tensor = torch.tensor([[0.00]])
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