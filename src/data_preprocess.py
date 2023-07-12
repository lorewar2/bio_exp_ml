import torch

def get_data(path, start, length, get_errors):
    file = open(path, "r")
    label_tensor = torch.empty((length, 1), dtype = torch.float32)
    input_tensor = torch.empty((length, 17), dtype = torch.float32)
    index = 0
    tensor_pos = 0
    for line in file:
        # only get the specified section in file
        if index < start:
            index += 1
            continue
        elif get_errors:
            if tensor_pos >= length:
                break
        elif index > start and index >= start + length:
            break
        if line != "\n":
            split_txt = line.split(" ")
            if len(split_txt) != 10:
                index += 1
                continue
            # get three base context in float
            base_0 = get_corrosponding_float_for_base_character(split_txt[1][0])
            base_1 = get_corrosponding_float_for_base_character(split_txt[1][1])
            base_2 = get_corrosponding_float_for_base_character(split_txt[1][2])
            # get quality in float
            quality = float(split_txt[3]) / 100
            # get the num of parallel bases in float
            parallel_0 = split_txt[6]
            parallel_1 = split_txt[7]
            parallel_2 = split_txt[8]
            parallel_3 = split_txt[9]
            char_remov = ["]", "[", ",", "\n"]
            for char in char_remov:
                parallel_0 = parallel_0.replace(char, "")
                parallel_1 = parallel_1.replace(char, "")
                parallel_2 = parallel_2.replace(char, "")
                parallel_3 = parallel_3.replace(char, "")
            parallel_0 = float(parallel_0)
            parallel_1 = float(parallel_1)
            parallel_2 = float(parallel_2)
            parallel_3 = float(parallel_3)
            #print(result, base_0, base_1, base_2, quality, parallel_0, parallel_1, parallel_2, parallel_3)
            # append to result tensor,
            result = split_txt[0] 
            if result == "false":
                if get_errors:
                    index += 1
                    continue
                label_tensor[tensor_pos] = torch.tensor([[1.0]])
                # continue if we want errors and its not a error
            else:
                label_tensor[tensor_pos] = torch.tensor([[0.00]])
            # make and append to the input tensor,
            sort_vec = [parallel_0, parallel_1, parallel_2, parallel_3]
            if base_1 <= 0.0:
                selected_base = sort_vec[0]
                del sort_vec[0]
            elif base_1 <= 0.334:
                selected_base = sort_vec[1]
                del sort_vec[1]
            elif base_1 <= 0.67:
                selected_base = sort_vec[2]
                del sort_vec[2]
            elif base_1 <= 1.01:
                selected_base = sort_vec[3]
                del sort_vec[3]
            sort_vec.sort(reverse=True)
            sort_vec = [selected_base] + sort_vec
            encoded_base_0 = one_hot_encoding_bases(base_0)
            encoded_base_1 = one_hot_encoding_bases(base_1)
            encoded_base_2 = one_hot_encoding_bases(base_2)
            input_tensor[tensor_pos] = torch.tensor([encoded_base_0 + encoded_base_1 + encoded_base_2 + [quality] + sort_vec])
            if index % 10000 == 0 and get_errors == False:
                print("line index: {}".format(index))
            elif tensor_pos % 100 == 0 and get_errors == True:
                print("line index: {} errors found: {}".format(index, tensor_pos))
            index += 1
            tensor_pos += 1
    file.close()
    return (input_tensor, label_tensor)

def one_hot_encoding_bases(base):
    one_hot_base = [0.0]*4
    if base <= 0.0:
        one_hot_base[0] = 1.0
    elif base <= 0.334:
        one_hot_base[1] = 1.0
    elif base <= 0.67:
        one_hot_base[2] = 1.0
    elif base <= 1.01:
        one_hot_base[3] = 1.0
    return one_hot_base

def get_corrosponding_float_for_base_character(character):
    return_value = 0.0
    if character == "A":
        return_value = 0.0
    elif character == "C":
        return_value = 0.33
    elif character == "G":
        return_value = 0.66
    elif character == "T":
        return_value = 1.0
    return return_value