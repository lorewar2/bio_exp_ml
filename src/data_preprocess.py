import torch

def get_data(path, start, len):
    file = open(path, "r")
    label_tensor = torch.empty((len, 1), dtype = torch.float32)
    input_tensor = torch.empty((len, 8), dtype = torch.float32)
    index = 0
    tensor_pos = 0
    for line in file:
        # only get the specified section in file
        if index < start:
            index += 1
            continue
        elif index > start and index >= start + len:
            break
        if line != "\n":
            split_txt = line.split(" ")
            # get three base context in float
            base_0 = get_corrosponding_float_for_base_character(split_txt[1][0])
            base_1 = get_corrosponding_float_for_base_character(split_txt[1][1])
            base_2 = get_corrosponding_float_for_base_character(split_txt[1][2])
            # get quality in float
            quality = float(split_txt[3])
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
                label_tensor[tensor_pos] = torch.tensor([[0.99]])
            else:
                label_tensor[tensor_pos] = torch.tensor([[0.00]])
            # make and append to the input tensor,
            input_tensor[tensor_pos] = torch.tensor([[base_0, base_1, base_2, quality, parallel_0, parallel_1, parallel_2, parallel_3]])
            if index % 10000 == 0:
                print(index)
            index += 1
            tensor_pos += 1
    file.close()
    return (input_tensor, label_tensor)

def get_corrosponding_float_for_base_character(character):
    if character == "A":
        return_value = 0.0
    elif character == "C":
        return_value = 0.33
    elif character == "G":
        return_value = 0.66
    elif character == "T":
        return_value = 1.0
    return return_value