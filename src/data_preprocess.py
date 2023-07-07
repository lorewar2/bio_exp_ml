def get_train_data():
    file = open("data/test_file.txt", "r")
    for line in file:
        if line != "\n":
            split_txt = line.split(" ")
            #print(split_txt)
            result = split_txt[0]
            base_0 = split_txt[1][0]
            base_1 = split_txt[1][1]
            base_2 = split_txt[1][2]
            quality = split_txt[3]
            char_remov = ["]", "[", ",", "\n"]
            parallel_0 = split_txt[6]
            parallel_1 = split_txt[7]
            parallel_2 = split_txt[8]
            parallel_3 = split_txt[9]
            for char in char_remov:
                parallel_0 = parallel_0.replace(char, "")
                parallel_1 = parallel_1.replace(char, "")
                parallel_2 = parallel_2.replace(char, "")
                parallel_3 = parallel_3.replace(char, "")
            print(result, base_0, base_1, base_2, quality, parallel_0, parallel_1, parallel_2, parallel_3)
    file.close()
    return (result, base_0, base_1, base_2, quality, parallel_0, parallel_1, parallel_2, parallel_3)