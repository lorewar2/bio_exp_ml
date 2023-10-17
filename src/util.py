
import torch
import random
import numpy as np
import scipy.special

def identify_error_threebase_context(data_path):
    three_base_context_error_vec = [0] * 256
    with open(data_path, 'r') as hr:
        for index, line in enumerate(hr):
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            ref_base_1 = split_txt[1][0]
            ref_base_2 = split_txt[1][1]
            ref_base_3 = split_txt[1][2]
            call_base = split_txt[3]
            if ref_base_2 != call_base:
                print("before conversion {} {} {}".format(ref_base_1, call_base, ref_base_3))
                converted_number = convert_3_bases_to_64_bit(ref_base_1, call_base, ref_base_3)
                print("converted number {}".format(converted_number))
                converted_base_1, converted_base_2, converted_base_3 = convert_64_bit_to_3_bases(converted_number)
                print("after conversion {} {} {}".format(converted_base_1, converted_base_2, converted_base_3))
                three_base_context_error_vec[converted_number] += 1
    print(three_base_context_error_vec)
    return

def convert_3_bases_to_64_bit(ref_base_1, call_base, ref_base_3):
    ref_int_1 = get_base_to_int(ref_base_1)
    call_int = get_base_to_int(call_base)
    ref_int_3 = get_base_to_int(ref_base_3)
    converted_number = (ref_int_1 + call_int * 4 + ref_int_3 * 16)
    print("before numbers {} {} {}".format(ref_int_1, call_int, ref_int_3))
    return converted_number

def convert_64_bit_to_3_bases(converted_number):
    current_number = converted_number
    ref_int_1 = current_number % 4
    current_number = current_number / 4
    call_int = current_number % 4
    current_number = current_number / 4
    ref_int_3 = current_number % 4
    print("after numbers {} {} {}".format(ref_int_1, call_int, ref_int_3))
    ref_base_1 = get_int_to_base(ref_int_1)
    call_base = get_int_to_base(call_int)
    ref_base_3 = get_int_to_base(ref_int_3)
    return  (ref_base_1, call_base, ref_base_3)

def get_base_to_int(base):
    result = 0
    if base == "A":
        result = 0
    elif base == "C":
        result = 1
    elif base == "G":
        result = 2
    elif base == "T":
        result = 3
    return result

def get_int_to_base(number):
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

def pipeline_calculate_topology_score_with_probability(read_path, prob):
    # arrays to save the result
    error_counts = [0] * 300
    all_counts = [0] * 300
    file = open(read_path, "r")
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        if len(split_txt) != 9:
            continue
        calling_base = split_txt[3]
        ref_base = split_txt[1][1]
        parallel_vec_s = [split_txt[5], split_txt[6], split_txt[7], split_txt[8]]
        char_remov = ["]", "[", ",", "\n"]
        for char in char_remov:
            for index_s in range(len(parallel_vec_s)):
                temp = parallel_vec_s[index_s].replace(char, "")
                parallel_vec_s[index_s] = temp
        parallel_vec_f = []
        for parallel in parallel_vec_s:
            parallel_vec_f.append(float(parallel))
        recalculated_score = int(calculate_topology_score(calling_base, parallel_vec_f[0], parallel_vec_f[1], parallel_vec_f[2], parallel_vec_f[3], (parallel_vec_f[0] + parallel_vec_f[1] + parallel_vec_f[2] + parallel_vec_f[3]), prob))
        all_counts[recalculated_score] += 1
        if ref_base != calling_base:
            error_counts[recalculated_score] += 1
        if index % 100000 == 0:
            print("Running line {}".format(index))
    print(error_counts)
    print(all_counts)
    return

def add_corrected_errors_to_file(read_path, write_path):
    file = open(read_path, "r")
    with open(write_path, 'a') as fw:
        for index, line in enumerate(file):
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            location = split_txt[0].zfill(9)
            three_base = split_txt[1]
            quality = split_txt[2].zfill(2)
            base = split_txt[3]
            count = split_txt[4].zfill(2)
            # get the parallel bases
            parallel_vec_s = [split_txt[5], split_txt[6], split_txt[7], split_txt[8]]
            char_remov = ["]", "[", ",", "\n"]
            for char in char_remov:
                for index_s in range(len(parallel_vec_s)):
                    temp = parallel_vec_s[index_s].replace(char, "")
                    parallel_vec_s[index_s] = temp
            parallel_vec_mod = []
            for parallel in parallel_vec_s:
                parallel_vec_mod.append(parallel.zfill(2))
            modified_line = "{} {} {} {} {} [{} {} {} {}]\n".format(location, three_base, quality, base, count, parallel_vec_mod[0], parallel_vec_mod[1], parallel_vec_mod[2], parallel_vec_mod[3])
            fw.write(modified_line)


def remove_errors_from_file(read_path, write_path):
    # arrays to save the result
    total_error_count = 0
    file = open(read_path, "r")
    modified_lines = []
    with open(write_path, 'a') as fw:
        for index, line in enumerate(file):
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            ref_base = split_txt[1][1]
            call_base = split_txt[3]
            if ref_base == call_base:
                total_error_count += 1
                modified_lines.append(line)
            if index % 100000 == 0:
                for write_line in modified_lines:
                    fw.write(write_line)
                modified_lines.clear()
                print("processed {} records, {}".format(index, total_error_count))
        for write_line in modified_lines:
            fw.write(write_line)
    print(read_path)
    return

def write_errors_to_file(read_path, write_path):
    # arrays to save the result
    total_error_count = 0
    file = open(read_path, "r")
    modified_lines = []
    with open(write_path, 'a') as fw:
        for index, line in enumerate(file):
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            base_quality = int(split_txt[2])
            ref_base = split_txt[1][1]
            call_base = split_txt[3]
            if ref_base != call_base:
                total_error_count += 1
                modified_lines.append(line)
            if index % 100000 == 0:
                for write_line in modified_lines:
                    fw.write(write_line)
                modified_lines.clear()
                print("processed {} records, {}".format(index, total_error_count))
        for write_line in modified_lines:
            fw.write(write_line)
    print(read_path)
    return

def use_himut_file_to_identify_errors(chromosone, data_path, filter_path, write_path):
    # ALL DATA IN ORDER
    # read the himut file put relavant chromosone data in array
    error_locations = []
    path = "{}/himut_data.txt".format(filter_path)
    with open(path, 'r') as hr:
        for index, line in enumerate(hr):
            split_txt = line.split(" ")
            if chromosone == split_txt[0]:
                location = int(split_txt[1])
                ref = split_txt[2]
                alt = split_txt[4].strip()
                error_locations.append((location, ref, alt))
    # read the data file, go line by line
    modified_lines = []
    read_file = open(data_path, 'r')
    himut_index = 0
    with open(write_path, 'a') as fw:
        for index, line in enumerate(read_file):
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            current_location = int(split_txt[0])
            # iterate to correct area of confident region
            while current_location > error_locations[himut_index][0]:
                if himut_index + 1 >= len(error_locations):
                    break
                himut_index += 1
            if current_location != error_locations[himut_index][0]:
                # check if error, if error do not append
                ref_base = split_txt[1][1]
                calling_base = split_txt[3]
                if ref_base == calling_base:
                    modified_lines.append(line)
            else:
                # check if correct error, if not ignore
                ref_base = split_txt[1][1]
                calling_base = split_txt[3]
                if ref_base == error_locations[himut_index][1] and calling_base == error_locations[himut_index][2]:
                    modified_lines.append(line)
                elif ref_base == error_locations[himut_index][1] and calling_base == error_locations[himut_index][1]:
                    modified_lines.append(line)
            if index % 1_000_000 == 0:
                for write_line in modified_lines:
                    fw.write(write_line)
                modified_lines.clear()
                print("processed {} records, {}/{}".format(index, himut_index, len(error_locations)))
    return

def go_through_and_get_high_qual_errors(read_path):
    file = open(read_path, "r")
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        if len(split_txt) != 9:
            continue
        base_quality = int(split_txt[2])
        ref_base = split_txt[1][1]
        call_base = split_txt[3]
        if ref_base != call_base and base_quality >= 93:
            if random.choice(False, False, False, False, True):
                print(line)
        if index % 100000 == 0:
            print("Running line {}".format(index))
            break
    return

def print_pacbio_scores(read_path):
    # arrays to save the result
    error_counts = [0] * 94
    all_counts = [0] * 94
    file = open(read_path, "r")
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        if len(split_txt) != 9:
            continue
        base_quality = int(split_txt[2])
        ref_base = split_txt[1][1]
        call_base = split_txt[3]
        all_counts[base_quality] += 1
        if ref_base != call_base:
            error_counts[base_quality] += 1
        if index % 100000 == 0:
            print("Running line {}".format(index))
            break
    print(error_counts)
    print(all_counts)
    print(read_path)
    return

def filter_data_using_confident_germline_indel_depth(chromosone, data_path, filter_path, write_path):
    # ALL DATA IN ORDER
    # read the confident file put relavant chromosone data in array
    confident_regions = []
    path = "{}/confident_data.txt".format(filter_path)
    with open(path, 'r') as cr:
        for index, line in enumerate(cr):
            split_txt = line.split(" ")
            if chromosone == split_txt[0]:
                start = int(split_txt[1])
                end = int(split_txt[3])
                confident_regions.append((start, end))
    # read germline file put relavant chromosone data in array
    germline_locations = []
    path = "{}/germline_data.txt".format(filter_path)
    with open(path, 'r') as gr:
        for index, line in enumerate(gr):
            split_txt = line.split(" ")
            if chromosone == split_txt[0]:
                location = int(split_txt[1])
                count = len(split_txt[2])
                germline_locations.append((location, count))
    # read indel file put relavant chromosone data in array
    indel_locations = []
    path = "{}/chr2_indel_data.txt".format(filter_path)
    with open(path, 'r') as ir:
        for index, line in enumerate(ir):
            split_txt = line.split(" ")
            if chromosone == split_txt[0]:
                location = int(split_txt[1])
                indel_locations.append(location)
    # read indel file put relavant chromosone data in array
    depth_locations = []
    path = "{}/chr2_depth_data.txt".format(filter_path)
    with open(path, 'r') as dr:
        for index, line in enumerate(dr):
            split_txt = line.split(" ")
            if chromosone == split_txt[0]:
                location = int(split_txt[1])
                depth_locations.append(location)
    # read the data file, go line by line
    modified_lines = []
    read_file = open(data_path, 'r')
    confident_index = 0
    germline_index = 0
    indel_index = 0
    depth_index = 0
    with open(write_path, 'a') as fw:
        for index, line in enumerate(read_file):
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            current_location = int(split_txt[0])
            # iterate to correct area of confident region
            while current_location > confident_regions[confident_index][1]:
                if confident_index + 1 >= len(confident_regions):
                    break
                confident_index += 1
            # iterate to correct area of germline region
            while current_location > germline_locations[germline_index][0]:
                if germline_index + 1 >= len(germline_locations):
                    break
                germline_index += 1
            # iterate to correct area of indel region
            while current_location > indel_locations[indel_index]:
                if indel_index + 1 >= len(germline_locations):
                    break
                indel_index += 1
            # iterate to correct area of depth region
            while current_location > depth_locations[depth_index]:
                if indel_index + 1 >= len(depth_locations):
                    break
                depth_index += 1
            # check if in confident region if not continue
            if (current_location < confident_regions[confident_index][0]) or (current_location > confident_regions[confident_index][1]):
                #print("Not confident region {} start: {} end: {} ".format(current_location, confident_regions[confident_index][0], confident_regions[confident_index][1]))
                continue
            # check if germline variant
            if (current_location >= germline_locations[germline_index][0]) and (current_location <= (germline_locations[germline_index][0] + germline_locations[germline_index][1])):
                #print("Germline variant location {} == {} +- {}".format(current_location, germline_locations[germline_index][0], germline_locations[germline_index][1]))
                continue
            # check if indel or depth location
            if current_location == indel_locations[indel_index] or current_location == depth_locations[depth_index]:
                continue
            # this is run if not filtered
            location = split_txt[0].zfill(9)
            three_base = split_txt[1]
            quality = split_txt[2].zfill(2)
            base = split_txt[3]
            count = split_txt[4].zfill(2)
            # get the parallel bases
            parallel_vec_s = [split_txt[5], split_txt[6], split_txt[7], split_txt[8]]
            char_remov = ["]", "[", ",", "\n"]
            for char in char_remov:
                for index_s in range(len(parallel_vec_s)):
                    temp = parallel_vec_s[index_s].replace(char, "")
                    parallel_vec_s[index_s] = temp
            parallel_vec_mod = []
            for parallel in parallel_vec_s:
                parallel_vec_mod.append(parallel.zfill(2))
            modified_line = "{} {} {} {} {} [{} {} {} {}]\n".format(location, three_base, quality, base, count, parallel_vec_mod[0], parallel_vec_mod[1], parallel_vec_mod[2], parallel_vec_mod[3])
            modified_lines.append(modified_line)
            if index % 1_000_000 == 0:
                for write_line in modified_lines:
                    fw.write(write_line)
                modified_lines.clear()
                print("processed {} records, {}/{}".format(index, germline_index, len(germline_locations)))
    return

def make_sub_array(error_lines, location):
    range = 100
    sub_error_array = []
    closest_error_value = 1000000
    closest_error_index = 0
    for index, error_line in enumerate(error_lines):
        if closest_error_value > abs(error_line[0] - location):
            closest_error_value = abs(error_line[0] - location)
            closest_error_index = index
    if len(error_lines) < closest_error_index + range:
        sub_error_array = error_lines[closest_error_index - range: len(error_lines)]
    elif closest_error_index < range:
        sub_error_array = error_lines[0: closest_error_index + range]
    else:
        sub_error_array = error_lines[closest_error_index - range: closest_error_index + range]
    sub_array_low = sub_error_array[0][0]
    sub_array_high = sub_error_array[len(sub_error_array) - 1][0]
    return sub_error_array, sub_array_low, sub_array_high

def make_unfiltered(read_path, error_path, write_path):
    # error list save
    error_lines = []
    error_count = 0
    last_error_location = 0
    modified_lines = []
    sub_error_array = []
    sub_array_low = -1
    sub_array_high = -1
    # open error file
    error_file = open(error_path, "r")
    for _, line in enumerate(error_file):
        split_txt = line.split(" ")
        location = int(split_txt[0])
        base = split_txt[1][0]
        error_lines.append((location, base))
    # open filtered data file
    read_file = open(read_path, 'r')
    with open(write_path, 'a') as fw:
        for index, line in enumerate(read_file):
            split_txt = line.split(" ")
            if len(split_txt) != 11:
                continue
            location = int(split_txt[0])
            base_context = split_txt[2]
            base_1 = split_txt[3]
            pac_qual = split_txt[4]
            base_2 = split_txt[5]
            total_count = split_txt[6]
            parallel1 = split_txt[7]
            parallel2 = split_txt[8]
            parallel3 = split_txt[9]
            parallel4 = split_txt[10]
            if not (location >= sub_array_low and location <= sub_array_high):
                sub_array, sub_array_low, sub_array_high = make_sub_array(error_lines, location)
            try:
                required_index = [y[0] for y in sub_array].index(location)
                if base_1 == sub_array[required_index][1]:
                    #if location == 37666995:
                    #    print("{} {}".format(base_1, sub_array[required_index]))
                    result = "true"
                    if last_error_location != location:
                        last_error_location = location
                        error_count += 1
                else:
                    result = "false"
            except ValueError:
                result = "false"
            modified_lines.append("{} {} {} {} {} {} {} {} {} {} {}".format(location, result, base_context, base_1, pac_qual, base_2, total_count, parallel1, parallel2, parallel3, parallel4))
            if index % 1_000_000 == 0:
                for write_line in modified_lines:
                    fw.write(write_line)
                modified_lines.clear()
                print("processed {} records, errors {}/{}".format(index, error_count, len(error_lines)))
    return

def old_format_to_new_format_converter(read_path, write_path):
    # array to save offsets
    modified_lines = []
    # indices to output
    write_index = 1
    read_index = 1
    # open files read and write files
    file = open(read_path, "r")
    with open(write_path, 'a') as fw:
        for index, line in enumerate(file):
            split_txt = line.split(" ")
            if len(split_txt) != 10:
                continue
            location = split_txt[0]
            base_context = split_txt[1]
            pac_qual = split_txt[2]
            base_1 = split_txt[3]
            total_count = split_txt[5]
            parallel1 = split_txt[6]
            parallel2 = split_txt[7]
            parallel3 = split_txt[8]
            parallel4 = split_txt[9]
            modified_lines.append("{} {} {} {} {} {} {} {} {}".format(location, base_context, pac_qual, base_1, total_count, parallel1, parallel2, parallel3, parallel4))
            if index % 1_000_000 == 0:
                for write_line in modified_lines:
                    fw.write(write_line)
                modified_lines.clear()
                print("indexed {} records".format(index))
    return

def check_and_clean_data (path):
    other_bigger_count = 0
    # open the file with ml data
    file = open(path, "r")
    # go line by line
    for index, line in enumerate(file):
        if index % 100000 == 0:
            print("current line {} other bigger count {}".format(index, other_bigger_count))
        split_txt = line.split(" ")
        if len(split_txt) != 11:
            print("line number {} is invalid, line: {}".format(index, line))
            continue
        #encoded_bases = self.one_hot_encoding_bases(split_txt[1][0]) + self.one_hot_encoding_bases(split_txt[1][1]) + self.one_hot_encoding_bases(split_txt[1][2])
        three_context_bases = [split_txt[2][0], split_txt[2][1], split_txt[2][2]]
        # get quality in float
        quality = float(split_txt[4]) / 100
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
        sorted_vec = rearrange_sort_parallel_bases(parallel_vec_f, split_txt[2][1])
        if sorted_vec[1] > sorted_vec[0] and split_txt[1] == "false":
            #print("line number {} is invalid, true with parallel higher line: {}".format(index, line))
            other_bigger_count += 1
    return

def rearrange_sort_parallel_bases(parallel_vec, base):
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

def list_corrected_errors_rust_input(read_path):
    result_array = np.zeros((4, 4, 16))
    # open the file with ml data
    file = open(read_path, "r")
    # go line by line
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        ref_base = get_base_to_int(split_txt[0][1])
        alt_base = get_base_to_int(split_txt[2][0])
        first_in_three_base = get_base_to_int(split_txt[0][0])
        third_in_three_base = get_base_to_int(split_txt[0][2])
        one_three_value = first_in_three_base * 4 + third_in_three_base
        result_array[ref_base][alt_base][one_three_value] += 1
    print(result_array)
    return

def list_corrected_errors(read_path, write_path):
    # open the file with ml data
    file = open(read_path, "r")
    # go line by line
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        # check if line is valid
        if len(split_txt) != 12:
            continue
        result = split_txt[1]
        # only check the false ones
        if result == 'false':
            parallel_vec_s = [split_txt[8], split_txt[9], split_txt[10], split_txt[11]]
            char_remov = ["]", "[", ",", "\n"]
            for char in char_remov:
                for index_s in range(len(parallel_vec_s)):
                    temp = parallel_vec_s[index_s].replace(char, "")
                    parallel_vec_s[index_s] = temp
            parallel_vec_f = []
            for parallel in parallel_vec_s:
                parallel_vec_f.append(float(parallel))
            # save required data
            calling_base = split_txt[2][1]
            calling_base_seq_num = 0
            top_base = "A"
            top_base_seq_num = max(parallel_vec_f)
            max_index = parallel_vec_f.index(max(parallel_vec_f))
            if max_index == 0:
                top_base = "A"
            elif max_index == 1:
                top_base = "C"
            elif max_index == 2:
                top_base = "G"
            elif max_index == 3:
                top_base = "T"

            if calling_base == "A":
                calling_base_seq_num = parallel_vec_f[0]
            elif calling_base == "C":
                calling_base_seq_num = parallel_vec_f[1]
            elif calling_base == "G":
                calling_base_seq_num = parallel_vec_f[2]
            elif calling_base == "T":
                calling_base_seq_num = parallel_vec_f[3]
            # if calling base is not equal to top base, error corrected? need reference to check
            if calling_base != top_base:
                print("well that was a waste of time")
    return

def index_file(read_path, write_path):
    # array to save offsets
    read_line_offset = [0]
    # indices to output
    write_index = 1
    read_index = 1
    # open files read and write files
    with open(read_path) as fr:
        with open(write_path, 'a') as fw:
            # go line by line in read
            read_line = fr.readline()
            while read_line:
                # get the current offset and add to array
                read_line_offset.append(fr.tell())
                # every million save array to file and clear array
                if read_index % 1_000_000 == 0:
                    for offset in read_line_offset:
                        write_line = "{:021d}\n".format(offset)
                        fw.write(write_line)
                        write_index += 1
                    read_line_offset.clear()
                    print("indexed {} records".format(read_index))
                read_index += 1
                read_line = fr.readline()
    return

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
    else:
        correct_rate = np.exp(ln_prob_data_given_A + ln_prob_base_A - ln_sum_of_probabilities)

    error_rate = 1.0 - correct_rate
    quality_score = (-10.00) * np.log10(error_rate + 0.000000000000000000001)
    #print(quality_score)
    return quality_score

def calculate_binomial(n, k, prob):
    binomial_coeff = scipy.special.binom(n, k)
    success = np.power(prob, k)
    failure = np.power(1.00 - prob, n - k)
    return (binomial_coeff * success * failure)

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