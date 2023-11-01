from multiprocessing import Process, Value, Array
import util

def print_pacbio_scores(read_path, start, end, error_counts, all_counts, thread_index):
    with open(read_path) as f1:
        for index in range(start, end):
            f1.seek(index * 36)
            line = f1.readline()
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            base_quality = int(split_txt[2])
            ref_base = split_txt[1][1]
            call_base = split_txt[3]
            all_counts[(194 * thread_index) + base_quality] += 1
            if ref_base != call_base:
                error_counts[(194 * thread_index) + base_quality] += 1
            if (index - start) % 100001 == 0:
                print("Thread {} Progress {}/{}".format(thread_index, index - start, end - start))
    return

def pipeline_calculate_topology_score_with_probability(read_path, start, end, error_counts, all_counts, thread_index):
    # get the prob list
    mutation_list = util.get_mutation_probablility_array(7)
    with open(read_path) as f1:
        for index in range(start, end):
            f1.seek(index * 60)
            line = f1.readline()
            split_txt = line.split(" ")
            if len(split_txt) != 14:
                continue
            base_context = [split_txt[6][0], split_txt[6][1], split_txt[6][2], split_txt[6][3], split_txt[6][4], split_txt[6][5], split_txt[6][6]]
            call_base = split_txt[8]
            ref_base = split_txt[1][1]
            # get parallel bases in float
            parallel_vec_s = [split_txt[10], split_txt[11], split_txt[12], split_txt[13]]
            char_remov = ["]", "[", ",", "\n"]
            for char in char_remov:
                for index_s in range(len(parallel_vec_s)):
                    temp = parallel_vec_s[index_s].replace(char, "")
                    parallel_vec_s[index_s] = temp
            parallel_vec_f = []
            for parallel in parallel_vec_s:
                parallel_vec_f.append(float(parallel))
            sum = (parallel_vec_f[0] + parallel_vec_f[1] + parallel_vec_f[2] + parallel_vec_f[3])
            recalculated_score = int(util.calculate_topology_score_variable_prob(mutation_list, base_context, call_base, parallel_vec_f[0], parallel_vec_f[1], parallel_vec_f[2], parallel_vec_f[3], sum))
            all_counts[(194 * thread_index) + recalculated_score] += 1
            if ref_base != call_base:
                error_counts[(194 * thread_index) + recalculated_score] += 1
            if (index - start) % 100001 == 0:
                print("Thread {} Progress {}/{}".format(thread_index, index - start, end - start))
                break
    return

# initialize variables
thread_number = 64
error_counts = Array('i', 194 * thread_number)
all_counts = Array('i', 194 * thread_number)
threads = [None] * thread_number
file_path = "/data1/hifi_consensus/all_data/7_base_context/chr1_pos_filtered.txt"
# get the length
total_len = 0
with open(file_path) as f:
    f.seek(0, 2)
    offset = f.tell()
    total_len = int((offset - 60) / 60)
one_thread_allocation = total_len / len(threads)
print(total_len)
for thread_index in range(len(threads)):
    # calculate the start and end of thread
    thread_start = int(one_thread_allocation * thread_index)
    thread_end = int(thread_start + one_thread_allocation)
    # run the thread
    #threads[thread_index] = Process(target=print_pacbio_scores, args=(file_path, thread_start, thread_end, error_counts, all_counts, thread_index))
    threads[thread_index] = Process(target=pipeline_calculate_topology_score_with_probability, args=(file_path, thread_start, thread_end, error_counts, all_counts, 0.85, thread_index))
    #threads[thread_index] = Thread(target=pipeline_calculate_topology_score_with_probability, args=(file_path, thread_start, thread_end, error_counts, all_counts, 0.85, thread_index))
    threads[thread_index].start()

# join the threads
for i in range(len(threads)):
    threads[i].join()

# sum up the result
all_count_final = [0] * 194
error_count_final = [0] * 194
for i in range(thread_number):
    for j in range(194):
        all_count_final[j] += all_counts[(194 * i) + j]
        error_count_final[j] += error_counts[(194 * i) + j]
print (all_count_final)
print (error_count_final)