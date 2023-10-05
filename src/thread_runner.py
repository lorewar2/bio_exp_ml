from threading import Thread

def print_pacbio_scores(read_path, start, end, error_counts, all_counts, thread_index):
    with open(read_path) as f1:
        for index in range(start, end):
            f1.seek(index * 36)
            line = f1.readline()
            print(line)
            split_txt = line.split(" ")
            if len(split_txt) != 9:
                continue
            base_quality = int(split_txt[2])
            ref_base = split_txt[1][1]
            call_base = split_txt[3]
            all_counts[thread_index][base_quality] += 1
            if ref_base != call_base:
                error_counts[thread_index][base_quality] += 1
            if index % 100001 == 0:
                print("Thread {} Progress {}/{}".format(thread_index, index - start, end - start))
                break
    return

thread_number = 64
error_counts = [[0 for i in range(94)] for j in range(thread_number)]
all_counts = [[0 for i in range(94)] for j in range(thread_number)]
threads = [None] * thread_number
file_path = "/data1/hifi_consensus/all_data/chr2_filtered.txt"
# get the length
total_len = 0
with open(file_path) as f:
    f.seek(0, 2)
    offset = f.tell()
    total_len = int((offset - 36) / 36)
one_thread_allocation = total_len / len(threads)
print(total_len)
for thread_index in range(len(threads)):
    # calculate the start and end of thread
    thread_start = one_thread_allocation * thread_index
    thread_end = thread_start + one_thread_allocation
    threads[thread_index] = Thread(target=print_pacbio_scores, args=(file_path, thread_start, thread_end, error_counts, all_counts, thread_index))
    threads[thread_index].start()

# join the threads
for i in range(len(threads)):
    threads[i].join()

print (" ".join(all_counts))
print (" ".join(error_counts))
