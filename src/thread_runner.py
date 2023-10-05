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
            if index % 100000 == 0:
                print("Running line {}".format(index))
                break
    return

error_counts = [[0] * 94] * 64
all_counts = [[0] * 94] * 64
threads = [None] * 64
results = [None] * 10
file_path = "/data1/hifi_consensus/all_data/chr2_filtered.txt"
# get the length
total_len = 0
with open(file_path) as f:
    f.seek(0, 2)
    offset = f.tell()
    total_len = int((offset - 36) / 36)
one_thread_allocation = total_len / len(threads)
start = 0
end = total_len
print(total_len)
print_pacbio_scores(file_path, start, end, error_counts, all_counts, 1)
print(error_counts)
print(all_counts)
for i in range(len(threads)):
    # calculate the start and end of thread
    print("test")
    #threads[i] = Thread(target=print_pacbio_scores, args=("/data1/hifi_consensus/all_data/chr2.txt", error_count, all_count, i))
    #threads[i].start()

# do some other stuff

for i in range(len(threads)):
    threads[i].join()

print (" ".join(results))
