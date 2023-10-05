def print_pacbio_scores(read_path, start, end, size, error_counts, all_counts, thread_index):
    # find the length of the file
    with open(read_path) as f1:
        for index in range(0, 2):
            f1.seek(index * 35)
            line = f1.readline()
            print(line)
    # file = open(read_path, "r")
    # for index, line in enumerate(file):
    #     split_txt = line.split(" ")
    #     if len(split_txt) != 9:
    #         continue
    #     base_quality = int(split_txt[2])
    #     ref_base = split_txt[1][1]
    #     call_base = split_txt[3]
    #     all_counts[thread_index][base_quality] += 1
    #     if ref_base != call_base:
    #         error_counts[thread_index][base_quality] += 1
    #     if index % 100001 == 0:
    #         print("Running line {}".format(index))
    #         break
    # print(error_counts)
    # print(all_counts)
    # print(read_path)
    return

from threading import Thread
error_counts = [[0] * 94] * 64
all_counts = [[0] * 94] * 64
threads = [None] * 64
results = [None] * 10
# get the length
start = 0
end = 100
size = 100
print_pacbio_scores("/data1/hifi_consensus/all_data/chr2_filtered.txt", start, end, size, error_counts, all_counts, 1)
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
