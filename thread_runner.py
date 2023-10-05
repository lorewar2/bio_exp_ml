def print_pacbio_scores(read_path, start, end, error_count, all_count, index):
    # find the length of the file

    # arrays to save the result
    error_counts = [0] * 93
    all_counts = [0] * 93
    file = open(read_path, "r")
    for index, line in enumerate(file):
        split_txt = line.split(" ")
        if len(split_txt) != 11:
            continue
        result = split_txt[1]
        position = int(split_txt[4])
        all_counts[position] += 1
        if result == "true":
            error_counts[position] += 1
        if index % 100000 == 0:
            print("Running line {}".format(index))
    print(error_counts)
    print(all_counts)
    print(read_path)
    return

from threading import Thread
error_count = [[]] * 64
all_count = [[]] * 64
threads = [None] * 64
results = [None] * 10
# get the length

for i in range(len(threads)):
    # calculate the start and end of thread

    threads[i] = Thread(target=print_pacbio_scores, args=("/data1/hifi_consensus/all_data/chr2.txt", error_count, all_count, i))
    threads[i].start()

# do some other stuff

for i in range(len(threads)):
    threads[i].join()

print (" ".join(results))  # what sound does a metasyntactic locomotive make?
