# make index file
read_path = "./train_file.txt"
write_path = "./train_file.idx"

# read the offsets of the file lines
line_offset = [0]
index = 1
with open(read_path) as f:
    line = f.readline()
    while line:
        if line == "\n":
            line_offset.append(f.tell())
            if index % 1_000_000 == 0:
                print(index)
            index += 1
        line = f.readline()

# write the offsets of the file lines in index
index = 0
with open(write_path, "a") as f:
    for offset in line_offset:
        write_line = "{:021d} {:021d}\n".format(index, offset)
        f.write(write_line)
        index += 1
        if index % 1_000_000 == 0:
            print(index)
