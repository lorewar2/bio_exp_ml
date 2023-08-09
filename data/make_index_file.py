# make index file
read_path = "./train_file.txt"
write_path = "./train_file.idx"

# read the offsets of the file lines
read_line_offset = [0]
index = 1
with open(read_path) as fr:
    with open(write_path) as fw:
        read_line = fr.readline()
        while read_line:
            if read_line == "\n":
                read_line_offset.append(fr.tell())
                if index % 1_000_000 == 0:
                    for offset in read_line_offset:
                        write_line = "{:021d} {:021d}\n".format(index, offset)
                        fw.write(write_line)
                        index += 1
                    read_line_offset.clear()
                    print("indexed {} records".format(index))
                index += 1
            read_line = fr.readline()
