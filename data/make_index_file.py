# make index file
read_path = "./train_file.txt"
write_path = "./train_file.idx"

# # read the offsets of the file lines
# line_offset = [0]
# index = 1
# with open(read_path) as f:
#     line = f.readline()
#     while line:
#         if line == "\n":
#             line_offset.append(f.tell())
#             if index % 1_000_000 == 0:
#                 print(index)
#             index += 1
#         line = f.readline()

# # write the offsets of the file lines in index
# index = 0
# with open(write_path, "a") as f:
#     for offset in line_offset:
#         write_line = "{:021d} {:021d}\n".format(index, offset)
#         f.write(write_line)
#         index += 1
#         if index % 1_000_000 == 0:
#             print(index)

# search the index file to file the location # index offset is 44
required_index = 77
location = 43

with open(write_path) as f1:
    f1.seek(required_index * 44)
    line = f1.readline()
    line_split = line.split(" ")
    #print(line_split[1])
    location = int(line_split[1])
        
# test
with open(read_path) as f2:
    f2.seek(location)
    line2 = f2.readline()
    print(line2)