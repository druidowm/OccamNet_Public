with open("eplex5.txt") as f:
    for line in f:
        line = line.replace("ARG0","x")
        line = line.replace("ARG1","y")
        line = line.replace("ARG2","1")
        line = line.replace("ARG3","2")
        print(line)