import os

order = ["1027_ESL",
         "1028_SWD",
         "1029_LEV",
         "1030_ERA",
         "1089_USCrime",
         "1096_FacultySalaries",
         "192_vineyard",
         "195_auto_price",
         "207_autoPrice",
         "210_cloud",
         "228_elusage",
         "229_pwLinear",
         "230_machine_cpu",
         "4544_GeographicalOriginalofMusic",
         "485_analcatdata_vehicle"]

def readFile(fileName):
    names = []
    train = []
    val = []
    test = []
    time = []
    with open(fileName, "r") as file:
        for line in file.readlines():
            if not "Dataset" in line:
                data = line.split(",\t\t")
                print(data)
                names.append(data[0])
                train.append(data[1])
                try:
                    float(data[4])
                    val.append(data[4])
                    test.append(data[7])
                    time.append(data[8][:-1])
                except:
                    val.append(data[3])
                    test.append(data[5])
                    time.append(data[6][:-1])
    
    return (names, train, val, test, time)

def readDirectory(directory):
    files = os.listdir(directory)
    names = []
    train = []
    val = []
    test = []
    time = []
    for file in files:
        n,tr,v,te,ti = readFile(f"{directory}/{file}")
        names += n
        train += tr
        val += v
        test += te
        time += ti

    name = directory.split('/')[-1]

    return (name, names, train, val, test, time)

def readData(directory):
    folders = [file.path for file in os.scandir(directory) if file.is_dir()]

    names = []
    train = []
    val = []
    test = []
    time = []
    methods = []
    for folder in folders:
        method,n,tr,v,te,ti = readDirectory(folder)
        methods.append(method)
        names.append(n)
        train.append(tr)
        val.append(v)
        test.append(te)
        time.append(ti)

    reorder = []
    for i in range(len(names)):
        reorder.append([])
        for name in order:
            found = False
            for j in range(len(names[i])):
                print(f"Name1: {name}")
                print(f"Name2: {names[i][j]}")
                if names[i][j] == name:
                    reorder[i].append(j)
                    found = True
            if not found:
                reorder[i].append(-1)

    #print(names)

    names = [item+[""] for item in names]
    train = [item+[""] for item in train]
    val = [item+[""] for item in val]
    test = [item+[""] for item in test]
    time = [item+[""] for item in time]

    #print(names)
    print(reorder)

    names = [[names[i][index] for index in reorder[i]] for i in range(len(names))]
    train = [[train[i][index] for index in reorder[i]] for i in range(len(train))]
    val = [[val[i][index] for index in reorder[i]] for i in range(len(val))]
    test = [[test[i][index] for index in reorder[i]] for i in range(len(test))]
    time = [[time[i][index] for index in reorder[i]] for i in range(len(time))]

    print(time)

    csv = "train\r"
    csv += "name,"
    for method in methods:
        csv += f"{method},"
    csv += "\r"

    for i in range(len(names[0])):
        csv += f"{names[0][i]},"
        for j in range(len(methods)):
            csv += f"{train[j][i]},"
        csv += "\r"

    csv += "val\r"
    csv += "name,"
    for method in methods:
        csv += f"{method},"
    csv += "\r"

    for i in range(len(names[0])):
        csv += f"{names[0][i]},"
        for j in range(len(methods)):
            csv += f"{val[j][i]},"
        csv += "\r"

    csv += "test\r"
    csv += "name,"
    for method in methods:
        csv += f"{method},"
    csv += "\r"

    for i in range(len(names[0])):
        csv += f"{names[0][i]},"
        for j in range(len(methods)):
            csv += f"{test[j][i]},"
        csv += "\r"

    csv += "time\r"
    csv += "name,"
    for method in methods:
        csv += f"{method},"
    csv += "\r"

    for i in range(len(names[0])):
        csv += f"{names[0][i]},"
        for j in range(len(methods)):
            csv += f"{time[j][i]},"
        csv += "\r"

    return csv

with open("/Users/dugan/Desktop/TrainTestValResults/CSVData.csv","w") as file:
    file.write(readData("/Users/dugan/Desktop/TrainTestValResults"))