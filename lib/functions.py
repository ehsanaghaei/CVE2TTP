def read_txt(fname, to_list=False):
    with open(fname, "r") as f:
        txt = f.readline()
    return txt
