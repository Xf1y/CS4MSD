def xie(str, path):
    file = open(path, "a", encoding='utf-8')
    print(str)
    file.writelines(str + "\n")
    file.close()