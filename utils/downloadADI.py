from urllib.request import urlretrieve

def download_files(file_path, dir_path):
    files = open(file_path)
    files = files.readlines()
    for f in files:
        name = dir_path+f.replace('\n','').split('/')[-1]
        urlretrieve(f, name)


