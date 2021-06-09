import pickle
import os.path

def make_dir(path):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

def easy_saver(sid, table, cache_dir, data):
    make_dir(cache_dir)
    path = os.path.join(cache_dir, f"{table}_{sid}.pickle")
    with open(path, mode="wb+") as file:
        pickle.dump(data,file)

def easy_loader(sid, table, cache_dir):
    make_dir(cache_dir)
    path = os.path.join(cache_dir, f"{table}_{sid}.pickle")
    with open(path, mode="rb") as file:
        return pickle.load(file)
    
def easy_checker(sid, table, cache_dir):
    try:
        easy_loader(sid, table, cache_dir)
    except Exception as er:
        print(er)
        return False
    return True