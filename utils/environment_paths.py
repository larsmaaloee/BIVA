from os.path import join, exists
from os import makedirs
import datetime
import time

def check_dir(path):
    if not exists(path):
        makedirs(path)
    return path

def get_data_path(name):
    primary_root_path = "" # TODO : ADD THIS ROOT PATH IF THE DATA IS TO BE LOCATED ELSEWHERE.
    root_path = "datasets"
    if exists(primary_root_path):
        root_path = join(primary_root_path, root_path)
    path = join(root_path, name)
    return check_dir(path)

def get_model_path(name):
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')
    path = join("output", "{}_{}".format(name, time_stamp))
    return check_dir(path)
