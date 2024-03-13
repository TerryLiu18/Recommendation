import os 
import json
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
from datetime import datetime
import torch 
from haversine import haversine

def haversine_numpy(source_lat, source_lon, target_lat, target_lon):
    assert source_lat.shape == source_lon.shape == target_lat.shape == target_lon.shape
    R = 6373.0
    
    source_lat = np.radians(source_lat)
    source_lon = np.radians(source_lon)
    target_lat = np.radians(target_lat)
    target_lon = np.radians(target_lon)
    
    dlat = target_lat - source_lat
    dlon = target_lon - source_lon
    
    a = np.sin(dlat / 2) ** 2 + np.cos(source_lat) * np.cos(target_lat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# create folder if not exist
if not os.path.exists('aux_data'):
    os.makedirs('aux_data')

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    assert isinstance(d, dict), f"{d} with type: {type(d)} should be a list"
    # d is a dictionary
    # json_path is a string
    # open json_path in write mode
    with open(json_path, 'w') as f:
        json.dump(d, f)
        
def save_list_to_json(l, json_path):
    """Saves list of floats in json file
    Args:
        l: (list) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    assert isinstance(l, list), f"{l} with type: {type(l)} should be a list"
    # l is a list
    # json_path is a string
    # open json_path in write mode
    with open(json_path, 'w') as f:
        json.dump(l, f)


def read_dict_from_json(json_path, key_as_int=True):
    """Reads dict of floats from json file
    Args:
        json_path: (string) path to json file
    Returns:
        d: (dict) of float-castable values (np.float, int, float, etc.)
    """
    # json_path is a string
    # open json_path in read mode
    with open(json_path, 'r') as f:
        # load json_path
        d = json.load(f)
    if key_as_int:
        d = {int(k): v for k, v in d.items()}
    return d

def read_list_from_json(json_path):
    """Reads list of floats from json file
    Args:
        json_path: (string) path to json file
    Returns:
        l: (list) of float-castable values (np.float, int, float, etc.)
    """
    # json_path is a string
    # open json_path in read mode
    with open(json_path, 'r') as f:
        # load json_path
        l = json.load(f)
    return l

def parse_to_days_since(date_string, reference_date_string='2023-01-01'):
    # Parse the datetime string and remove timezone information
    
    # todo: this is a adhoc hotfix and should be remove later...
    if isinstance(date_string, int):
        return date_string
    elif not isinstance(date_string, str):
        raise TypeError(f"{date_string} should be a string or int")
    
    date = datetime.fromisoformat(date_string)
    date = date.replace(tzinfo=None)  # Make the date timezone-naive
    reference_date = datetime.fromisoformat(reference_date_string)

    days_since = (date - reference_date).days
    return days_since

def read_raw_data_new(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    for col in df.columns:
        if col.endswith('_list') and isinstance(df[col][0], str):
            df[col] = df[col].apply(lambda x:
                [item.strip() for item in x.lstrip('[').rstrip(']').split(',')])
        elif col.endswith('_list') and isinstance(df[col][0], int):
            df[col] = df[col].apply(lambda x:
                [int(item.strip()) for item in x.lstrip('[').rstrip(']').split(',')])
    return df

def read_raw_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    for col in df.columns:
        if col.endswith('_list') and isinstance(df[col][0], str):
            df[col] = df[col].apply(literal_eval)
            # df[col] = df[col].apply(lambda x:
                # [item.strip() for item in x.lstrip('[').rstrip(']').split(',')])
    return df

def haversine_tensor(source_lat, source_lon, target_lat, target_lon):
    assert source_lat.shape == source_lon.shape == target_lat.shape == target_lon.shape
    R = 6373.0
    
    source_lat = source_lat * torch.pi / 180.0
    source_lon = torch.deg2rad(source_lon)
    target_lat = torch.deg2rad(target_lat)
    target_lon = torch.deg2rad(target_lon)
    
    d = torch.sin((target_lat - source_lat) / 2) ** 2 + torch.cos(source_lat) * torch.cos(target_lat) * torch.sin((target_lon - source_lon) / 2) ** 2
    return 2 * R * torch.arcsin(torch.sqrt(d))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    def remove_repeats(arr):
        return [arr[i] for i in range(len(arr)) if i == 0 or (arr[i] != arr[i - 1]) and arr[i] > 3]
    
    train_data = read_raw_data('aux_data/category_seq.csv')
    train_data['visit_category_id_list'] = train_data['visit_category_id_list'].apply(remove_repeats)
    train_data = train_data[train_data['visit_category_id_list'].apply(len) > 2]
    train_data[['visit_category_id_list']].to_csv('aux_data/category_seq_trimmed.csv', index=False)
