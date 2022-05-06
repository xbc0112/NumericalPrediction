# auxiliary functions for all methods

import numpy as np
import pandas as pd
import json


# get train/valid/test data
def get_data(data):
    dir = "./data/" + data
    train, test, valid = [], [], []
    with open(dir + '/train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            train.append(line[:-1].split(' '))
    with open(dir + '/test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            test.append(line[:-1].split(' '))
    with open(dir + '/valid.txt', 'r', encoding='utf-8') as f:
        for line in f:
            valid.append(line[:-1].split(' '))
    # numerical values to float
    for tlist in [train, test, valid]:
        for item in tlist:
            item[2] = float(item[2])
    return train, test, valid


# get data by pandas
def get_pd_data(data):
    dir = "./data/" + data
    train = pd.read_table(dir + '/train.txt', sep=' ', header=None)
    test = pd.read_table(dir + '/test.txt', sep=' ', header=None) 
    valid = pd.read_table(dir + '/valid.txt', sep=' ', header=None)
    return train, test, valid


# get mlm data
def get_mlm_data(args):
    if args.prompt is None:
        file_name = './data/' + args.data + '/test_mlm.json'
    else:
        file_name = './data/' + args.data + '/test_prompt_' + args.prompt + '.json'
    with open(file_name, 'r', encoding='utf-8') as file:
        attr_dic = json.load(file)
    return attr_dic


def is_number(num):
    import re
    # pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    pattern = re.compile(r'^[0-9]+([.]{1}[0-9]+){0,1}$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False

# get performance, including mae, mse, rmse, r2
def get_performance(ground, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import torch
    if torch.is_tensor(ground):
        ground = ground.cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    mae = mean_absolute_error(ground, pred)
    mse = mean_squared_error(ground, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground, pred)
    num = len(ground)
    result = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'num': num}
    return result


# get total performance of micro and macro
def get_total_result(attr_of_int, attr_test_result):
    total_result = {'micro': {}, 'macro': {}}
    # metric_dic = attr_test_result[attr_of_int[0]].keys()
    metric_dic = ['mae', 'mse', 'rmse', 'r2', 'num']
    for t in ['micro', 'macro']:
        for metric in metric_dic:
            if metric == 'rmse' or metric == 'num':
                continue
            sum = divisor = 0
            for attr in attr_of_int:
                if attr in ['location.location.area','topic_server.population_number']:
                    # skip the two FB15K attributes
                    continue
                if attr not in attr_test_result or metric not in attr_test_result[attr]:
                    continue
                factor = 1 if t == 'macro' else attr_test_result[attr]['num']
                divisor += factor
                sum += attr_test_result[attr][metric] * factor
            total_result[t][metric] = sum / divisor
        total_result[t]['rmse'] = np.sqrt(total_result[t]['mse'])
    return total_result


# check diversity for MLM
def check_diversity(pred):
    nums = np.unique(pred)
    num_dic = {k: np.sum(pred == k) for k in nums}
    sorted_dic = sorted(num_dic.items(), key=lambda kv: (kv[1], kv[0]))
    print(f'{nums.shape[0]} different prediction nums of the {pred.shape[0]} samples')
    print(f'the max num is {sorted_dic[-1]} and the min num is {sorted_dic[0]}')
    return num_dic


def check_total_diversity(attr_diverse_result):
    total_dic = {}
    for attr in attr_diverse_result:
        dic = attr_diverse_result[attr]
        for k, v in dic.items():
            if k in total_dic:
                total_dic[k] += v
            else:
                total_dic[k] = v
    return total_dic


# save to file
def save_to_file(save_file, result_dic):
    with open(save_file, 'w') as f:
        f.write(json.dumps(result_dic, default=str))
