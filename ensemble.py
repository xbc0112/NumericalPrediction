# combination (ensemble learning)

import argparse, os, json
import numpy as np
from modules import regModule, mlmModule, graphModule
from utils import get_performance, get_total_result, save_to_file

def main():
    # chosen base models: reg（transe、bert、bert-desc）、mlm、graph
    print('hello, ensemble')
    # 0. parse args
    parser = argparse.ArgumentParser(description='ensemble')
    parser.add_argument('--data', type=str, default='FB15K', choices=['YAGO15K', 'FB15K'], help='used dataset')
    parser.add_argument('--models', nargs='+', help='model list, eg, --models mlm transe', required=True)
    #parser.add_argument('--mode', default='best', choices=['best', 'mean', 'median'], help='integration mode')
    args = parser.parse_args()
    print(args)

    save_dir = './results/' + args.data + '/ensemble/' + '+'.join(args.models) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attr_of_int = ['wasBornOnDate', 'wasCreatedOnDate', 'wasDestroyedOnDate', 'diedOnDate', 'happenedOnDate',
                   'hasLatitude', 'hasLongitude'] if args.data == 'YAGO15K' \
        else ['people.person.date_of_birth', 'film.film.initial_release_date', 'organization.organization.date_founded',
              'location.dated_location.date_founded', 'people.deceased_person.date_of_death',
              'people.person.weight_kg', 'people.person.height_meters', 'location.geocode.latitude',
              'location.geocode.longitude', 'location.location.area', 'topic_server.population_number']

    attr_valid_ground = {}
    attr_test_ground = {}
    # get ground truth
    dir = './data/' + args.data + '/'
    with open(dir + '/valid_mlm.json', 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    with open(dir + '/test_mlm.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    for k in attr_of_int:
        attr_valid_ground[k] = np.array(valid_data[k]['ground'])
        attr_test_ground[k] = np.array(test_data[k]['ground'])

    attr_valid_pred = {k:{} for k in attr_of_int}
    attr_test_pred = {k:{} for k in attr_of_int}
    # get predictions
    for model in args.models:
        if model == 'mlm':
            valid,test = mlmModule(args.data)
        elif model == 'graph':
            valid,test = graphModule(args.data)
        else:
            valid,test = regModule(args.data, model)
        for k in attr_of_int:
            attr_valid_pred[k][model] = valid[k]
            attr_test_pred[k][model] = test[k]

    attr_test_pred_concat = {k:{} for k in attr_of_int}
    # concat
    for attr in attr_of_int:
        pred = attr_test_pred[attr][args.models[0]].reshape(-1,1)
        for model in args.models[1:]:
            pred = np.hstack((pred, attr_test_pred[attr][model].reshape(-1,1)))
        attr_test_pred_concat[attr] = pred

    attr_test_result = {k: {} for k in attr_of_int}
    # mean
    for attr in attr_of_int:
        pred = np.mean(attr_test_pred_concat[attr],axis=1) 
        result = get_performance(attr_test_ground[attr], pred)
        attr_test_result[attr] = result
    total_result = get_total_result(attr_of_int, attr_test_result)
    print('mean', total_result)

    save_to_file(save_dir + 'mean_attr_test_result.json', attr_test_result)
    save_to_file(save_dir + 'mean_total_result.json', total_result)

    # median
    for attr in attr_of_int:
        pred = np.median(attr_test_pred_concat[attr],axis=1) 
        result = get_performance(attr_test_ground[attr], pred)
        attr_test_result[attr] = result
    total_result = get_total_result(attr_of_int, attr_test_result)
    print('median', total_result)

    save_to_file(save_dir + 'median_attr_test_result.json', attr_test_result)
    save_to_file(save_dir + 'median_total_result.json', total_result)

    # best (choose from validation set)
    attr_valid_result = {k: {} for k in attr_of_int}
    for attr in attr_of_int: 
        for model in args.models:
            result = get_performance(attr_valid_ground[attr], attr_valid_pred[attr][model])
            attr_valid_result[attr][model] = result
    # measured on the MAE metric
    for attr in attr_of_int:
        best_model = args.models[0]
        best_result = attr_valid_result[attr][best_model]
        for model in args.models:
            if attr_valid_result[attr][model]['mae'] < best_result['mae']:
                best_model = model
                best_result = attr_valid_result[attr][model]
        result = get_performance(attr_test_ground[attr], attr_test_pred[attr][best_model])
        attr_test_result[attr] = result
        attr_test_result[attr]['model'] = best_model
    total_result = get_total_result(attr_of_int, attr_test_result)
    print('best', total_result)

    save_to_file(save_dir + 'best_attr_test_result.json', attr_test_result)
    save_to_file(save_dir + 'best_total_result.json', total_result)

    print('finish')


if __name__ == '__main__':
    main()