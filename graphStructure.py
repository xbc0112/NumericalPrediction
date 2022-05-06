# Three graph-based methods：GLOBAL, LOCAL, MRAP
# Refer to the implementation of MrAP from https://github.com/bayrameda/MrAP

import argparse
import torch
import pandas as pd
import numpy as np
import os
from utils import *

def run_MrAP(data):
    from MrAP.utils import extract_edges_YAGO, extract_edges_FB, estimate_params, drop_sym, reduce_to_singles
    from MrAP.Models.MrAP import MrAP
    from MrAP.Models.algs import Global, Local, iter_MrAP

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dir = './results/' + data + '/graphStructure/'

    train, test, valid = get_pd_data(data)
    #literal_triples = pd.concat([train, test, valid], ignore_index=True)
    literal_triples = pd.concat([train, test], ignore_index=True)
    literal_triples.set_axis(['node','attribute','numeric'], axis=1, inplace=True)
    relation_triples = pd.read_table('./data/'+data+'/EntityTriples_handled.txt', sep=' ', header=None) # 0FB15K_
    #relation_triples = pd.read_table('./data/'+data+'/0train_relation.txt', sep=' ', header=None)
    relation_triples.set_axis(['node_1','relation','node_2'], axis=1, inplace=True)

    # attributes of interesting
    attr_of_group = [['wasBornOnDate','wasCreatedOnDate','wasDestroyedOnDate','diedOnDate','happenedOnDate'],['hasLatitude'],['hasLongitude']] if data == 'YAGO15K' \
        else [['people.person.date_of_birth','film.film.initial_release_date','organization.organization.date_founded','location.dated_location.date_founded','people.deceased_person.date_of_death'],
          ['people.person.weight_kg', 'people.person.height_meters'],
          ['location.geocode.latitude'],
          ['location.geocode.longitude'],
          ['location.location.area', 'topic_server.population_number']]
    attr_of_int = [a for group in attr_of_group for a in group]

    edge_list = []
    relations = []
    for group in attr_of_group:
        #print('group: ',group)
        literal_of_int = literal_triples[literal_triples.attribute.isin(group)]
        edge_of_int, relation_of_int = extract_edges_YAGO(relation_triples, literal_of_int) \
            if data == 'YAGO15K' else extract_edges_FB(relation_triples, literal_of_int)
        edge_list += edge_of_int
        relations += relation_of_int
    print('-----')

    asym_edge_list = drop_sym(edge_list)

    x = literal_triples.numeric.values.copy()
    u = np.array([1] * len(train) + [0] * len(test), dtype=bool)
    taus, omegas, _, _ = estimate_params(edge_list, x)

    x_0 = torch.tensor(x, device=device)
    u_0 = torch.tensor(u, device=device)
    x_0[u_0 == 0] = 0 
    attrs = literal_triples.attribute.values

    x = torch.tensor(x, device=device)

    def get_model(method):
        if method == 'MrAP_cross':
            model = MrAP(device=device, edge_list=asym_edge_list, omega=omegas, tau=taus)
        else:
            edge_list_singles, relations_singles, attribute_coupled = reduce_to_singles(edge_list, attrs)
            asym_edge_list_singles = drop_sym(edge_list_singles)
            tau_singles = taus[relations_singles]  
            omega_singles = omegas[relations_singles]
            model = MrAP(device=device, edge_list=asym_edge_list_singles, omega=omega_singles, tau=tau_singles)
        return model

    for method in ['global','local','MrAP_single','MrAP_cross']: 
        if method == 'global':
            pred = Global(x_0,u_0,attrs)
        elif method == 'local':
            pred = Local(asym_edge_list,x_0,u_0,attrs)
        else:
            model = get_model(method)
            pred = iter_MrAP(x_0,u_0,model,xi=0.5,entity_labels=attrs)

        attr_test_result = {}
        for attr in attr_of_int:
            idx = torch.tensor(attrs == attr, device=device) & (u_0 == 0)
            if len(x[idx]) == 0: 
                result = {}
            else:
                result = get_performance(x[idx], pred[idx])
            attr_test_result[attr] = result
            print(attr, result)
        #print(method, attr_test_result)
        total_result = get_total_result(attr_of_int, attr_test_result)
        print(method, total_result)

        save_dir = dir + '/' + method
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_to_file(save_dir+'/attr_test_result.json', attr_test_result)
        save_to_file(save_dir+'/total_result.json', total_result)

    return

def main():
    print('hello graphStructure')
    parser = argparse.ArgumentParser(description='graphStructure: global/local/MrAP')
    parser.add_argument('--data', type=str, default='YAGO15K', choices=['YAGO15K', 'FB15K'], help='used dataset')
    args = parser.parse_args()
    print(args)

    run_MrAP(args.data)

    print('finish')

if __name__ == '__main__':
    main()