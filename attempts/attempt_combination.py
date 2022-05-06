# fine-grained combination of different embeddings

from utils import *
import torch
import numpy as np
import argparse, os


# get entity/attribute/value lists
def get_lists(*lists):
    entity, attribute, value = [], [], []
    for tlist in lists:
        for item in tlist:
            entity.append(item[0])
            attribute.append(item[1])
            value.append(item[2])
    entity = list(set(entity))
    attribute = list(set(attribute))
    value = list(set(value))
    return entity, attribute, value


# get regression model
def get_model(modeltype):
    from sklearn import linear_model, neural_network
    if modeltype == 'linear':
        model = linear_model.LinearRegression()
    elif modeltype == 'ridge':
        model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    elif modeltype == 'lasso':
        model = linear_model.LassoCV(alphas=[0.1, 1.0, 10.0])
    elif modeltype == 'mlp':
        model = neural_network.MLPRegressor(random_state=1, max_iter=500)
    return model


# get saved plm embeddings
def get_plm_emb(entity, args):
    plm_dir = './results/' + args.data + '/regression/' + args.plm + '/'
    if args.desc is not None:
        plm_dir += args.desc + '_'
    plm_emb = np.load(plm_dir + 'emb.npy')
    plm_e2id = {}
    cnt = 0
    with open(plm_dir + 'entitys.txt', 'r') as f:
        for line in f:
            plm_e2id[line[:-1]] = cnt
            cnt += 1
    print(len(plm_e2id))

    emb = [plm_emb[plm_e2id[e]] for e in entity]
    emb = np.array(emb)
    return emb


# get saved kge embeddings
def get_kge_emb(entity, args):
    kge_dir = './pretrainedModels/' + args.data if args.kge == 'complex' else './pretrainedModels/FB15K/'
    kge_model = torch.load(kge_dir + '/' + args.kge + '.pt', map_location=torch.device('cpu'))
    kge_emb = kge_model['model'][0]['_entity_embedder.embeddings.weight'] if '_entity_embedder.embeddings.weight' \
                                                                             in kge_model['model'][0].keys() else \
        kge_model['model'][0]['_entity_embedder._embeddings.weight']
    kge_e2id = {}
    with open(kge_dir + '/entity_ids.del', 'r') as f:
        for line in f:
            tris = line[:-1].split()
            kge_e2id[tris[1]] = int(tris[0])
    print(len(kge_e2id))
    kge_dim = kge_emb.shape[1]
    print(kge_dim)  # 128 for transe

    emb = [np.random.random_sample(kge_dim) if e not in kge_e2id else kge_emb[kge_e2id[e]].numpy() for e in entity]
    emb = np.array(emb)
    return emb


# combine different embeddings
def comb_emb(plm_emb, kge_emb, comb='concat'):
    emb = []
    if comb == 'concat':  # plm+kge
        emb = np.hstack((plm_emb, kge_emb))
    elif comb == 'r-concat':  # kge+plm
        emb = np.hstack((kge_emb, plm_emb))
    elif 'quantile' in comb:
        # quantile-normal; quantile-uniform
        from sklearn.preprocessing import QuantileTransformer
        emb = np.hstack((plm_emb, kge_emb))
        distribution = comb.split('-')[1]
        transformer = QuantileTransformer(output_distribution=distribution, random_state=99)  # uniform or normal
        emb = transformer.fit_transform(emb)
    elif 'power' in comb:
        # power-yeo;power-box
        from sklearn.preprocessing import PowerTransformer
        emb = np.hstack((plm_emb, kge_emb))
        method = comb.split('-')[1]
        method = 'yeo-johnson' if method == 'yeo' else 'box-cox'
        transformer = PowerTransformer(method=method)
        emb = transformer.fit_transform(emb)
    return emb


def main():
    print('hello, combination')
    # 0. parse args
    parser = argparse.ArgumentParser(description='embedding + regression')
    parser.add_argument('--data', type=str, default='FB15K', choices=['YAGO15K', 'FB15K', 'DB15K'], help='used dataset')
    parser.add_argument('--plm', default='bert-base-uncased', help='type of plm')
    parser.add_argument('--desc', type=str, default='mid2description', help='type of description')
    parser.add_argument('--kge', default='transe', help='type of kge')
    parser.add_argument('--comb', default='concat',
                        choices=['concat', 'r-concat', 'quantile-normal', 'quantile-uniform', 'power-yeo', 'power-box'],
                        help='type of combination')
    args = parser.parse_args()
    print(args)

    save_dir = './results/' + args.data + '/combination/' + args.plm + '+' + args.kge + '/' + args.comb + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.desc is not None:
        save_dir += args.desc + '_'
    args.save_dir = save_dir

    # 1. get data
    train, test, valid = get_data(args.data)
    print(len(train), len(test), len(valid))
    entity, attribute, value = get_lists(train, test, valid)
    print(len(entity), len(attribute), len(value))
    e2id = dict(zip(entity, range(len(entity))))
    a2id = dict(zip(attribute, range(len(attribute))))
    print(len(e2id), len(a2id))

    # 2. get embed
    plm_emb = get_plm_emb(entity, args)
    kge_emb = get_kge_emb(entity, args)
    print(plm_emb.shape, kge_emb.shape)  # (12493,768),(12493,128)
    entityEmb = comb_emb(plm_emb, kge_emb, args.comb)
    print(entityEmb.shape)

    # attr_of_int = literal_triples.attribute.unique().tolist()
    attr_of_int = ['wasBornOnDate', 'wasCreatedOnDate', 'wasDestroyedOnDate', 'diedOnDate', 'happenedOnDate',
                   'hasLatitude', 'hasLongitude'] if args.data == 'YAGO15K' \
        else ['people.person.date_of_birth', 'film.film.initial_release_date', 'organization.organization.date_founded',
              'location.dated_location.date_founded', 'people.deceased_person.date_of_death',
              'people.person.weight_kg', 'people.person.height_meters', 'location.geocode.latitude',
              'location.geocode.longitude', 'location.location.area', 'topic_server.population_number']

    attr_trainX = {k: [] for k in attr_of_int}
    attr_trainY = {k: [] for k in attr_of_int}
    attr_validX = {k: [] for k in attr_of_int}
    attr_validY = {k: [] for k in attr_of_int}
    attr_testX = {k: [] for k in attr_of_int}
    attr_testY = {k: [] for k in attr_of_int}

    for s, p, o in train:
        if p in attr_of_int:
            attr_trainX[p].append(entityEmb[e2id[s]])
            attr_trainY[p].append(o)
    for s, p, o in valid:
        if p in attr_of_int:
            attr_validX[p].append(entityEmb[e2id[s]])
            attr_validY[p].append(o)
    for s, p, o in test:
        if p in attr_of_int:
            attr_testX[p].append(entityEmb[e2id[s]])
            attr_testY[p].append(o)
    # print(len(attr_trainX['people.deceased_person.date_of_death']),len(attr_trainY['people.deceased_person.date_of_death']))
    # print(len(attr_trainX['time.event.start_date']), len(attr_trainY['time.event.start_date']))

    # 3. trainging & model selection
    attr_valid_result = {k: {} for k in attr_of_int}
    for attr in attr_of_int:
        for m in ['linear', 'ridge', 'lasso']:
            model = get_model(m)
            model.fit(attr_trainX[attr], attr_trainY[attr])
            # print(model.coef_)
            pred = model.predict(attr_validX[attr])
            result = get_performance(attr_validY[attr], pred)
            print(m, result)
    
            if len(attr_valid_result[attr]) == 0 or attr_valid_result[attr]['mae'] > result['mae']:
                attr_valid_result[attr] = result
                attr_valid_result[attr]['model'] = model
    
        print(attr, attr_valid_result[attr])

    # 4. evaluate on test set
    attr_test_result = {k: {} for k in attr_of_int}
    for attr in attr_of_int:
        # scaler = StandardScaler()
        # scaler.fit(attr_trainX[attr])
        # attr_trainX[attr] = scaler.transform(attr_trainX[attr])
        # attr_testX[attr] = scaler.transform(attr_testX[attr])

        estimator.fit(attr_trainX[attr], attr_trainY[attr])
        print(estimator.best_params_)
        pred = estimator.predict(attr_testX[attr])
        res = get_performance(attr_testY[attr], pred)
        attr_test_result[attr] = res
        print(attr, res)
        

    
    # 5. get total result
    total_result = get_total_result(attr_of_int, attr_test_result)
    print(attr_test_result)
    print(total_result)

    # 6. save to file
    save_to_file(save_dir + 'attr_valid_result.json', attr_valid_result)
    save_to_file(save_dir + 'attr_test_result.json', attr_test_result)
    save_to_file(save_dir + 'total_result.json', total_result)

    print('finish')


if __name__ == '__main__':
    main()
