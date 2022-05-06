# KGE-reg & PLM-reg

import argparse
import numpy as np
import torch
import json, os
from utils import *


# get entity/attribute/numeric lists
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


# id to entity names
def id2name(entity):
    mid2name = {}  
    with open('./helpers/FB15K_mid2name.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split()
            mid2name[tris[0]] = tris[1]
    # print(len(mid2name))

    ename = []
    for id in entity:
        ename.append(id if id not in mid2name else mid2name[id])
    return ename


# get multilingual embeddings
def get_ML_emb(entity, ML):
    print('hello,', ML)
    c2lan = {'E': 'English', 'F': 'French', 'G': 'German'}
    lan_emb = {}
    lan_e2id = {}
    dname = './results/FB15K/regression/bert-base-uncased/Description_'
    for lan in ['English', 'German', 'French']:
        lan_emb[lan] = np.load(dname + lan + '_emb.npy')
        cnt = 0
        e2id = {}
        with open(dname + lan + '_entitys.txt', 'r') as f:
            for line in f:
                e2id[line[:-1]] = cnt
                cnt += 1
        lan_e2id[lan] = e2id

    emb = [[] for e in entity]
    e2id = {e: entity.index(e) for e in entity}
    for c in ML:
        lan = c2lan[c]
        for e in entity:
            emb[e2id[e]].extend(lan_emb[lan][lan_e2id[lan][e]])
    emb = np.array(emb)
    return emb


# get entity embeddings
def get_emb(entity, args, dim=50):
    emb = []
    embed = args.embed
    # several random methods from numpy
    if embed == 'random':  
        emb = np.random.random([len(entity), dim])
    elif embed == 'uniform':  
        emb = np.random.uniform(0, 1, [len(entity), dim])
    elif embed == 'randn':  
        emb = np.random.randn(len(entity), dim)
    elif embed == 'sample':  
        emb = np.random.random_sample([len(entity), dim])
    # combine the embeddings of KGE and PLM
    elif '+' in embed:
        plm_dir = args.save_dir.split('+')[0] + '/'
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

        kge_type = embed.split('+')[1]

        kge_dir = './pretrainedModels/' + args.data if kge_type == 'complex' else './pretrainedModels/FB15K/'
        kge_model = torch.load(kge_dir + '/' + kge_type + '.pt', map_location=torch.device('cpu'))
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

        emb = []
        for e in entity:
            emb1 = plm_emb[plm_e2id[e]]
            emb2 = np.random.random_sample(kge_dim) if e not in kge_e2id else kge_emb[kge_e2id[e]].numpy()
            emb.append(np.hstack((emb1, emb2)))
        emb = np.array(emb)
    # PLM embeddings
    elif 'bert' in embed or 'checkpoint' in embed:
        if os.path.exists(args.save_dir + 'emb.npy'):
            # load saved embeddings
            ent_emb = np.load(args.save_dir + 'emb.npy')
            e2id = {}
            cnt = 0
            with open(args.save_dir + 'entitys.txt', 'r') as f:
                for line in f:
                    e2id[line[:-1]] = cnt
                    cnt += 1
            print(len(e2id))
            for e in entity:
                emb.append(ent_emb[e2id[e]])
            emb = np.array(emb)
        elif args.desc in ['EG', 'EF', 'FG', 'EFG']:
            # multilingual combinations
            emb = get_ML_emb(entity, args.desc)
        else:
            from transformers import AutoTokenizer, AutoModel
            try:
                tokenizer = AutoTokenizer.from_pretrained(embed)
            except:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained(embed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # using entity names or descriptions
            if args.desc is not None:
                entity_desc = {}
                with open('./helpers/' + args.data + '_' + args.desc + '.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        tris = line[:-1].split('\t', 1)
                        entity_desc[tris[0]] = tris[1].replace('"', '')

                desc = []
                for e in entity:
                    # If no description texts exist, fill with the entity name instead.
                    desc.append(entity_desc[e] if e in entity_desc else id2name([e])[0])
                print(len(desc))  # 15081
                inputs = tokenizer(desc, padding='max_length', truncation=True, return_tensors='pt')
            else:  
                ename = id2name(entity)
                inputs = tokenizer(ename, padding=True, truncation=True, return_tensors='pt')

            emb = []
            batch_size = 4  # 1024  # 128
            for i in range(0, len(entity), batch_size):
                output = model(inputs['input_ids'][i:i + batch_size].to(device)).last_hidden_state
                emb.extend(torch.mean(output, dim=1).detach().cpu().numpy())
            emb = np.array(emb)
            np.save(args.save_dir + 'emb.npy', emb)
            with open(args.save_dir + 'entitys.txt', 'w', encoding='utf-8') as f:
                for e in entity:
                    f.write(e + '\n')
            print('embed ok')
    # KGE embeddings
    else:  
        # mainly from FB15K, and YAGO15K are mapped by the SameAs links
        dir = './pretrainedModels/' + args.data if embed == 'complex' else './pretrainedModels/FB15K/'
        model = torch.load(dir + '/' + embed + '.pt', map_location=torch.device('cpu'))
        ent_emb = model['model'][0]['_entity_embedder.embeddings.weight'] if '_entity_embedder.embeddings.weight' \
                                                                             in model['model'][0].keys() else \
            model['model'][0]['_entity_embedder._embeddings.weight']
        e2id = {}
        with open(dir + '/entity_ids.del', 'r') as f:
            for line in f:
                tris = line[:-1].split()
                e2id[tris[1]] = int(tris[0])
        print(len(e2id))
        dim = ent_emb.shape[1]
        cnt = 0
        for e in entity:
            if e not in e2id:
                # print(e)
                cnt += 1
                emb.append(np.random.random_sample(dim))
            else:
                emb.append(ent_emb[e2id[e]].numpy())
        emb = np.array(emb)
        print(f"{cnt} entities don't have a pretrained embed.")
    
    return emb


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


def main():
    print('hello')
    # 0. parse args
    parser = argparse.ArgumentParser(description='embedding + regression')
    parser.add_argument('--data', type=str, default='FB15K', choices=['YAGO15K', 'FB15K'], help='used dataset')
    parser.add_argument('--embed', default='randn', help='type of embedding, like randn, bert, transe or transe+bert')
    # parser.add_argument('--use_description', type=bool, default=False, help='use description rather than entity name in plm')
    parser.add_argument('--desc', type=str, default=None, help='type of description')
    # parser.add_argument('--model', default='linear',choices=['linear','ridge'],help='type of regression model')
    args = parser.parse_args()
    print(args)

    save_dir = './results/' + args.data + '/regression/' + args.embed + '/'
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
    entityEmb = get_emb(entity, args)
    print(entityEmb.shape)
    # print(entityEmb)

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

    # 3. training & model selection
    attr_valid_result = {k: {} for k in attr_of_int}
    for attr in attr_of_int:
        for m in ['linear', 'ridge', 'lasso']:
            model = get_model(m)
            model.fit(attr_trainX[attr], attr_trainY[attr])
            # print(model.coef_)
            pred = model.predict(attr_validX[attr])
            result = get_performance(attr_validY[attr], pred)
            print(m,result)

            if len(attr_valid_result[attr]) == 0 or attr_valid_result[attr]['mae'] > result['mae']:
                attr_valid_result[attr] = result
                attr_valid_result[attr]['model'] = model

        print(attr, attr_valid_result[attr])

    # 4. evaluating on the test set
    attr_test_result = {k: {} for k in attr_of_int}
    for attr in attr_of_int:
        model = attr_valid_result[attr]['model']
        pred = model.predict(attr_testX[attr])
        attr_test_result[attr] = get_performance(attr_testY[attr], pred)

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
