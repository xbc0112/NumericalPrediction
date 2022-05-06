# auxiliary modules for ensemble.py

from utils import *
import numpy as np
import torch
import json

def get_reg_emb(entity, data, embed, dim = 50):
    emb = []
    if 'bert' in embed:
        # bert-base-uncased as the default
        dir = './results/'+data+'/regression/bert-base-uncased/'
        if 'desc' in embed:
            dir += 'mid2description_'
        ent_emb = np.load(dir+'emb.npy')
        e2id = {}
        cnt = 0
        with open(dir+'entitys.txt','r') as f:
            for line in f:
                e2id[line[:-1]] = cnt
                cnt += 1
        for e in entity:
            emb.append(ent_emb[e2id[e]])
        emb = np.array(emb)
    else: # transe as the default
        dir = './pretrainedModels/FB15K/'
        model = torch.load(dir + 'transe.pt', map_location=torch.device('cpu'))
        ent_emb = model['model'][0]['_entity_embedder.embeddings.weight']
        e2id = {}
        with open(dir + '/entity_ids.del', 'r') as f:
            for line in f:
                tris = line[:-1].split()
                e2id[tris[1]] = int(tris[0])
        dim = ent_emb.shape[1]
        for e in entity:
            emb.append(ent_emb[e2id[e]].numpy() if e in e2id else np.random.random_sample(dim))
        emb = np.array(emb)
    return emb


def regModule(data='FB15K', embed='transe'):
    # data choices: FB15K\YAGO15K
    # embed choices: transe\bert\bert-desc
    from regression import get_lists, get_model
    train, test, valid = get_data(data)
    entity, attribute, value = get_lists(train, test, valid)
    e2id = dict(zip(entity, range(len(entity))))
    a2id = dict(zip(attribute, range(len(attribute))))
    entityEmb = get_reg_emb(entity, data, embed)
    print(entityEmb.shape)

    attr_of_int = ['wasBornOnDate', 'wasCreatedOnDate', 'wasDestroyedOnDate', 'diedOnDate', 'happenedOnDate',
                   'hasLatitude', 'hasLongitude'] if data == 'YAGO15K' \
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

    attr_valid_result = {k: {} for k in attr_of_int} 
    attr_valid_pred = {k: {} for k in attr_of_int} 
    for attr in attr_of_int:
        for m in ['linear', 'ridge', 'lasso']:
            model = get_model(m)
            model.fit(attr_trainX[attr], attr_trainY[attr])
            pred = model.predict(attr_validX[attr])
            result = get_performance(attr_validY[attr], pred)

            if len(attr_valid_result[attr]) == 0 or attr_valid_result[attr]['mae'] > result['mae']:
                attr_valid_result[attr] = result
                attr_valid_result[attr]['model'] = model
                attr_valid_pred[attr] = np.array(pred)
        #print(attr, attr_valid_pred[attr])

    attr_test_pred = {k: {} for k in attr_of_int} 
    for attr in attr_of_int:
        model = attr_valid_result[attr]['model']
        pred = model.predict(attr_testX[attr])
        attr_test_pred[attr] = np.array(pred)
        #print(attr, pred.shape)
    print('regModule ok')

    return attr_valid_pred, attr_test_pred


def mlmModule(data='FB15K'):
    # data choices: FB15K\YAGO15K
    dir = './data/' + data + '/'
    with open(dir+'/valid_mlm.json','r',encoding='utf-8') as f:
        valid_data = json.load(f)
    with open(dir+'/test_mlm.json','r',encoding='utf-8') as f:
        test_data = json.load(f)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    ckpt = './pretrainedModels/'+data+'/finetune_3e-5/bert-base-uncased/checkpoint-' + ('5830' if data == 'FB15K' else '5890')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(ckpt)
    model.to(device)

    numdic = {k: tokenizer.vocab[k] for k in tokenizer.vocab if is_number(k)}
    chosen = list(numdic.values())
    masked = torch.ones(len(tokenizer.vocab), dtype=bool)
    masked[chosen] = False  

    attr_valid_pred = {k: {} for k in valid_data}
    for attr in valid_data:
        text = valid_data[attr]['text']
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)

        pred = []
        batch_size = 64  # 256 
        for i in range(0, len(text), batch_size):
            input = inputs[i:i + batch_size]
            token_logits = model(input).logits
            mask_token_index = torch.where(input == tokenizer.mask_token_id)

            for k, v in zip(mask_token_index[0], mask_token_index[1]):
                mask_logits = token_logits[k, v, :]
                probability = torch.nn.functional.softmax(mask_logits, dim=-1)
                probability[masked] = 0  
                pred.append(float(tokenizer.decode(torch.argmax(probability))))
        attr_valid_pred[attr] = np.array(pred)
    attr_test_pred = {k: {} for k in test_data}
    for attr in test_data:
        text = test_data[attr]['text']
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)

        pred = []
        batch_size = 256  # 4
        for i in range(0, len(text), batch_size):
            input = inputs[i:i + batch_size]
            token_logits = model(input).logits
            mask_token_index = torch.where(input == tokenizer.mask_token_id)

            for k, v in zip(mask_token_index[0], mask_token_index[1]):
                mask_logits = token_logits[k, v, :]
                probability = torch.nn.functional.softmax(mask_logits, dim=-1)
                probability[masked] = 0  
                pred.append(float(tokenizer.decode(torch.argmax(probability))))
        attr_test_pred[attr] = np.array(pred)
        #print(attr, len(pred))

    print('mlmModule ok')
    return attr_valid_pred, attr_test_pred


def graphModule(data='FB15K'):
    # data choices: FB15K\YAGO15K
    from MrAP.utils import extract_edges_YAGO, extract_edges_FB, estimate_params, drop_sym, reduce_to_singles
    from MrAP.Models.MrAP import MrAP
    from MrAP.Models.algs import iter_MrAP

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train, test, valid = get_pd_data(data)
    # concat
    literal_triples = pd.concat([train, valid, test], ignore_index=True)
    literal_triples.set_axis(['node', 'attribute', 'numeric'], axis=1, inplace=True)
    relation_triples = pd.read_table('./data/' + data + '/EntityTriples_handled.txt', sep=' ', header=None)  
    relation_triples.set_axis(['node_1', 'relation', 'node_2'], axis=1, inplace=True)

    attr_of_group = [['wasBornOnDate', 'wasCreatedOnDate', 'wasDestroyedOnDate', 'diedOnDate', 'happenedOnDate'],
                     ['hasLatitude'], ['hasLongitude']] if data == 'YAGO15K' \
        else [
        ['people.person.date_of_birth', 'film.film.initial_release_date', 'organization.organization.date_founded',
         'location.dated_location.date_founded', 'people.deceased_person.date_of_death'],
        ['people.person.weight_kg', 'people.person.height_meters'],
        ['location.geocode.latitude'],
        ['location.geocode.longitude'],
        ['location.location.area', 'topic_server.population_number']]
    attr_of_int = [a for group in attr_of_group for a in group]

    edge_list = []
    relations = []
    for group in attr_of_group:
        literal_of_int = literal_triples[literal_triples.attribute.isin(group)]
        edge_of_int, relation_of_int = extract_edges_YAGO(relation_triples, literal_of_int) \
            if data == 'YAGO15K' else extract_edges_FB(relation_triples, literal_of_int)
        edge_list += edge_of_int
        relations += relation_of_int

    asym_edge_list = drop_sym(edge_list)

    x = literal_triples.numeric.values.copy()
    u = np.array([1] * len(train) + [0]*len(valid) + [0]*len(test), dtype=bool)
    taus, omegas, _, _ = estimate_params(edge_list, x)

    x_0 = torch.tensor(x, device=device)
    u_0 = torch.tensor(u, device=device)
    x_0[u_0 == 0] = 0  
    attrs = literal_triples.attribute.values

    model = MrAP(device=device, edge_list=asym_edge_list, omega=omegas, tau=taus)
    pred = iter_MrAP(x_0, u_0, model, xi=0.5, entity_labels=attrs) 

    attr_valid_pred = {k: {} for k in attr_of_int}
    attr_test_pred = {k:{} for k in attr_of_int}
    u_valid = np.array([1] * len(train) + [0] * len(valid) + [1] * len(test), dtype=bool)
    u_valid = torch.tensor(u_valid, device=device)
    u_test = np.array([1] * len(train) + [1] * len(valid) + [0] * len(test), dtype=bool)
    u_test = torch.tensor(u_test, device=device)
    for attr in attr_of_int:
        valid_idx = torch.tensor(attrs == attr, device=device) & (u_valid == 0)
        attr_valid_pred[attr] = np.array(pred[valid_idx].cpu())
        test_idx = torch.tensor(attrs == attr, device=device) & (u_test == 0)
        attr_test_pred[attr] = np.array(pred[test_idx].cpu())
        print(attr, attr_test_pred[attr].shape)

    print('graphModule ok')
    return attr_valid_pred, attr_test_pred
