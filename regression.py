# 各种嵌入（随机、transe系列、plm编码）+各种回归模型的实验

import argparse
import numpy as np
import torch
import json, os
from utils import *


# 提取entity/attribute/value
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


# 把id转为name列表,统一处理FB15K和YAGO15K，不在转换表中的就返回本身就可以了
def id2name(entity):
    mid2name = {}  # FB15K的实体名转换表
    with open('./helpers/FB15K_mid2name.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split()
            mid2name[tris[0]] = tris[1]
    # print(len(mid2name))

    ename = []
    for id in entity:
        ename.append(id if id not in mid2name else mid2name[id])
    return ename


# 获取多语向量编码
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


# 获取实体编码,都输出为numpy类型
def get_emb(entity, args, dim=50):
    emb = []
    embed = args.embed
    # 这几种随机数结果差别不大，randn可能稍微好一点
    if embed == 'random':  # 0-1之间的随机数
        emb = np.random.random([len(entity), dim])
    elif embed == 'uniform':  # 0-1间均匀分布的随机数
        emb = np.random.uniform(0, 1, [len(entity), dim])
    elif embed == 'randn':  # 标准正态的随机数
        emb = np.random.randn(len(entity), dim)
    elif embed == 'sample':  # 随机浮点数
        emb = np.random.random_sample([len(entity), dim])
    # 使用plm与kge结合进行编码,这个里面也会有bert，所以要放在下一类的前面
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

        # 临时的，后面这个给去掉
        if kge_type == 'randn':
            emb = []
            for e in entity:
                emb1 = plm_emb[plm_e2id[e]]
                emb2 = np.random.randn(128)
                emb.append(np.hstack((emb1, emb2)))
            emb = np.array(emb)
            return emb

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
    # 使用plm进行编码
    elif 'bert' in embed or 'checkpoint' in embed:
        if os.path.exists(args.save_dir + 'emb.npy'):
            # 之前已经求过embedding，可以直接加载，不过要注意实体顺序变换
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
        elif args.desc in ['EG', 'EF', 'FG', 'EFG']:  # 多语描述，直接向量组合
            emb = get_ML_emb(entity, args.desc)
        else:
            from transformers import AutoTokenizer, AutoModel
            try:
                tokenizer = AutoTokenizer.from_pretrained(embed)
            except:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained(embed)
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            # device = 'cpu'
            model.to(device)

            # 这里区分编码实体名or实体描述
            if args.desc is not None:
                entity_desc = {}
                # 可以使用20iswc的多语实体描述，也可以使用16aaai的更多实体的英文描述等
                with open('./helpers/' + args.data + '_' + args.desc + '.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        tris = line[:-1].split('\t', 1)
                        entity_desc[tris[0]] = tris[1].replace('"', '')

                desc = []
                for e in entity:
                    # 如果没有实体描述的话，就把它自己的名字放进去
                    desc.append(entity_desc[e] if e in entity_desc else id2name([e])[0])
                print(len(desc))  # 15081
                # 把描述trunc到模型最大长度，bert是512。这个挺快
                inputs = tokenizer(desc, padding='max_length', truncation=True, return_tensors='pt')
            else:  # 对实体名进行tokenizer，按最长的来
                # FB15K和YAGO15K统一进行id2name，不过YAGO15K的其实不变
                ename = id2name(entity)
                inputs = tokenizer(ename, padding=True, truncation=True, return_tensors='pt')

            # 这一步还是需要计算挺久的,大模型跑不出来，需要分批次
            emb = []
            batch_size = 4  # 1024  # 128
            for i in range(0, len(entity), batch_size):
                # print(i)
                # 对大模型来说有点慢，每1024个大约需要半分钟；
                # 用实体描述的话更慢，放在gpu上batch_size=16都cuda溢出，4对bert-base可以
                # bert-large对实体描述最多只能跑4，bert-base的大概可以跑8
                output = model(inputs['input_ids'][i:i + batch_size].to(device)).last_hidden_state
                emb.extend(torch.mean(output, dim=1).detach().cpu().numpy())
            emb = np.array(emb)
            np.save(args.save_dir + 'emb.npy', emb)
            with open(args.save_dir + 'entitys.txt', 'w', encoding='utf-8') as f:
                for e in entity:
                    f.write(e + '\n')
            print('embed ok')
    # 使用libkge训练好的实体向量
    # 其中yago的向量有两种方式:
    # 一是基于MMKB的sameas，有11142个实体有对应，直接使用fb15k237上训练好的（已经写入entity_ids.del）；
    # 二是使用专为yago3训练的complex
    else:  # if embed == 'transe' or 'rotate' or 'complex':
        # 除了complex在自身目录下找，其他都在FB15K下，因为YAGO15K可使用FB15K的model
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
        # 91 not found for fb15k
        # 4208 not found for yago in transe/rotate
        # 3597 not found for yago in complex

    return emb


# 加载回归模型
def get_model(modeltype):
    from sklearn import linear_model, neural_network
    if modeltype == 'linear':
        model = linear_model.LinearRegression()
    elif modeltype == 'ridge':
        model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    elif modeltype == 'lasso':
        # lasso对正则化系数非常敏感，经常不收敛
        # 可以设置norminalize=True或增加tol以消除警告，但是前者建议使用StandardScaler
        # 数据进行归一化了之后似乎也没有用，不显示设置alphas让系统自己找100组，结果没有变好
        model = linear_model.LassoCV(alphas=[0.1, 1.0, 10.0])
    elif modeltype == 'mlp':
        model = neural_network.MLPRegressor(random_state=1, max_iter=500)
    return model


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main():
    print('hello')
    # 0. 解析参数
    parser = argparse.ArgumentParser(description='embedding + regression')
    parser.add_argument('--data', type=str, default='FB15K', choices=['YAGO15K', 'FB15K', 'DB15K'], help='used dataset')
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

    # 1. 读取数据
    train, test, valid = get_data(args.data)
    print(len(train), len(test), len(valid))
    entity, attribute, value = get_lists(train, test, valid)
    print(len(entity), len(attribute), len(value))
    # print(entity[:100]) # 注意每次返回的实体列表顺序不一样
    e2id = dict(zip(entity, range(len(entity))))
    a2id = dict(zip(attribute, range(len(attribute))))
    print(len(e2id), len(a2id))

    # 2. 获取embed
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

    # 3. 训练回归模型 & 模型选择
    # 对每个属性尝试多种回归模型
    # 选取在验证集上最好的模型，等下在测试集上评估
    # linear没有超参数，ridge和lasso的alpha使用自带的交叉验证搜索
    # mlp使用GridSearchCV搜索参数，并尝试正则化与否，单独训练
    # 计算的指标包括mae、rmse和r2，当回归函数拟合效果差于取平均值时R2会为负数
    attr_valid_result = {k: {} for k in attr_of_int}
    for attr in attr_of_int:
        # 归一化好像也没什么用
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # scaler.fit(attr_trainX[attr])
        # attr_trainX[attr] = scaler.fit_transform(attr_trainX[attr])
        # attr_validX[attr] = scaler.transform(attr_validX[attr])
        # 按照官网教程进行pipeline也不能让lasso收敛或结果提高
        # from sklearn.preprocessing import StandardScaler
        # from sklearn.pipeline import  make_pipeline
        # from sklearn.linear_model import LassoCV
        # model = make_pipeline(StandardScaler(),LassoCV()).fit(attr_trainX[attr], attr_trainY[attr])
        # pred = model.predict(attr_validX[attr])
        # result = get_performance(attr_validY[attr], pred)
        # print(result)
        # return
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

    # # mlp不太容易收敛和找到合适的参数
    # model = get_model('mlp')
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.preprocessing import StandardScaler
    # parameters = {'hidden_layer_sizes': [(100,), (100, 30)],
    #               'activation': ['tanh', 'relu'],
    #               'solver':['sgd', 'adam'],
    #               'alpha': [0.01, 0.1, 1.0],
    #               }
    # estimator = GridSearchCV(model, parameters, n_jobs=4)
    #
    # for attr in attr_of_int:
    #     # 只在训练集上fit，然后在train和test上以相同的指标缩放
    #     # 但是归一化之后结果好像更差了
    #     scaler = StandardScaler()
    #     scaler.fit(attr_trainX[attr])
    #     attr_trainX[attr] = scaler.transform(attr_trainX[attr])
    #     attr_testX[attr] = scaler.transform(attr_testX[attr])
    #
    #     estimator.fit(attr_trainX[attr], attr_trainY[attr])
    #     print(estimator.best_params_)
    #     pred = estimator.predict(attr_testX[attr])
    #     mae = mean_absolute_error(attr_testY[attr], pred)
    #     mse = mean_squared_error(attr_testY[attr], pred)
    #     print(f'for {attr}: mae={mae}, mse={mse}, rmse={np.sqrt(mse)}')
    #     attr_result[attr]['mae'] = mae
    #     attr_result[attr]['mse'] = mse
    #     attr_result[attr]['rmse'] = np.sqrt(mse)

    print('----------')

    # 4. 在测试集上评估
    # 虽然测试集和验证集上的结果可能差了20，但基本上在验证集上最好的模型的确在测试集上表现最好
    attr_test_result = {k: {} for k in attr_of_int}
    for attr in attr_of_int:
        model = attr_valid_result[attr]['model']
        pred = model.predict(attr_testX[attr])
        attr_test_result[attr] = get_performance(attr_testY[attr], pred)

    # 5. 计算整体分数
    total_result = get_total_result(attr_of_int, attr_test_result)
    print(attr_test_result)
    print(total_result)

    # 6. 结果保存
    save_to_file(save_dir + 'attr_valid_result.json', attr_valid_result)
    save_to_file(save_dir + 'attr_test_result.json', attr_test_result)
    save_to_file(save_dir + 'total_result.json', total_result)

    print('finish')


if __name__ == '__main__':
    # 只有当作main时才运行，当作模块导入时不运行
    main()
