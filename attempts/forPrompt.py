# Auxiliary functions for prompt attempt

# 对于待预测的属性，选取具有该属性的一跳关系，即相当于local了？
# 这样严格限制下，YAGO15K的2354个样本里，有1315个有
# 这里得到的是三元组，还没有进行paraphrase
def get_prompt_triple():
    dir = '../data/YAGO15K/'
    relation_dic = {}
    with open(dir+'/EntityTriples_handled.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            if tris[0] in relation_dic:
                relation_dic[tris[0]].append(tris)
            else:
                relation_dic[tris[0]] = [tris]
            if tris[2] in relation_dic:
                relation_dic[tris[2]].append(tris)
            else:
                relation_dic[tris[2]] = [tris]
    print(len(relation_dic)) # 15404

    attr_of_int = ['wasBornOnDate', 'wasCreatedOnDate', 'wasDestroyedOnDate', 'diedOnDate', 'happenedOnDate',
                   'hasLatitude', 'hasLongitude']
    known_attr_dic = {k:{} for k in attr_of_int}
    with open(dir+'/train_100.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            known_attr_dic[tris[1]][tris[0]] = tris[2]
    with open(dir + '/valid.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            known_attr_dic[tris[1]][tris[0]] = tris[2]
    print(len(known_attr_dic)) # 7
    print([len(known_attr_dic[k]) for k in known_attr_dic]) # []

    cnt = 0
    with open(dir + '/test.txt', 'r', encoding='utf-8') as f:
        fout = open(dir+'/test_prompt_1hop.txt','w',encoding='utf-8')
        for line in f:
            tris = line[:-1].split(' ')
            attr = tris[1]
            res = [tris]
            for relation in relation_dic[tris[0]]:
                if relation[0] in known_attr_dic[attr]:
                    res.append(relation)
                    res.append([relation[0],attr,known_attr_dic[attr][relation[0]]])
                if relation[2] in known_attr_dic[attr]:
                    res.append(relation)
                    res.append([relation[2], attr, known_attr_dic[attr][relation[2]]])
            #print(res)
            if (len(res) > 1):
                cnt += 1
            res = [' '.join(res[i]) for i in range(len(res))]
            fout.write('\t'.join(item for item in res) + '\n')
        fout.close()

    print(cnt) # 1315
    print('ok')
#get_prompt()

# 把关系谓词按大写字母拆开作为释义
def split_pred(pred):
    res = ""
    for i in pred:
        res += ' ' + i.lower() if 'A' <= i <= 'Z' else i
    return res

# yago的32个关系直接按大写字母拆开
# fb的1345个再说
def get_relation_dic():
    relations = ['isCitizenOf', 'hasAcademicAdvisor', 'isMarriedTo', 'participatedIn', 'hasNeighbor', 'hasCapital',
                 'isLocatedIn', 'isConnectedTo', 'hasChild', 'actedIn', 'isAffiliatedTo', 'wroteMusicFor',
                 'hasOfficialLanguage', 'happenedIn', 'isLeaderOf', 'isInterestedIn', 'diedIn', 'hasWonPrize',
                 'worksAt', 'influences', 'dealsWith', 'isPoliticianOf', 'livesIn', 'owns', 'directed', 'playsFor',
                 'hasCurrency', 'wasBornIn', 'created', 'graduatedFrom','edited','isKnownFor']
    pred_dic = {}
    for relation in relations:
        pred = split_pred(relation)
        pred_dic[relation] = '[s] ' + pred + ' [v].'
    print(len(pred_dic))
    return pred_dic

# 把prompt的三元组数据转为json格式里的文本
def build_plm_prompt_json():
    import json, random

    para_dic = {}
    with open('paraphrase-YAGO15K.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ', 1)
            para_dic[tris[0]] = tris[1]
    print(len(para_dic))
    attr_dic = {k: {'text': [], 'ground': []} for k in para_dic}

    # 把关系和属性的paraphrase形式统一了放在一个dic里
    tmp = get_relation_dic()
    for k in tmp:
        if k in para_dic:
            print(k)
    para_dic.update(get_relation_dic())
    print(len(para_dic))

    dir = '../data/YAGO15K/'
    with open(dir+'/test_prompt_1hop.txt','r',encoding='utf-8') as f:
        for line in f:
            triples = line[:-1].split('\t')
            test = triples[0].split(' ')
            triples = list(set(triples[1:]))
            random.shuffle(triples)
            attr_dic[test[1]]['ground'].append(float(test[2]))
            # 为了避免在trunc的时候MASK被删去，这里统一把待预测句子放在开头，后续可考虑shuffle的影响
            # text = para_dic[test[1]].replace('[s]',test[0]).replace('[v]','[MASK]')
            # for triple in triples:
            #     triple = triple.split(' ')
            #     text += ' '+para_dic[triple[1]].replace('[s]',triple[0]).replace('[v]',triple[2])

            # 这里尝试把test句子放在中间--不行，还是可能会被切掉
            test_text = para_dic[test[1]].replace('[s]',test[0]).replace('[v]','[MASK]')
            text = ''
            # 这里的25是为了防止取的太大导致等下MASK被trunck掉，别的数据集可能需要换一个数
            idx = random.randint(0,min(len(triples),25))
            for triple in triples[:idx]:
                triple = triple.split(' ')
                text += ' '+para_dic[triple[1]].replace('[s]',triple[0]).replace('[v]',triple[2])
            text += ' ' + test_text
            for triple in triples[idx:]:
                triple = triple.split(' ')
                text += ' ' + para_dic[triple[1]].replace('[s]', triple[0]).replace('[v]', triple[2])
            attr_dic[test[1]]['text'].append(text.strip())
    fout = open(dir + '/test_prompt_1hop_rand1.json', 'w', encoding='utf-8')
    json.dump(attr_dic, fout)
    fout.close()

    print('ok')
#build_plm_prompt_json()

# 分析不同属性的prompt数据里关系类别，check变好/坏的原因
def analysis():
    attr_of_int = ['wasBornOnDate', 'wasCreatedOnDate', 'wasDestroyedOnDate', 'diedOnDate', 'happenedOnDate',
                   'hasLatitude', 'hasLongitude']
    attr_relation_dic = {k:[] for k in attr_of_int}
    with open('../data/YAGO15K/test_prompt_1hop.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split('\t')
            attr = tris[0].split(' ')[1]
            for tri in tris[1:]:
                rel = tri.split(' ')[1]
                attr_relation_dic[attr].append(rel)
    attr_relation_dic = {k:list(set(attr_relation_dic[k])) for k in attr_relation_dic}
    print(attr_relation_dic)
#analysis()

def test():
    entities = []
    # 训练集里18825条属性，涉及13039个实体（yago总实体是15404个）
    # 这些实体拥有总的122886个关系里的119122个
    # test.txt里有2354条，涉及2226个不同实体，其中有1272个在训练集里已知其他属性
    with open('../data/YAGO15K/train_100.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            entities.append(tris[0])
    print(len(entities))
    entities = list(set(entities))
    print(len(entities))
    dic = {k:1 for k in entities}

    cnt = 0
    test_entities = []
    with open('../data/YAGO15K/test.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            test_entities.append(tris[0])
            if tris[0] in dic:
                cnt += 1
    print(len(test_entities))
    test_entities = list(set(test_entities))
    print(len(test_entities))
    print(cnt)

    # with open('../data/YAGO15K/EntityTriples_handled.txt','r',encoding='utf-8') as f:
    #     for line in f:
    #         tris = line[:-1].split(' ')
    #         if tris[0] in dic or tris[2] in dic:
    #             cnt += 1
    # print(cnt)

#test()

# 对2354个测试样本里，基于一跳关系涉及1315个样本
# 关系条目和属性条目各用到了6268条
# 去重后其实一共只有8666个，包含了5707个关系和2959个属性
def read():
    cnt = 0
    rel_dic = []
    attr_dic = []
    with open('../data/YAGO15K/test_prompt_1hop.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split('\t')
            for i in range(1,len(tris),2):
                rel_dic.append(tris[i])
                attr_dic.append(tris[i+1])

    print(cnt/2)
    print(len(rel_dic),len(attr_dic))
    print(len(list(set(rel_dic))), len(list(set(attr_dic))))
#read()



# 关系数据和属性数据联合使用的一种方式：都拿去finetune模型
# 这里构造一下对应的数据集，关系数据也按照8:1:1放入train/test/valid
def add_relation_text():
    import random
    rel_dic = get_relation_dic()
    relations = []
    with open('../data/YAGO15K/EntityTriples_handled.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            if tris[1] not in rel_dic:
                print(tris)
                return
            relations.append(rel_dic[tris[1]].replace('[s]',tris[0]).replace('[v]',tris[2]))
    random.shuffle(relations)
    num = len(relations)
    with open('../data/YAGO15K/train_100_mlm.txt','r',encoding='utf-8') as f:
        dic = relations[0:int(num*0.8)]
        for line in f:
            dic.append(line[:-1])
        print(len(dic))
        random.shuffle(dic)
        fout = open('../data/YAGO15K/train_100+rel_mlm.txt','w',encoding='utf-8')
        fout.write('\n'.join(dic))
        fout.close()
    with open('../data/YAGO15K/valid_mlm.txt','r',encoding='utf-8') as f:
        dic = relations[int(num*0.8):int(num*0.9)]
        for line in f:
            dic.append(line[:-1])
        print(len(dic))
        random.shuffle(dic)
        fout = open('../data/YAGO15K/valid+rel_mlm.txt','w',encoding='utf-8')
        fout.write('\n'.join(dic))
        fout.close()
    with open('../data/YAGO15K/test_mlm.txt','r',encoding='utf-8') as f:
        dic = relations[int(num*0.9):]
        for line in f:
            dic.append(line[:-1])
        print(len(dic))
        random.shuffle(dic)
        fout = open('../data/YAGO15K/test+rel_mlm.txt','w',encoding='utf-8')
        fout.write('\n'.join(dic))
        fout.close()
    print('ok')
#add_relation_text()

# mnm是固定的超参数：train_batch_size=32,epoch=10,lr=3e-5/1e-2
# 跟之前mlm里面的finetune方式一样，只是数据集里增加了关系数据
def finetune_plm(data='YAGO15K', checkpoint='bert-base-uncased'):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from datasets import load_dataset
    dir = '../data/'+data
    data_files = {'train':dir+'/train_100+rel_mlm.txt','test':dir+'/test+rel_mlm.txt',
                  'validation':dir+'/valid+rel_mlm.txt'}
    raw_datasets = load_dataset('text',data_files=data_files)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = '[PAD]'

    def tokenize_function(sample):
        # 单独使用tokenizer的话可以指定pytorch类型，但是这里使用map和batched用这个会有问题
        # return tokenizer(sample['text'], return_tensors='pt',truncation=True, padding=True)
        # padding的配置与参数组合看https://huggingface.co/docs/transformers/preprocessing#everything-you-always-wanted-to-know-about-padding-and-truncation
        result = tokenizer(sample['text'], truncation=True, padding='max_length', max_length=50)
        result['labels'] = result["input_ids"].copy()
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        # 把原始的那些列都删掉，只剩下input_ids,token_type_ids和attention_mask
        remove_columns=raw_datasets['train'].column_names,
    )
    print(tokenized_datasets)

    # samples = tokenized_datasets['train'][0:]
    # print([len(x) for x in samples['input_ids']])
    # print([len(x) for x in samples['labels']])

    from transformers import DataCollatorForLanguageModeling,AutoModelForMaskedLM
    # 注意使用DataCollatorForLanguageModeling不会动态padding，需要在前面显式指定
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm_probability=0.15)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    save_dir = '../pretrainedModels/finetune+rel_3e-5/' + checkpoint

    # 这里详细配置一下参数
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-5,#3e-5, # 1e-2
        weight_decay=0.01,
        num_train_epochs=20, # 训练100个epoch看看会不会能更高
        per_device_train_batch_size=32,#32
        # 在prediction的时候，即使batch_size设成1 cuda也不够
        per_device_eval_batch_size=512,
        save_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        # 因为词表都是bert，所以先不保存了
        # tokenizer=tokenizer,# 注意，这个不加就不会保存tokenizer的东西（
    )

    trainer.train()
    print('train finish')

    # 这个cuda跑不动，后面单独跑吧
    # predictions = trainer.predict(tokenized_datasets['test'])
    # print(predictions.predictions.shape)  # logits
    # print(predictions.metrics)

    print('ok')
#finetune_plm()
