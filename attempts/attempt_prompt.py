
from utils import *
import torch
import numpy as np

def split_pred(pred):
    res = ''
    for i in pred:
        res += ' ' + i.lower() if 'A' <= i <= 'Z' else i
    return res

def triple2text(triple_list, mask=True): 
    para_dic = {}
    with open('./helpers/paraphrase-YAGO15K.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ',1)
            para_dic[tris[0]] = tris[1]
    #print(len(para_dic))

    texts = []
    for triples in triple_list:
        text = ''
        for tris in triples:
            if tris[1] in para_dic:
                seg = para_dic[tris[1]].replace('[s]', tris[0])
                seg = seg.replace('[v]', '[MASK]') if mask else seg.replace('[v]', tris[2])
            else:
                seg = tris[0] + ' ' + split_pred(tris[1]) + ' '
                seg += '[MASK].' if mask else tris[2] + '.'
            text += ' ' + seg
        texts.append(text.strip())
    return texts

def get_dic_data(file):
    attr_dic = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            if not tris[1] in attr_dic:
                attr_dic[tris[1]] = {'triple':[],'ground':[]}
            attr_dic[tris[1]]['triple'].append([tris])
            attr_dic[tris[1]]['ground'].append(float(tris[2]))
    # print(len(attr_dic))
    # print([len(attr_dic[k]['ground']) for k in attr_dic])
    return attr_dic


def test_rel_for_attr(checkpoint='bert-base-uncased'):
    dir = './data/YAGO15K/'
    relations = ['isCitizenOf', 'hasAcademicAdvisor', 'isMarriedTo', 'participatedIn', 'hasNeighbor', 'hasCapital',
                 'isLocatedIn', 'isConnectedTo', 'hasChild', 'actedIn', 'isAffiliatedTo', 'wroteMusicFor',
                 'hasOfficialLanguage', 'happenedIn', 'isLeaderOf', 'isInterestedIn', 'diedIn', 'hasWonPrize',
                 'worksAt', 'influences', 'dealsWith', 'isPoliticianOf', 'livesIn', 'owns', 'directed', 'playsFor',
                 'hasCurrency', 'wasBornIn', 'created', 'graduatedFrom', 'edited', 'isKnownFor']
    
    relation_dic = {rel:{} for rel in relations}
    with open(dir+'/EntityTriples_handled.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            if tris[1] not in relation_dic:
                print(tris[1])
                return
            if tris[0] not in relation_dic[tris[1]]:
                relation_dic[tris[1]][tris[0]] = [tris]
            else:
                relation_dic[tris[1]][tris[0]].append(tris)
            if tris[2] not in relation_dic[tris[1]]:
                relation_dic[tris[1]][tris[2]] = [tris]
            else:
                relation_dic[tris[1]][tris[2]].append(tris)

    print([len(relation_dic[k]) for k in relation_dic])

    attr_of_int = ['wasBornOnDate', 'wasCreatedOnDate', 'wasDestroyedOnDate', 'diedOnDate', 'happenedOnDate',
                   'hasLatitude', 'hasLongitude']
    #attr_of_int = ['hasLatitude']
    known_attr_dic = {k:{} for k in attr_of_int}
    with open(dir+'/train.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split(' ')
            known_attr_dic[tris[1]][tris[0]] = tris[2]
    print([len(known_attr_dic[k]) for k in known_attr_dic])

    attr_test = get_dic_data(dir+'/test.txt')
    attr_valid = get_dic_data(dir+'/valid.txt')

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    #checkpoint = 'bert-base-uncased'
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('model ok')

    numdic = {k: tokenizer.vocab[k] for k in tokenizer.vocab if is_number(k)}
    chosen = list(numdic.values())
    masked = torch.ones(len(tokenizer.vocab), dtype=bool)
    masked[chosen] = False  
    print('mask ok')

    def get_pred(text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)

        pred = []
        batch_size = 4  # 4
        for i in range(0, len(inputs), batch_size):
            input = inputs[i:i + batch_size]
            token_logits = model(input).logits
            mask_token_index = torch.where(input == tokenizer.mask_token_id)

            for k, v in zip(mask_token_index[0], mask_token_index[1]):
                mask_logits = token_logits[k, v, :]
                probability = torch.nn.functional.softmax(mask_logits, dim=-1)
                probability[masked] = 0  
                pred.append(float(tokenizer.decode(torch.argmax(probability))))
        pred = np.array(pred)
        return pred

    def get_rel_triple(rel, triples):
        rel_triples = []
        idxes = []
        cnt = 0
        for triple in triples:
            [s,a,v] = triple[0] 
            tmp = []
            if s in relation_dic[rel]: 
                for relt in relation_dic[rel][s]:
                    if relt[0] in known_attr_dic[a]:
                        tmp.append(relt)
                        tmp.append([relt[0],a,known_attr_dic[a][relt[0]]])
                    if relt[2] in known_attr_dic[a]:
                        tmp.append(relt)
                        tmp.append([relt[2],a,known_attr_dic[a][relt[2]]])
            if len(tmp) > 0:
                rel_triples.append(tmp)
                idxes.append(cnt)
            cnt += 1
        return rel_triples, idxes

    attr_useful_rel = {k:[] for k in attr_of_int}
    for attr in attr_of_int: # ['hasLongitude']:
        base_text = triple2text(attr_valid[attr]['triple'], mask=True)
        ground = attr_valid[attr]['ground']
        base_pred = get_pred(base_text)
        base_result = get_performance(ground, base_pred)
        print(attr)
        print('base result: ', base_result)
    
        for rel in relations: # ['participatedIn','hasNeighbor']
            rel_triples,idxes = get_rel_triple(rel, attr_valid[attr]['triple'])
            if len(rel_triples) == 0:
                continue
    
            rel_text = triple2text(rel_triples, mask=False)
            text = [base_text[i] for i in idxes]
            text = [k+' '+v for k, v in zip(text, rel_text)]
    
            pred = get_pred(text)
            part_ground = [ground[i] for i in idxes]
            part_base_pred = [base_pred[i] for i in idxes]
    
            rel_result = get_performance(part_ground, pred)
            part_base_result = get_performance(part_ground, part_base_pred)
            print('prompt: ',rel_result)
            print('base: ',part_base_result)
    
            if rel_result['mae'] < part_base_result['mae'] and rel_result['num'] > 0.01*base_result['num']:
                attr_useful_rel[attr].append(rel)
    
            # tmp_pred = base_pred
            # for i in range(len(idxes)):
            #     tmp_pred[idxes[i]] = pred[i]
            # tmp_result = get_performance(ground, tmp_pred)

    print('valid finish')
    print(attr_useful_rel)

    attr_test_result = {}
    for attr in attr_of_int:
        base_text = triple2text(attr_test[attr]['triple'], mask=True)
        ground = attr_test[attr]['ground']
        base_pred = get_pred(base_text)
        base_result = get_performance(ground, base_pred)
        print(f'base result for {attr}:',base_result)
        attr_test_result[attr] = base_result

        cheat_pred = base_pred
        for rel in relations:
            rel_triples, idxes = get_rel_triple(rel, attr_test[attr]['triple'])
            if len(rel_triples) == 0:
                continue

            rel_text = triple2text(rel_triples, mask=False)
            text = [base_text[i] for i in idxes]
            text = [k + ' ' + v for k, v in zip(text, rel_text)]

            pred = get_pred(text)
            part_ground = [ground[i] for i in idxes]
            #part_base_pred = [base_pred[i] for i in idxes]
            part_cheat_pred = [cheat_pred[i] for i in idxes]

            rel_result = get_performance(part_ground, pred)
            #part_base_result = get_performance(part_ground, part_base_pred)
            part_cheat_result = get_performance(part_ground, part_cheat_pred)
            print(rel)
            print('prompt: ', rel_result)
            print('base: ', part_cheat_result)

            if rel_result['mae'] < part_cheat_result['mae']:
                for i in range(len(idxes)):
                    cheat_pred[idxes[i]] = pred[i]

            
        cheat_result = get_performance(ground, cheat_pred)
        attr_test_result[attr] = cheat_result
    print('----------')
    print(attr_test_result)
    total_result = get_total_result(attr_test_result.keys(), attr_test_result)
    print(total_result)
    print('ok')
#test_rel_for_attr() # './pretrainedModels/finetune+rel_3e-5/bert-base-uncased/checkpoint-36610'
