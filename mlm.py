# Implementations of MLM and several probing tests

import argparse
import os
from utils import *
import torch
import numpy as np


# impact of paraphrase
def test_paraphrase():
    # paraphrase templates for 'wasCreatedOnDate'
    templates = ['[s] was created on the date of [MASK].', '[s] is created on the date of [MASK].', \
                 '[s] was created in the year of [MASK].', '[s] is created in the year of [MASK].', \
                 '[s] was created in [MASK].', '[s] is created in [MASK].']

    test_entity = []
    test_value = []
    with open('./data/YAGO15K/test.txt', 'r', encoding='utf-8') as test_file:
        for line in test_file:
            tris = line[:-1].split(' ')
            if tris[1] == 'wasCreatedOnDate':
                test_entity.append(tris[0])
                test_value.append(float(tris[2]))
    ground = np.array(test_value)
    print(len(test_entity))  # 651

    checkpoint = 'bert-base-uncased'
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    numdic = {k: tokenizer.vocab[k] for k in tokenizer.vocab if is_number(k)}
    nums = np.array([float(k) for k in numdic.keys()])
    print(f'{nums.shape} numbers in {checkpoint}, max={np.max(nums)}, \
        min={np.min(nums)}, mean={np.mean(nums)}, median={np.median(nums)}')

    chosen = list(numdic.values())
    masked = torch.ones(len(tokenizer.vocab), dtype=bool)
    masked[chosen] = False  

    # test for each template
    temp_test_result = {}
    for temp in templates:
        print(temp)
        text = []
        for entity in test_entity:
            text.append(temp.replace('[s]', entity))

        inputs = tokenizer(text, padding=True, return_tensors="pt")
        token_logits = model(**inputs).logits
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
        pred = []
        rank = []  # get the highest rank of numerical values
        for i, k in zip(mask_token_index[0], mask_token_index[1]):
            mask_logits = token_logits[i, k, :]
            sorted, indices = torch.sort(mask_logits, descending=True)
            mask_logits[masked] = 0  
            pred_id = torch.argmax(mask_logits)  
            pred.append(float(tokenizer.decode(pred_id)))
            rank.append(indices.numpy().tolist().index(pred_id))
        pred = np.array(pred)
        result = get_performance(ground, pred)
        print(result)
        rank = np.array(rank)
        print(f'max={np.max(rank)},min={np.min(rank)},mean={np.mean(rank)},median={np.median(rank)}')
        temp_test_result[temp] = {'result': result, 'rank-mean': np.mean(rank), 'rank-max': np.max(rank),
                                  'rank-min': np.min(rank)}
    print(temp_test_result)
    print('test finish')


# impact of epoch
def test_checkpoint(file_name='./data/FB15K/test_mlm.json'):
    dir = './pretrainedModels/finetune_3e-5/bert-base-uncased/'
    ckpt_list = os.listdir(dir)
    print(dir,ckpt_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import json
    with open(file_name, 'r', encoding='utf-8') as file:
        attr_dic = json.load(file)
    print(file_name, len(attr_dic))

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(dir + '/checkpoint-589/')
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print('tokenizer ok')

    numdic = {k: tokenizer.vocab[k] for k in tokenizer.vocab if is_number(k)}
    print('numdic ok')
    chosen = list(numdic.values())
    masked = torch.ones(len(tokenizer.vocab), dtype=bool)
    masked[chosen] = False  

    best_ckpt = 0
    best_mae = 10000
    for ckpt in ckpt_list:
        if ckpt == 'runs':
            continue
        model = AutoModelForMaskedLM.from_pretrained(dir + ckpt)
        model.to(device)
        #print('model ok')

        attr_test_result = {}
        attr_diverse_result = {}
        for attr in attr_dic:
            text = [k.replace('[MASK]', tokenizer.mask_token) for k in attr_dic[attr]['text']]
            inputs = tokenizer(text, padding=True, return_tensors="pt")['input_ids'].to(device)
            ground = np.array(attr_dic[attr]['ground'])

            pred = []
            batch_size = 64  # 4
            for i in range(0, len(text), batch_size):
                input = inputs[i:i + batch_size]
                token_logits = model(input).logits
                mask_token_index = torch.where(input == tokenizer.mask_token_id)

                for k, v in zip(mask_token_index[0], mask_token_index[1]):
                    mask_logits = token_logits[k, v, :]
                    probability = torch.nn.functional.softmax(mask_logits, dim=-1)
                    probability[masked] = 0  
                    pred.append(float(tokenizer.decode(torch.argmax(probability))))

            pred = np.array(pred)
            result = get_performance(ground, pred)
            attr_test_result[attr] = result

        total_result = get_total_result(attr_test_result.keys(), attr_test_result)
        print(ckpt, total_result)
        if total_result['micro']['mae'] < best_mae:
            best_mae = total_result['micro']['mae']
            best_ckpt = ckpt
    print(best_mae, best_ckpt)
    print('ok')


# impact of padding
def test_padding_length(checkpoint='bert-base-uncased', file_name='./data/YAGO15K/test_mlm.json'):
    print(checkpoint, file_name)
    import json
    with open(file_name, 'r', encoding='utf-8') as file:
        attr_dic = json.load(file)
    print(len(attr_dic))

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    except:  
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    model.to(device)
    print('model ok')

    numdic = {k: tokenizer.vocab[k] for k in tokenizer.vocab if is_number(k)}
    print('check numdic ok')
    nums = np.array([float(k) for k in numdic.keys()])
    print(f'{nums.shape[0]} numbers in {checkpoint}, max={np.max(nums)}, \
        min={np.min(nums)}, mean={np.mean(nums)}, median={np.median(nums)}')

    chosen = list(numdic.values())
    masked = torch.ones(len(tokenizer.vocab), dtype=bool)
    masked[chosen] = False  

    batch_size = 4  # 64
    for length in [32,64,128,256,512]: 
        attr_test_result = {}
        for attr in attr_dic:
            text = [k.replace('[MASK]', tokenizer.mask_token) for k in attr_dic[attr]['text']]
            ground = np.array(attr_dic[attr]['ground'])

            # # padding to batch max_length
            # inputs = tokenizer(text,truncation=True,padding=True,return_tensors='pt')['input_ids'].to(device)
            # # padding to specific length：padding='max_length', max_length=length, truncation=True
            # inputs = tokenizer(text, truncation=True, padding='max_length', max_length=length,
            #                    return_tensors='pt')['input_ids'].to(device)

            pred = []
            for i in range(0, len(text), batch_size):
                input = tokenizer(text[i], return_tensors='pt')['input_ids'].to(device)
                # input = inputs[i:i+batch_size]
                token_logits = model(input).logits
                mask_token_index = torch.where(input == tokenizer.mask_token_id)

                for k, v in zip(mask_token_index[0], mask_token_index[1]):
                    mask_logits = token_logits[k, v, :]
                    probability = torch.nn.functional.softmax(mask_logits, dim=-1)
                    probability[masked] = 0  
                    pred.append(float(tokenizer.decode(torch.argmax(probability))))
            pred = np.array(pred)

            result = get_performance(ground, pred)
            print(attr, result)
            attr_test_result[attr] = result

        total_result = get_total_result(attr_test_result.keys(), attr_test_result)
        print(length, total_result)

    print('finish')


def main():
    print('hello mlm')
    parser = argparse.ArgumentParser(description='mlm')
    parser.add_argument('--data', type=str, default='FB15K', choices=['YAGO15K', 'FB15K', 'DB15K'],
                        help='used dataset')
    parser.add_argument('--checkpoint', type=str, default='bert-base-uncased')
    # attempt for prompt
    parser.add_argument('--prompt', type=str, default=None, help='use prompt data, e.g., 1hop')
    args = parser.parse_args()
    print(args)

    save_dir = './results/' + args.data + '/mlm/' + args.checkpoint + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.prompt is not None:
        save_dir += '/prompt_' + args.prompt + '_'

    attr_dic = get_mlm_data(args)
    print(len(attr_dic))

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    except:  
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint)
    model.to(device)
    print('model ok')

    numdic = {k: tokenizer.vocab[k] for k in tokenizer.vocab if is_number(k)}
    print('check numdic ok')
    nums = np.array([float(k) for k in numdic.keys()])
    print(f'{nums.shape[0]} numbers in {args.checkpoint}, max={np.max(nums)}, \
    min={np.min(nums)}, mean={np.mean(nums)}, median={np.median(nums)}')

    chosen = list(numdic.values())
    masked = torch.ones(len(tokenizer.vocab), dtype=bool)
    masked[chosen] = False  

    attr_test_result = {}
    attr_diverse_result = {}
    for attr in attr_dic:
        text = [k.replace('[MASK]', tokenizer.mask_token) for k in attr_dic[attr]['text']]
        ground = np.array(attr_dic[attr]['ground'])
        # padding to max length and trunc
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)

        pred = []
        batch_size = 64  # 4
        for i in range(0, len(text), batch_size):
            input = inputs[i:i + batch_size]
            token_logits = model(input).logits
            mask_token_index = torch.where(input == tokenizer.mask_token_id)

            for k, v in zip(mask_token_index[0], mask_token_index[1]):
                mask_logits = token_logits[k, v, :]
                probability = torch.nn.functional.softmax(mask_logits, dim=-1)
                probability[masked] = 0  
                pred.append(float(tokenizer.decode(torch.argmax(probability))))

        pred = np.array(pred)
        print(pred.shape)  # 834

        # check diversity
        diverse = check_diversity(pred)
        attr_diverse_result[attr] = diverse
        result = get_performance(ground, pred)
        print(attr, result)
        attr_test_result[attr] = result

    total_result = get_total_result(attr_test_result.keys(), attr_test_result)
    print(total_result)
    total_diversity = check_total_diversity(attr_diverse_result)
    # print(total_diversity)
    print(len(total_diversity))
    sorted_div = sorted(total_diversity.items(), key=lambda kv: (kv[1], kv[0]))
    print(sorted_div[0], sorted_div[-1])

    save_to_file(save_dir + 'attr_test_result.json', attr_test_result)
    save_to_file(save_dir + 'total_result.json', total_result)

    save_to_file(save_dir + 'total_diversity.json', total_diversity)

    print('finish')


if __name__ == '__main__':
    main()
    # test_paraphrase()
    # test_checkpoint()
    # test_padding_length('roberta-large')
    # test_padding_length(checkpoint='./pretrainedModels/finetune_3e-5/bert-base-uncased/checkpoint-5890')
