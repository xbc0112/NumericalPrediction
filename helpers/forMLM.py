# Auxiliary functions for MLM.


# change from triples to texts and save in json format
def build_plm_text_json():
    import json
    for data in ['YAGO15K','FB15K']:
        save_dir = '../data/' + data + '/'
        # mid2name = {}
        # # FB15K need a change from id to name
        # with open('FB15K_mid2name.txt','r',encoding='utf-8') as f:
        #     for line in f:
        #         tris = line[:-1].split()
        #         mid2name[tris[0]] = tris[1]
        # print(len(mid2name))

        para_file = 'paraphrase-' + data + '.txt'
        para_dic = {}
        with open(para_file, 'r', encoding='utf-8') as f:
            for line in f:
                tris = line[:-1].split(' ', 1)
                para_dic[tris[0]] = tris[1]
        #fout = open(save_dir+'/test_mlm.json','w',encoding='utf-8')
        fout = open(save_dir + '/valid_mlm.json', 'w', encoding='utf-8')
        attr_dic = {k:{'text':[],'ground':[]} for k in para_dic}
        with open(save_dir+'valid.txt','r',encoding='utf-8') as f:
            for line in f:
                tris = line[:-1].split(' ')
                temp = para_dic[tris[1]]
                attr_dic[tris[1]]['text'].append(temp.replace('[s]',tris[0]).replace('[v]','[MASK]'))
                attr_dic[tris[1]]['ground'].append(float(tris[2]))

                # if tris[1] not in para_dic:
                #     continue
                # if tris[0] not in mid2name:
                #     print(tris[0])
                #     continue
                #
                # temp = para_dic[tris[1]]
                # attr_dic[tris[1]]['text'].append(temp.replace('[s]',mid2name[tris[0]]).replace('[v]','[MASK]'))
                # attr_dic[tris[1]]['ground'].append(float(tris[2]))
        json.dump(attr_dic,fout)
        fout.close()
    print('ok')
#build_plm_text_json()

# change from triples to texts
def build_plm_text(data):
    save_dir = '../data/'+data+'/'
    para_file = 'paraphrase-'+data+'.txt'
    para_dic = {}
    with open(para_file, 'r', encoding='utf-8') as f:
        for line in f:
            tris = line.split(' ', 1)
            para_dic[tris[0]] = tris[1]

    mid2name = {} 
    if data == 'FB15K':
        with open('FB15K_mid2name.txt', 'r', encoding='utf-8') as f:
            for line in f:
                tris = line[:-1].split()
                mid2name[tris[0]] = tris[1]
    print(len(mid2name))

    for ttype in ['train','test','valid']:
        with open(save_dir+ttype+'.txt','r',encoding='utf-8') as f:
            fout = open(save_dir+ttype+'_mlm.txt','w',encoding='utf-8')
            for line in f:
                tris = line[:-1].split(' ')
                if tris[1] not in para_dic:
                    continue
                temp = para_dic[tris[1]]
                if tris[0] in mid2name:
                    tris[0] = mid2name[tris[0]]
                fout.write(temp.replace('[s]',tris[0]).replace('[v]',tris[2]))
            fout.close()
    print('ok')
#build_plm_text('FB15K')


# fine-tuning pre-trained language models
# hyper-parameters are set according to https://github.com/dspoka/mnm
# train_batch_size=32,epoch=10,lr=3e-5/1e-2
def finetune_plm(data='FB15K', checkpoint='bert-base-uncased'):
    from datasets import load_dataset
    
    dir = '../data/'+data
    data_files = {'train':dir+'/train_mlm.txt','test':dir+'/test_mlm.txt',
                  'validation':dir+'/valid_mlm.txt'}
    raw_datasets = load_dataset('text',data_files=data_files)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = '[PAD]'

    def tokenize_function(sample):
        result = tokenizer(sample['text'], truncation=True, padding='max_length', max_length=50)
        result['labels'] = result["input_ids"].copy()
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets['train'].column_names,
    )
    print(tokenized_datasets)

    from transformers import DataCollatorForLanguageModeling,AutoModelForMaskedLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm_probability=0.15)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    save_dir = '../pretrainedModels/'+data+'/finetune_3e-5/'+checkpoint

    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-5, # 1e-2
        weight_decay=0.01,
        num_train_epochs=10, 
        per_device_train_batch_size=32,
        per_device_eval_batch_size=512,
        save_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        # tokenizer=tokenizer, # used to save the vocab file
    )

    trainer.train()
    print('train finish')

    print('ok')
#finetune_plm()

# fine-tuning and adding new vocabs
def finetune_plm_with_vocab(data='YAGO15K', checkpoint='bert-base-uncased'):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from datasets import load_dataset
    dir = '../data/' + data
    data_files = {'train': dir + '/train_mlm.txt', 'test': dir + '/test_mlm.txt',
                  'validation': dir + '/valid_mlm.txt'}
    raw_datasets = load_dataset('text', data_files=data_files)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = '[PAD]'

    with open(dir + '/train.txt', 'r') as f:
        numeric_list = []
        for line in f:
            num = line[:-1].split(' ')[2]
            numeric_list.append(num)
        add_num = tokenizer.add_tokens(numeric_list)
        print(f'We have added {add_num} nums into the tokenizer')


    def tokenize_function(sample):
        result = tokenizer(sample['text'], truncation=True, padding='max_length', max_length=50)
        result['labels'] = result["input_ids"].copy()
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets['train'].column_names,
    )
    print(tokenized_datasets)

    from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    # resize
    model.resize_token_embeddings(len(tokenizer))

    save_dir = '../pretrainedModels/'+data+'/finetune_with_vocab_3e-5/' + checkpoint

    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-5,  # 1e-2
        weight_decay=0.01,
        num_train_epochs=10,  
        per_device_train_batch_size=32,  # 32
        per_device_eval_batch_size=512,
        save_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,# 注意，这个不加就不会保存tokenizer的东西（
    )

    trainer.train()
    print('train finish')
    tokenizer.save_pretrained(save_dir)
    print(len(tokenizer.vocab))

    print('ok')

#finetune_plm_with_vocab()

def add_vocab():
    import json
    with open('../pretrainedModels/finetune_with_vocab_3e-5/bert-base-uncased/added_tokens.json','r') as f:
        data = json.load(f)
        fout = open('../pretrainedModels/finetune_with_vocab_3e-5/bert-base-uncased/vocab.txt','a+')
        for key in data:
            fout.write(key+'\n')
        fout.close()
#add_vocab()

# get length of prompt texts
# test_mlm.json: 24、27、25、21、22、21、21
# test_prompt_1hop.json: 550\4032\182\764\22\950\2745
# test_prompt_1hop_rand.json: 550\4032\182\764\22\950\2745
def get_maxlen():
    import json
    file_name = '../data/YAGO15K/test_prompt_1hop_rand.json'
    with open(file_name, 'r', encoding='utf-8') as file:
        attr_dic = json.load(file)
    print(len(attr_dic))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    for attr in attr_dic:
        text = attr_dic[attr]['text']
        inputs = tokenizer(text, padding=True, return_tensors='pt')
        sizes = [inputs['input_ids'][i].shape[0] for i in range(len(inputs['input_ids']))]
        print(attr)
        print(sizes)
    print('finish')
#get_maxlen()


