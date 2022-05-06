# Auxiliary functions for KGE-reg and PLM-reg.


# add YGAO15K entities to vocab by the SameAs links
def add_word():
    dir = '../pretrainedModels/FB15K/'
    e2id = {}
    with open(dir + '/entity_ids0.del', 'r') as f:
        for line in f:
            tris = line[:-1].split()
            e2id[tris[1]] = tris[0]
    print(len(e2id))

    fout = open(dir+'/entity_ids.del','a', encoding='utf-8')
    cnt = 0
    with open('./YAGO15K_SameAsLink.txt','r') as f:
        for line in f:
            tris = line[:-1].split()
            e_yago = tris[2][1:-1].rsplit('/',1)[1]
            if tris[0] in e2id:
                cnt += 1
                fout.write(e2id[tris[0]]+'  '+e_yago+'\n')
    fout.close()
    print(cnt) # 11142
#add_word()

# get all attributes and count
def get_attr():
    dir = '../data/FB15K/'
    attr_dic = {}
    for file in ['train.txt','valid.txt','test.txt']:
        with open(dir+file, 'r', encoding='utf-8') as f:
            for line in f:
                tris = line[:-1].split(' ')
                if tris[1] in attr_dic:
                    attr_dic[tris[1]] += 1
                else:
                    attr_dic[tris[1]] = 1

    print(len(attr_dic))
    sorted_dic = sorted(attr_dic.items(), key=lambda kv : (kv[1],kv[0]))
    print(sorted_dic[100:])
#get_attr()

# get YAGO15K description texts by the SameAs links
def get_yago_description():
    fin = open('./YAGO15K_SameAsLink.txt','r',encoding='utf-8')
    yago2fb = {}
    fb2yago = {}
    for line in fin:
        tris = line[:-1].split(' ')
        yago_name = tris[2][1:-1].split('resource/',1)[1]
        yago2fb[yago_name] = tris[0]
        fb2yago[tris[0]] = yago_name
    fin.close()
    print(len(yago2fb), len(fb2yago))

    for language in ['English','French','German']:
        fin = open('./FB15K_Description_'+language+'.txt', 'r', encoding='utf-8')
        fout = open('./YAGO15K_Description_'+language+'.txt', 'w', encoding='utf-8')
        cnt = 0
        for line in fin:
            tris = line.split('\t')
            if tris[0] in fb2yago:
                cnt += 1
                fout.write(fb2yago[tris[0]] + '\t' + tris[1])
        fout.close()
        fin.close()
        print(language, cnt) 

    fin = open('./FB15K_mid2description.txt', 'r', encoding='utf-8')
    fout = open('./YAGO15K_mid2description.txt', 'w', encoding='utf-8')
    cnt = 0
    for line in fin:
        tris = line.split('\t')
        if tris[0] in fb2yago:
            cnt += 1
            fout.write(fb2yago[tris[0]] + '\t' + tris[1])
    fout.close()
    fin.close()
    print('mid2description: ',cnt) # 11174
#get_yago_description()

# count description nums
# iswc20: 1,301 missing
# aaai16: 29 missing
def count():
    entity = []
    with open('../results/FB15K/regression/bert-base-uncased/entitys.txt','r',encoding='utf-8') as f:
        for line in f:
            entity.append(line[:-1])
    print(len(entity))

    e2desc = {}
    with open('FB15K_Description_English.txt','r',encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split('\t',1)
            e2desc[tris[0]] = 1
    print(len(e2desc))

    cnt = 0
    for e in entity:
        if e not in e2desc:
            cnt += 1
    print(f"{cnt} entities don't have desc in FB15K_Description_English.")
    print('----------')

    e2desc = {}
    with open('FB15K_mid2description.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tris = line[:-1].split('\t', 1)
            e2desc[tris[0]] = 1
    print(len(e2desc))

    cnt = 0
    for e in entity:
        if e not in e2desc:
            cnt += 1
    print(f"{cnt} entities don't have desc in FB15K_mid2description.")
#count()
