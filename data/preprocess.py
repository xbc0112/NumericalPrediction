# Preprocessing the two datasets
# Including: normalization, de-duplication, shuffle and split

from random import shuffle

def preprocess_FB():
    fin = open('./FB15K/FB15K_NumericalTriples.txt','r',encoding='utf-8')
    # count & normalization & de-duplication
    edic = {}
    pdic = {}
    dic = {}
    for line in fin:
        tris = line[:-3].replace('<http://rdf.freebase.com/ns/','').replace('>','').split('\t')
        if(len(tris) != 3):
            print(tris)
            continue
        edic.setdefault(tris[0],0)
        pdic.setdefault(tris[1],0)
        edic[tris[0]] += 1
        pdic[tris[1]] += 1
        tstr = tris[0]+' '+tris[1]+' '+tris[2]+'\n'
        if tstr not in dic:
            dic[tstr] = 1
    #print(pdic)
    print(len(edic), len(pdic), len(dic)) # 12493,116,29395
    with open('./FB15K/FB15K_NumericalTriples_handled_rdup.txt','w',encoding='utf-8') as f:
        for k in dic:
            f.write(k)
    fin.close()
    
    # shuffle & split
    dic = list(dic.keys())
    cnt = len(dic)
    shuffle(dic)
    i = 0
    with open('./FB15K/test.txt', 'w', encoding='utf-8') as f:
        while i < cnt * 0.1:
            f.write(dic[i])
            i += 1
    with open('./FB15K/valid.txt', 'w', encoding='utf-8') as f:
        while i < cnt * 0.2:
            f.write(dic[i])
            i += 1
    with open('./FB15K/train.txt', 'w', encoding='utf-8') as f:
        while i < cnt:
            f.write(dic[i])
            i += 1
    
    # handle relational triples
    fin = open('./FB15K/FB15K_EntityTriples.txt','r',encoding='utf-8')
    fout = open('./FB15K/FB15K_EntityTriples_handled.txt','w',encoding='utf-8')
    sdic = {}
    pdic = {}
    odic = {}
    for line in fin:
        tris = line[:-3].split(' ')
        if(len(tris) != 3):
            print(tris)
            continue
        sdic.setdefault(tris[0],0)
        pdic.setdefault(tris[1],0)
        odic.setdefault(tris[2],0)
        sdic[tris[0]] += 1
        pdic[tris[1]] += 1
        odic[tris[2]] += 1
        fout.write(tris[0]+' '+tris[1]+' '+tris[2]+'\n')
    fin.close()
    fout.close()
    print(len(sdic), len(pdic), len(odic)) # 14866,1345,14931
    print(len(list(set(sdic.keys()).difference(set(odic.keys()))))) # 20
    print(len(list(set(odic.keys()).difference(set(sdic.keys()))))) # 85
    print('finish')
#preprocess_FB()

def preprocess_YAGO():
    fin = open('./YAGO15K/YAGO15K_NumericalTriples.txt','r',encoding='utf-8')
    # count
    edic = {}
    pdic = {}
    tdic = {}
    for line in fin:
        tris = line[:-3].split(' ')
        if(len(tris) != 3):
            print(tris)
            continue
        edic.setdefault(tris[0],0)
        pdic.setdefault(tris[1],0)
        tmp = tris[2].split('^^')[1]
        tdic.setdefault(tmp,0)
        edic[tris[0]] += 1
        pdic[tris[1]] += 1
        tdic[tmp] += 1
    print(pdic)
    print(tdic)
    print(len(edic), len(pdic), len(tdic))

    # normalization
    fout = open('./YAGO15K/YAGO15K_Numerical_handled.txt','w',encoding='utf-8')
    for line in fin:
        tris = line[:-3].split(' ')
        s = tris[0][:-1].split('resource/')[-1]
        p = tris[1][:-1].split('resource/')[-1]
        # unknown '#' is replaced into '0'
        o = tris[2].split('^^')[0].replace('"','').replace('#','0')
        if '-' in o[1:]:
            os = o[1:].split('-')
            vstr = o[0] + os[0] + '.'
            for oo in os[1:]:
                vstr += oo
            v = float(vstr)
        else:
            v = float(o)
        fout.write(s+' '+p+' '+str(v)+'\n')
    fout.close()
    fin.close()

    # de-duplication
    fin = open('./YAGO15K/YAGO15K_Numerical_handled.txt','r',encoding='utf-8')
    fout = open('./YAGO15K/YAGO15K_Numerical_handled_rdup.txt','w',encoding='utf-8')
    dic = {}
    for line in fin:
        pre = line[:-1].rsplit(' ',1)[0]
        if pre not in dic:
            dic[pre] = 1
            fout.write(line)
    fout.close()
    fin.close()

    # shuffle & split
    fin = open('./YAGO15K/YAGO15K_Numerical_handled_rdup.txt','r',encoding='utf-8')
    dic = []
    cnt = 0
    for line in fin:
        dic.append(line)
        cnt += 1
    shuffle(dic)
    i = 0
    with open('./YAGO15K/test.txt','w',encoding='utf-8') as f:
        while i < cnt*0.1:
            f.write(dic[i])
            i += 1
    with open('./YAGO15K/valid.txt','w',encoding='utf-8') as f:
        while i < cnt*0.2:
            f.write(dic[i])
            i += 1
    with open('./YAGO15K/train.txt','w',encoding='utf-8') as f:
        while i < cnt:
            f.write(dic[i])
            i += 1
    fin.close()

    # handle relational triples
    fin = open('./YAGO15K/YAGO15K_EntityTriples.txt','r',encoding='utf-8')
    fout = open('./YAGO15K/YAGO15K_EntityTriples_handled.txt','w',encoding='utf-8')
    sdic = {}
    pdic = {}
    odic = {}
    for line in fin:
        tris = line[:-3].replace('<http://yago-knowledge.org/resource/','').replace('>','').split(' ')
        if(len(tris) != 3):
            print(tris)
            continue
        sdic.setdefault(tris[0],0)
        pdic.setdefault(tris[1],0)
        odic.setdefault(tris[2],0)
        sdic[tris[0]] += 1
        pdic[tris[1]] += 1
        odic[tris[2]] += 1
        fout.write(tris[0]+' '+tris[1]+' '+tris[2]+'\n')
    fin.close()
    fout.close()
    print(len(sdic), len(pdic), len(odic))
    print(len(list(set(sdic.keys()).difference(set(odic.keys())))))
    print(len(list(set(odic.keys()).difference(set(sdic.keys())))))
    print('finish')
#preprocess_YAGO()


# check duplication
def check_dup():
    fin = open('./YAGO15K/train.txt','r',encoding='utf-8')
    dic = {}
    cnt = 0
    for line in fin:
        if line in dic:
            dic[line] += 1
        else:
            dic[line] = 1
        cnt += 1
        # pre = line[:-1].rsplit(' ',1)
        # if pre[0] in dic:
        #     dic[pre[0]].append(pre[1])
        # else:
        #     dic[pre[0]] = [pre[1]]
    fin.close()

    print(cnt)
    print(len(dic))
#check_dup()
