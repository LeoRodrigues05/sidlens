# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math
import json
import pandas as pd
import numpy as np
    
from tqdm import tqdm
def gao(path, item_path):
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC=0
        
    
    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    # item_names = [ _[:-len(_.split('\t')[-1])].strip() for _ in items]
    item_names= [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    # item_dict = {
    # '<a_8><b_129><c_231>': [0],           # 键: semantic_id, 值: [行索引]
    # '<a_8><b_18><c_149>': [1],
    # '<a_47><b_251><c_217>': [2, 4],       # 同一个semantic_id出现在第2行和第4行
    # '<a_47><b_251><c_109>': [3]
    
    
    # 读取 info 文件，格式：semantic_id \t title \t old_item_id
    sid_to_title = {}      # semantic_id -> title
    sid_to_itemid = {}     # semantic_id -> old_item_id
    title_to_sids = {}     # title -> set of semantic_ids (同一title可能有多个sid)
    itemid_to_sids = {}    # old_item_id -> set of semantic_ids

    for line in items:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            sid = parts[0].strip()
            title = parts[1].strip()
            item_id = parts[2].strip()
            
            sid_to_title[sid] = title
            sid_to_itemid[sid] = item_id
            
            if title not in title_to_sids:
                title_to_sids[title] = set()
            title_to_sids[title].add(sid)
            
            if item_id not in itemid_to_sids:
                itemid_to_sids[item_id] = set()
            itemid_to_sids[item_id].add(sid)


    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        # text = [                  # 617 个样本的预测列表
        #     ["pred1", "pred2", ..., "pred50"],  # 样本0的预测
        #     ["pred1", "pred2", ..., "pred50"],  # 样本1的预测
        #     ...
        # ]
        
        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            if type(test_data[index]['output']) == list:
                target_item = test_data[index]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]['output'].strip(" \n\"")
            minID = 1000000
            # for i in range(len(sample)):
                
            #     if sample[i] not in item_dict:
            #         CC += 1
            #         print(sample[i])
            #         print(target_item)
            #     if sample[i] == target_item:
            #         minID = i
            #         break
            for i in range(len(sample)):
                matched = False

                # 1. 先尝试精确匹配 semantic_id
                if sample[i] == target_item:
                    matched = True
                    minID = i

                # 2. 如果精确匹配失败，尝试通过 title 匹配
                elif sample[i] in sid_to_title and target_item in sid_to_title:
                    pred_title = sid_to_title[sample[i]]
                    target_title = sid_to_title[target_item]
                    if pred_title == target_title:
                        matched = True
                        minID = i

                # 3. 如果 title 匹配失败，尝试通过 old_item_id 匹配
                elif sample[i] in sid_to_itemid and target_item in sid_to_itemid:
                    pred_itemid = sid_to_itemid[sample[i]]
                    target_itemid = sid_to_itemid[target_item]
                    if pred_itemid == target_itemid:
                        matched = True
                        minID = i

                # 4. 记录无法匹配的情况
                if not matched and sample[i] not in sid_to_title:
                    CC += 1
                    print(sample[i])
                    print(target_item)

                if matched:
                    break
            
            for index, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                    ALLHR[index] = ALLHR[index] + 1
        print(n_beam)
        valid_topk = [k for k in topk_list if k <= n_beam]
        print(valid_topk)
        print(f"NDCG:\t{ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR\t{ALLHR / len(text)}")
        print(CC)

if __name__=='__main__':
    fire.Fire(gao)
