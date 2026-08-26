import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import *
from transformers import AutoTokenizer, AutoModel, Qwen2Model, Qwen2Tokenizer

def load_data_t(args):
    print("args.root: ", args.root)
    
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item_time.json')
   
    item2feature = load_json(item2feature_path)

    return item2feature

def load_data_interval(args):
    print("args.root: ", args.root)
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item_interval.json')
    item2feature = load_json(item2feature_path)
    return item2feature

def load_data(args):
    print("args.root: ", args.root)
       
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    return item2feature

def generate_text(item2feature, features):
    item_text_list = []

    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())

        item_text_list.append([int(item), text])

    return item_text_list

def generate_text_time(item2feature, features):
    item_text_list = []

    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features[:2]:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())

        item_text_list.append([int(item), text, data[features[2]]])

    return item_text_list

def generate_text_interval(item2feature, features):
    """
    features: ['title', 'description', 'item_id', 'inter_time', 'interval']
    返回: [[idx, [title, description], item_id, inter_time, interval], ...]
    """
    item_text_list = []
    for idx in item2feature:
        data = item2feature[idx]
        text = []
        for meta_key in features[:2]:  # title, description
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())
        
        item_text_list.append([
            int(idx),
            text,
            data['item_id'],
            data['inter_time'],
            data['interval']
        ])
    return item_text_list

def preprocess_text_interval(args):
    print('Process interval data: ')
    print('Dataset: ', args.dataset)
    
    item2feature = load_data_interval(args)
    print(type(item2feature), len(item2feature))
    
    item_text_interval_list = generate_text_interval(
        item2feature, 
        ['title', 'description', 'item_id', 'inter_time', 'interval']
    )
    print(type(item_text_interval_list), len(item_text_interval_list), item_text_interval_list[0])
    return item_text_interval_list

def preprocess_text_time(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)

    item2feature = load_data_t(args)
    print(type(item2feature), len(item2feature))
    # load item text and clean
    item_text_time_list = generate_text_time(item2feature, ['title', 'description', 'inter_time'])
    print(type(item_text_time_list), len(item_text_time_list), item_text_time_list[0])
    # item_text_list = generate_text(item2feature, ['title'])
    # return: list of (item_ID, cleaned_item_text)
    return item_text_time_list

def preprocess_text(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)

    item2feature = load_data(args)
    print(type(item2feature), len(item2feature))
    # load item text and clean
    item_text_list = generate_text(item2feature, ['title', 'description'])
    print(type(item_text_list), len(item_text_list), item_text_list[0])
    # item_text_list = generate_text(item2feature, ['title'])
    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def random_time_encoding_single(num_items, embedding_dim, time_range=None, seed=None):
    """
    为每个item生成一个随机时间编码（unix时间戳格式）
    
    Args:
        num_items: item数量
        embedding_dim: 编码维度（768）
        time_range: (start_timestamp, end_timestamp)，默认2013-2018
        seed: 随机种子
    
    Returns:
        time_encodings: shape为 (num_items, embedding_dim) 的numpy数组
        random_timestamps: 生成的随机unix时间戳列表
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 默认时间范围：2013-01-01 到 2018-12-31（unix时间戳）
    if time_range is None:
        time_range = (1356998400, 1546214400)
    
    # 为每个item生成一个随机unix时间戳
    random_timestamps = np.random.randint(time_range[0], time_range[1], size=num_items)
    
    # 转换为日期（去掉时分秒，只保留日期）
    import datetime
    dates = []
    for ts in random_timestamps:
        dt = datetime.datetime.fromtimestamp(ts)
        # date_only = datetime.datetime(dt.year, dt.month, dt.day)
        date_ts = int(dt.timestamp())
        dates.append(date_ts)
    
    dates = np.array(random_timestamps).reshape(-1, 1)  # shape: (num_items, 1)
    print(dates)
    
    # 生成位置编码（与sinusoidal_time_encoding相同的逻辑）
    position = dates
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    
    time_encodings = np.zeros((num_items, embedding_dim))
    time_encodings[:, 0::2] = np.sin(position * div_term)  # 偶数维度
    time_encodings[:, 1::2] = np.cos(position * div_term)  # 奇数维度
    
    return time_encodings, random_timestamps.tolist()

def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding using Qwen: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 1
    with torch.no_grad():
        while start < len(order_texts):
            if (start+1)%100==0:
                print("==>",start+1)
            field_texts = order_texts[start: start + batch_size]
            # print(field_texts)
            field_texts = zip(*field_texts)
    
            field_embeddings = []
            for sentences in field_texts:
                sentences = list(sentences)
                # print(sentences)
                if word_drop_ratio > 0:
                    print(f'Word drop with p={word_drop_ratio}')
                    new_sentences = []
                    for sent in sentences:
                        new_sent = []
                        sent = sent.split(' ')
                        for wd in sent:
                            rd = random.random()
                            if rd > word_drop_ratio:
                                new_sent.append(wd)
                        new_sent = ' '.join(new_sent)
                        new_sentences.append(new_sent)
                    sentences = new_sentences
                
                # For Qwen, we need to handle tokenization differently
                encoded_sentences = tokenizer(sentences, max_length=args.max_sent_len,
                                              truncation=True, return_tensors='pt', padding="longest").to(args.device)
                
                # Get model outputs
                outputs = model(input_ids=encoded_sentences.input_ids,
                                attention_mask=encoded_sentences.attention_mask)
    
                # For Qwen models, use the last hidden state
                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                field_embeddings.append(mean_output)
    
            field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
            embeddings.append(field_mean_embedding)
            start += batch_size

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)
    # print(embeddings[0])


    # ============ 新增：为每个item生成一个随机时间编码 ============
    num_items = embeddings.shape[0]
    embedding_dim = 768  # 768
    
    # 生成随机时间编码（每个item一个）
    random_time_encs, random_timestamps = random_time_encoding_single(
        num_items=num_items,
        embedding_dim=embedding_dim,
        # time_range=(1380600000, 1541001600),  # 2013-2018
        time_range=(1682899200, 1698796800),    # 202305-202310
        seed=None  # 可以设置固定seed保证可复现
    )
    
    # 拼接：text_emb + random_time_emb
    combined_embeddings = np.concatenate([embeddings, random_time_encs], axis=1)  # shape: (num_items, 1536)
    
    print('Combined Embeddings shape (text+random_time): ', combined_embeddings.shape)
    print(f'Format: each row is [text_emb(2560) + random_time_emb(768)]')

    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
    # file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td-randompos" + ".npy")
    np.save(file, embeddings)

def sinusoidal_time_encoding(timestamps, embedding_dim):
    """
    对时间戳进行正余弦编码
    timestamps: unix时间戳列表
    embedding_dim: 编码维度（与文本embedding维度一致）
    返回: shape为 (len(timestamps), embedding_dim) 的numpy数组
    """
    import datetime
    
    # 将unix时间戳转换为日期（去掉时分秒）
    dates = []
    for ts in timestamps:
        dt = datetime.datetime.fromtimestamp(ts)
        # 只保留日期，将时分秒设为0
        # date_only = datetime.datetime(dt.year, dt.month, dt.day)
        # 转换回时间戳（这样同一天的所有时间戳都相同）
        date_ts = int(dt.timestamp())
        dates.append(date_ts)
    
    dates = np.array(dates).reshape(-1, 1)  # shape: (n, 1)
    
    # 生成位置编码
    position = dates
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    
    # 创建编码矩阵
    time_encoding = np.zeros((len(timestamps), embedding_dim))
    time_encoding[:, 0::2] = np.sin(position * div_term)  # 偶数维度用sin
    time_encoding[:, 1::2] = np.cos(position * div_term)  # 奇数维度用cos
    
    return time_encoding

def generate_item_time_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding using Qwen: ')
    print(' Dataset: ', args.dataset)

    # 新格式：[item_id, [text1, text2, ...], [time1, time2, ...]]
    items, texts, time_lists = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]
    
    order_times = [[]] * len(items)
    for item, times in zip(items, time_lists):
        order_times[item] = times

    expanded_embeddings = []  # 存储格式：[text_emb, time_emb, item_id, time_idx, unix_time]

    start, batch_size = 0, 1
    with torch.no_grad():
        while start < len(order_texts):
            if (start+1)%100==0:
                print("==>",start+1)
            field_texts = order_texts[start: start + batch_size]
            # print(field_texts)
            field_texts = zip(*field_texts)
    
            field_embeddings = []
            for sentences in field_texts:
                sentences = list(sentences)
                # print(sentences)
                if word_drop_ratio > 0:
                    print(f'Word drop with p={word_drop_ratio}')
                    new_sentences = []
                    for sent in sentences:
                        new_sent = []
                        sent = sent.split(' ')
                        for wd in sent:
                            rd = random.random()
                            if rd > word_drop_ratio:
                                new_sent.append(wd)
                        new_sent = ' '.join(new_sent)
                        new_sentences.append(new_sent)
                    sentences = new_sentences
                
                # For Qwen, we need to handle tokenization differently
                encoded_sentences = tokenizer(sentences, max_length=args.max_sent_len,
                                              truncation=True, return_tensors='pt', padding="longest").to(args.device)
                
                # Get model outputs
                outputs = model(input_ids=encoded_sentences.input_ids,
                                attention_mask=encoded_sentences.attention_mask)
    
                # For Qwen models, use the last hidden state
                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                field_embeddings.append(mean_output)
    
            
            field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)  # shape: (batch_size, hidden_dim)
            text_embedding = field_mean_embedding.numpy()[0].tolist()  # shape: (hidden_dim,)

            # ============ 新增：处理当前item的所有时间 ============
            current_item_id = start  # 当前item的ID
            current_times = order_times[current_item_id]  # 当前item的所有时间戳

            if len(current_times) > 0:
                # 生成时间编码
                embedding_dim = 768
                time_embeddings = sinusoidal_time_encoding(current_times, embedding_dim)
                
                # 为每个时间创建一行
                for time_idx, (time_emb, unix_time) in enumerate(zip(time_embeddings, current_times), 1):
                    # 格式: [text_emb, time_emb, item_id, time_idx, unix_time]
                    row = [
                        text_embedding,           # 文本embedding
                        time_emb.tolist(),                 # 时间embedding
                        int(current_item_id),        # item ID
                        int(time_idx),               # 第几个时间（从1开始）
                        int(unix_time)               # unix时间戳
                    ]
                    expanded_embeddings.append(row)
            else:
                raise ValueError('no time information!')

            start += batch_size

    embeddings = np.array(expanded_embeddings, dtype = object)
    print('Embeddings shape: ', embeddings.shape)
    # print(embeddings[0])
    print(f'Each row format: [text_emb(list of {len(text_embedding)}), time_emb(list of {embedding_dim}), item_id(int), time_idx(int), unix_time(int)]')
    print(f'Total rows: {embeddings.shape[0]} (expanded from {len(order_texts)} items)')
    print(f'Example first row types: text_emb type={type(embeddings[0][0])}, time_emb type={type(embeddings[0][1])}, item_id type={type(embeddings[0][2])}')


    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td-time" + ".npy")
    np.save(file, embeddings)
    print(f'Saved to: {file}')

def generate_item_interval_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    """
    输入格式: [[idx, [title, description], item_id, inter_time, interval], ...]
    输出格式: [text_emb, time_emb, item_id, idx, unix_time]
    """
    print(f'Generate Interval Embedding using Qwen: ')
    print(' Dataset: ', args.dataset)

    # 按 idx 排序
    item_text_list.sort(key=lambda x: x[0])
    
    expanded_embeddings = []

    item_id_to_texts = {}
    for idx, texts, item_id, inter_time, interval in item_text_list:
        if item_id not in item_id_to_texts:
            item_id_to_texts[item_id] = texts
    print(f'Total records: {len(item_text_list)}')
    print(f'Unique items: {len(item_id_to_texts)}')
    print(f'Speedup ratio: {len(item_text_list) / len(item_id_to_texts):.2f}x')


    # 步骤2: 为每个唯一的 item_id 生成文本embedding (只计算一次)
    item_id_to_embedding = {}
    with torch.no_grad():
        for i, (item_id, texts) in enumerate(item_id_to_texts.items()):
            if (i+1) % 100 == 0:
                print(f"==> Computing embeddings for unique items: {i+1}/{len(item_id_to_texts)}")
            
            # 生成文本 embedding (保持原有逻辑)
            field_embeddings = []
            for sentence in texts:
                if not sentence:
                    continue
                encoded = tokenizer([sentence], max_length=args.max_sent_len,
                                truncation=True, return_tensors='pt', padding="longest").to(args.device)
                outputs = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
                masked_output = outputs.last_hidden_state * encoded['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded['attention_mask'].sum(dim=-1, keepdim=True)
                field_embeddings.append(mean_output.detach().cpu())
            
            if field_embeddings:
                text_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0).numpy()[0].tolist()
            else:
                text_embedding = [0.0] * 2560
            
            # 缓存该 item_id 的文本embedding
            item_id_to_embedding[item_id] = text_embedding
    print(f'Finished computing {len(item_id_to_embedding)} unique text embeddings')
    
    
    for i, (idx, texts, item_id, inter_time, interval) in enumerate(item_text_list):
        if (i+1) % 100 == 0:
            print("==>", i+1)
        
        # 从缓存中获取文本embedding (复用!)
        text_embedding = item_id_to_embedding[item_id]
        
        # 生成 interval 的时间编码
        embedding_dim = 768
        time_emb = sinusoidal_time_encoding([interval], embedding_dim)[0].tolist()
        
        # 格式: [text_emb, time_emb, item_id, idx, unix_time]
        row = [
            text_embedding,
            time_emb,
            int(item_id),
            int(idx),
            int(inter_time)
        ]
        expanded_embeddings.append(row)
    
    embeddings = np.array(expanded_embeddings, dtype=object)
    print('Embeddings shape: ', embeddings.shape)
    print(f'Each row format: [text_emb(list of {len(text_embedding)}), time_emb(list of {len(time_emb)}), item_id(int), idx(int), unix_time(int)]')
    
    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td-interval-768.npy")
    np.save(file, embeddings)
    print(f'Saved to: {file}')

def load_qwen_model(model_path):
    """Load Qwen model and tokenizer"""
    print("Loading Qwen Model:", model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model


def set_device(gpu_id):
    """Set device for model"""
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Office', help='Beauty / Sports / Toys')
    # parser.add_argument('--root', type=str, default="data/Amazon/index")
    parser.add_argument('--root', type=str, default="/data1/dnian/OneDiffRec/data/Amazon18/Office")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='qwen')
    parser.add_argument('--plm_checkpoint', type=str,
                        default='emb_model/Qwen3-Embedding-4B', help='Qwen model path')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    # parser.add_argument('--use_time', type=bool, default = False, help='whether use time features')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    args.device = device

    item_text_list = preprocess_text(args)
    # item_text_list[0]
    # [0, ['SUPCO SPP6 Relay/Capacitor Hard Start Kit with 500% Increase Starting Torque.', 
    # "['Relay/CAPACITOR hard start kit 500% incr starting torque,increased compressor starting torque. Designed for use on permanent split CAPACITOR single phase a/c and heat pump systems. 
    # Multiple use includes: room air conditioner units, residential or commercial psc AC units.. the country of origin is china.']."]
    # ]

    # item_text_time_list = preprocess_text_time(args)
    # item_text_list[0]
    # [0, ['SUPCO SPP6 Relay/Capacitor Hard Start Kit with 500% Increase Starting Torque.', 
    # "['Relay/CAPACITOR hard start kit 500% incr starting torque,increased compressor starting torque. Designed for use on permanent split CAPACITOR single phase a/c and heat pump systems. 
    # Multiple use includes: room air conditioner units, residential or commercial psc AC units.. the country of origin is china.']."]
    # [1411603200, 1412640000, 1428278400, 1463270400, 1464566400, 1469318400]
    # ]
    

    # Load Qwen model and tokenizer
    plm_tokenizer, plm_model = load_qwen_model(args.plm_checkpoint)
    
    # Set pad token if not exists
    if plm_tokenizer.pad_token_id is None:
        if plm_tokenizer.eos_token_id is not None:
            plm_tokenizer.pad_token_id = plm_tokenizer.eos_token_id
        else:
            plm_tokenizer.pad_token_id = 0
    
    plm_model = plm_model.to(device)
    plm_model.eval()  # Set model to evaluation mode

    generate_item_embedding(args, item_text_list, plm_tokenizer,
                            plm_model, word_drop_ratio=args.word_drop_ratio)

    # generate_item_time_embedding(args, item_text_time_list, plm_tokenizer,
    #                         plm_model, word_drop_ratio=args.word_drop_ratio)


    # item_text_interval_list = preprocess_text_interval(args)
    # generate_item_interval_embedding(args, item_text_interval_list, plm_tokenizer,
    #                                  plm_model, word_drop_ratio=args.word_drop_ratio)


