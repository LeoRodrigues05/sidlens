import collections
import json
import logging
import sys

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset, TimeEmbDataset
from models.rqvae import RQVAE, MLPRQVAE, CrossAttnRQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--num_codebooks", type=int, required=True)
parser.add_argument("--codebook_size", type=int, required=True)
parser.add_argument("--dataset", type=str, default="Industrial_and_Scientific")
parser.add_argument("--device", type=str, default="cuda:0")
gen_args = parser.parse_args()

dataset = gen_args.dataset
ckpt_path = gen_args.ckpt_path
output_dir = f"../data/Amazon18/{dataset}/"
output_file = f"{dataset}.index_{gen_args.num_codebooks}codebook_{gen_args.codebook_size}.json"
output_file = os.path.join(output_dir, output_file)
device = torch.device(gen_args.device)
print(ckpt_path)

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

print(args.data_path)
# sys.exit()


data = EmbDataset(args.data_path)

common_kwargs = dict(
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

if hasattr(args, 'model_type'):
    if args.model_type == 'mlp_rqvae':
        model = MLPRQVAE(in_dim=data.dim, fusion_layers=args.fusion_layers, **common_kwargs)
    elif args.model_type == 'cross_attn_rqvae':
        model = CrossAttnRQVAE(text_dim=args.text_dim, time_dim=data.dim - args.text_dim,
                               num_heads=args.num_heads, **common_kwargs)
    else:
        model = RQVAE(in_dim=data.dim, **common_kwargs)
else:
    model = RQVAE(in_dim=data.dim, **common_kwargs)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)

        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)


all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)



with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
