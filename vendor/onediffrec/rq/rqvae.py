import argparse
import random
import torch
import numpy as np
from time import time
import logging
import sys

from torch.utils.data import DataLoader

from datasets import EmbDataset, TimeEmbDataset
from models.rqvae import RQVAE, MLPRQVAE, CrossAttnRQVAE
from trainer import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy",
                        # default="../data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy",
                        help="Input data path.")

    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0,0.0,0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:5", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=512, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument('--model_type', type=str, default='rqvae',
                        choices=['rqvae', 'mlp_rqvae', 'cross_attn_rqvae'])
    parser.add_argument('--fusion_layers', type=int, nargs='+', default=[1024],
                        help='pre-fusion MLP hidden dims for mlp_rqvae, e.g. 2048 1024')
    parser.add_argument('--text_dim', type=int, default=2560,
                        help='text embedding dim for cross_attn_rqvae')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of attention heads for cross_attn_rqvae')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")

    parser.add_argument("--resume_path", type=str, default="", help="path to the checkpoint to resume from")

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")
    print(args.data_path)

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    data = EmbDataset(args.data_path)
    # data = TimeEmbDataset(args.data_path)

    common_kwargs = dict(
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    if args.model_type == 'mlp_rqvae':
        model = MLPRQVAE(in_dim=data.dim, fusion_layers=args.fusion_layers, **common_kwargs)
    elif args.model_type == 'cross_attn_rqvae':
        model = CrossAttnRQVAE(text_dim=args.text_dim, time_dim=data.dim - args.text_dim,
                               num_heads=args.num_heads, **common_kwargs)
    else:
        model = RQVAE(in_dim=data.dim, **common_kwargs)
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)


    # print("Checking DataLoader contents:")
    # for i, batch in enumerate(data_loader):
    #     print(f"\n=== Batch {i} ===")
    #     print(f"Batch shape: {batch.shape}")  # (batch_size, combined_dim)
        
    #     # 假设text_emb和time_emb维度相同，都是总维度的一半
    #     print(batch)
        
        
    #     # 只打印前几个batch
    #     if i >= 2:
    #         print("\n... (showing only first 3 batches)")
    #         break


    trainer = Trainer(args,model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)

