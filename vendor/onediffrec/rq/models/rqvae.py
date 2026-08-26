import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon


class MLPRQVAE(nn.Module):
    """
    在进入 encoder 之前，先将拼接好的 [text_emb || time_emb] 过一个 pre-fusion MLP，
    再送入 encoder -> RQ -> decoder。decoder 重建原始拼接向量。
    """
    def __init__(self,
                 in_dim=3328,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 fusion_layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
    ):
        super(MLPRQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # pre-fusion MLP: [in_dim] + fusion_layers, 例如 fusion_layers=[2048,1024]
        assert fusion_layers is not None and len(fusion_layers) > 0, \
            "fusion_layers must be a non-empty list, e.g. [2048, 1024]"
        self.pre_fusion = MLPLayers(layers=[in_dim] + fusion_layers,
                                    dropout=dropout_prob, bn=bn)
        encoder_in_dim = fusion_layers[-1]

        self.encode_layer_dims = [encoder_in_dim] + layers + [e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=dropout_prob, bn=bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=beta,
                                          kmeans_init=kmeans_init,
                                          kmeans_iters=kmeans_iters,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters)

        # decoder: e_dim -> ... -> in_dim (重建原始拼接向量)
        decode_layer_dims = self.encode_layer_dims[::-1][:-1] + [in_dim]
        self.decoder = MLPLayers(layers=decode_layer_dims,
                                 dropout=dropout_prob, bn=bn)

    def forward(self, x, use_sk=True):
        x = self.pre_fusion(x)
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x = self.pre_fusion(xs)
        x_e = self.encoder(x)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon


class CrossAttnRQVAE(nn.Module):
    """
    text_emb 作为 Query，time_emb 作为 Key/Value，通过 cross-attention 融合后
    送入 encoder -> RQ -> decoder。decoder 重建原始拼接向量 [text_emb || time_emb]。

    使用 nn.MultiheadAttention(embed_dim=text_dim, kdim=time_dim, vdim=time_dim)，
    PyTorch 内部自动处理 Q/K/V 的投影，无需手动加投影层。
    """
    def __init__(self,
                 text_dim=2560,
                 time_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 num_heads=8,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
    ):
        super(CrossAttnRQVAE, self).__init__()

        self.text_dim = text_dim
        self.time_dim = time_dim
        self.in_dim = text_dim + time_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # cross-attention: Q=text_emb, K=V=time_emb
        # PyTorch 内部处理三个投影矩阵，text_dim != time_dim 时指定 kdim/vdim 即可
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            kdim=time_dim,
            vdim=time_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.attn_norm = nn.LayerNorm(text_dim)

        # encoder 以 text_dim 为输入（cross-attn 输出维度 = embed_dim = text_dim）
        self.encode_layer_dims = [text_dim] + layers + [e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=dropout_prob, bn=bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=beta,
                                          kmeans_init=kmeans_init,
                                          kmeans_iters=kmeans_iters,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters)

        # decoder: e_dim -> ... -> in_dim (重建原始拼接向量)
        decode_layer_dims = self.encode_layer_dims[::-1][:-1] + [self.in_dim]
        self.decoder = MLPLayers(layers=decode_layer_dims,
                                 dropout=dropout_prob, bn=bn)

    def _fuse(self, x):
        text_emb = x[:, :self.text_dim]           # (B, text_dim)
        time_emb = x[:, self.text_dim:]           # (B, time_dim)
        # 添加 seq 维度: (B, 1, dim)
        q = text_emb.unsqueeze(1)
        kv = time_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, text_dim)
        attn_out = attn_out.squeeze(1)             # (B, text_dim)
        # residual + LayerNorm
        return self.attn_norm(attn_out + text_emb)

    def forward(self, x, use_sk=True):
        x = self._fuse(x)
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x = self._fuse(xs)
        x_e = self.encoder(x)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon