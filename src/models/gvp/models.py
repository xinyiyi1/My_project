import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
import torch.utils.checkpoint as checkpoint
import sys
from torch_geometric.nn import aggr
from torch.nn import MultiheadAttention
from src.feature_processor import FeatureProcessor


class SSEmbGNN(torch.nn.Module):
    '''
    GVP-GNN for structure-conditioned autoregressive
    protein design as described in manuscript.

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 20 amino acids at each position in a `torch.Tensor` of
    shape [n_nodes, 20].

    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.

    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''

    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim, feat_processor = None,
                 num_layers=4, drop_rate=0.0, vector_gate=True):
        super(SSEmbGNN, self).__init__()
        self.feat_processor = feat_processor
        self.node_h_dim = node_h_dim
        # 打印参数总数验证
        total_params = sum(p.numel() for p in self.parameters())
        featproc_params = sum(p.numel() for p in self.feat_processor.parameters())
        print(f"Total params: {total_params}, FeatProcessor params: {featproc_params}")
        # 原始GVP编码器
        self.W_v = nn.Sequential(
            GVP(
                node_in_dim,
                node_h_dim,
                activations=(None, None),
                vector_gate=vector_gate
            ),
            LayerNorm(node_h_dim)
        )

        self.msa_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        self.esm2_proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        self.W_e = nn.Sequential(
            GVP(
                edge_in_dim,
                edge_h_dim,
                activations=(None, None),
                vector_gate=vector_gate
            ),
            LayerNorm(edge_h_dim)
        )
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim,
                         drop_rate=drop_rate,
                         vector_gate=vector_gate)
            for _ in range(num_layers)
        )
        # 多模态融合模块
        self.esm2_dim = 1280
        self.msa_dim = 768
        self.seq_feature_fusion = nn.Sequential(
            nn.Linear(self.esm2_dim + self.msa_dim, node_h_dim[0]),
            LayerNorm((node_h_dim[0], node_h_dim[1])),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.seq_fusion = FeatureProcessor()  # 融合ESM-2和MSA
        self.struct_fusion = nn.MultiheadAttention(256, num_heads=8)
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=drop_rate
        )
        # 结构特征增强
        self.struct_proj = nn.Sequential(
            GVP(node_h_dim, node_h_dim,
                activations=(F.relu, None)),
            LayerNorm(node_h_dim)
        )
        # 动态门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Sigmoid()
        )
        # 解码器增强
        self.W_S = nn.Embedding(21, 21)
        # self.W_decoder_in = nn.Sequential(
        #     nn.Linear(512, node_h_dim[0]),  # 输入维度512 → 输出256
        #     #LayerNorm((node_h_dim[0], node_h_dim[1])),
        #     nn.ReLU(),
        #     nn.Dropout(drop_rate)
        # )
        if feat_processor is None:
            self.feat_processor = FeatureProcessor(
                esm_dim=1280,
                msa_dim=768,
                embed_dim=256,
                num_heads=8
            )
        else:
            self.feat_processor = feat_processor
        # Decode

        edge_h_dim = (edge_h_dim[0] + 21, edge_h_dim[1])
        self.node_embedding = nn.Sequential(
            LayerNorm((node_h_dim[0] * 2, node_h_dim[1])),
        )
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim,
                         drop_rate=drop_rate, vector_gate=vector_gate)
            for _ in range(num_layers))

        # Out
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None), vector_gate=vector_gate)

    def forward(self, h_V, edge_index, h_E, seq, fused_feats, esm2_feats, msa_feats):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.

        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''

        # 输入校验
        assert h_V[0].size(0) == len(seq), (
            f"结构特征长度{h_V[0].shape[0]}与序列长度{len(seq)}不匹配！"
            f"检查蛋白数据是否存在问题。"
        )


        h_V = self.W_v(h_V)
        print(f"初始 h_V type: {type(h_V)}, h_V[0].shape: {h_V[0].shape}, h_V[1].shape: {h_V[1].shape}")
        struct_scalar, struct_vector = h_V

        struct_feats = h_V[0]  # [N, D]

        # 序列特征融合（通过FeatureProcessor）
        fused_seq = self.feat_processor(esm2_feats, msa_feats)  # [L, 256]

        # 结构特征与序列特征的跨模态交互
        struct_feats = struct_feats.unsqueeze(1)  # [N, 1, D]
        fused_seq = fused_seq.unsqueeze(1)  # [L, 1, D]

        # 调整维度为 [SeqLen, Batch=1, Dim]
        struct_feats = struct_feats.permute(1, 0, 2)  # [1, N, D]
        fused_seq = fused_seq.permute(1, 0, 2)  # [1, L, D]

        # 跨模态注意力
        attn_output, _ = self.cross_attn(
            query=struct_feats,
            key=fused_seq,
            value=fused_seq
        )
        attn_output = attn_output.squeeze(0)  # [N, D]

        # 动态门控融合
        gate = self.fusion_gate(torch.cat([h_V[0], attn_output], dim=-1))
        h_V_scalar = gate * h_V[0] + (1 - gate) * attn_output
        h_V = (h_V_scalar, h_V[1])


        # 原有GVP处理流程
        h_E = self.W_e(h_E)
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)



        # 确保 fused_feats 为2D
        fused_feats = fused_feats.squeeze()

        struct_vector_norm = torch.norm(struct_vector, dim=-1).mean(1)





        # Add sequence info
        h_S = self.W_S(seq)
        h_S = h_S[edge_index[0]]
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        # 解码器输入拼接（维度校验）
        scalar_feats = torch.cat([h_V[0], fused_feats], dim=-1)
        processed_scalar = self.W_decoder_in[0](scalar_feats)
        h_V = (processed_scalar, h_V[1])

        # 调试打印
        print(f"解码器输入后 h_V[0].shape: {h_V[0].shape}, h_V[1].shape: {h_V[1].shape}")
        assert h_V[0].size(0) == len(seq), f"Structure-seq mismatch: {h_V[0].shape[0]} vs {len(seq)}"

        # 解码器处理
        # for layer in self.decoder_layers:
        #     h_V = layer(h_V, edge_index, h_E)
        #
        # return self.W_out(h_V)
        logits = self.W_out(h_V)
        assert logits.size(0) == len(seq), "模型输出维度与序列长度不一致"

        return logits