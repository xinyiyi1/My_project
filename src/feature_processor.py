import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class FeatureProcessor(nn.Module):
    def __init__(self, esm_dim=1280, msa_dim=768, embed_dim=256, num_heads=8):
        super().__init__()

        # ESM2特征投影
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # MSA特征投影
        self.msa_proj = nn.Sequential(
            nn.Linear(msa_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1))

        # 跨模态注意力（
        self.cross_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=False
        )

        # 动态门控机制
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, esm_feats, msa_feats):
        """
        Args:
            esm_feats:   ESM-2特征 [L, D_esm]
            msa_feats:   MSA特征  [L, D_msa]
        Returns:
            fused_feats: 融合后的特征 [L, D_embed]
        """
        # 投影到统一维度
        esm_proj = self.esm_proj(esm_feats)  # [L, 256]
        msa_proj = self.msa_proj(msa_feats)  # [L, 256]

        # 调整维度为 [L, 1, 256] 以适配MultiheadAttention
        esm_proj = esm_proj.unsqueeze(1)  # [L, 1, 256]
        msa_proj = msa_proj.unsqueeze(1)  # [L, 1, 256]

        # 跨模态注意力
        attn_output_esm, _ = self.cross_attn(
            query=esm_proj,  # [L, 1, 256]
            key=msa_proj,
            value=msa_proj
        )  # [L, 1, 256]

        attn_output_msa, _ = self.cross_attn(
            query=msa_proj,
            key=esm_proj,
            value=esm_proj
        )  # [L, 1, 256]

        # 拼接注意力输出
        combined = torch.cat([
            attn_output_esm.squeeze(1),  # [L, 256]
            attn_output_msa.squeeze(1)  # [L, 256]
        ], dim=-1)  # [L, 512]

        # 动态门控融合
        gate = self.gate(combined)  # [L, 256]
        fused_feats = gate * esm_proj.squeeze(1) + (1 - gate) * attn_output_msa.squeeze(1)

        return fused_feats