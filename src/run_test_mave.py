import sys
import subprocess
import glob
import re
import torch
import models.gvp.data, models.gvp.models
import json
import os
import torch_geometric
import torch.nn as nn
import esm
import pandas as pd
import random
import torch.multiprocessing
import pickle
from feature_processor import FeatureProcessor
from collections import OrderedDict
torch.multiprocessing.set_sharing_strategy("file_system")
from models.msa_transformer.model import MSATransformer
from models.gvp.models import SSEmbGNN
from statistics import mean
from helpers import (
    read_msa,
    mave_val_pdb_to_prot,
    loop_pred,
    save_df_to_prism,
    get_prism_corr,
    get_prism_corr_all,
)
import pdb_parser_scripts.parse_pdbs as parse_pdbs
import torch.utils.data
from collections import OrderedDict
from ast import literal_eval
import subprocess
import shutil

def test(run_name, epoch, msa_row_attn_mask=True, get_only_ssemb_metrics=True, device=None):


    # Load data and dict of variant positions
    with open(f"../data/test/mave_val/data_with_msas.pkl", "rb") as fp:
        data = pickle.load(fp)

    with open(f"../data/test/mave_val/variant_pos_dict.pkl", "rb") as fp:
        variant_pos_dict = pickle.load(fp)

    # Convert to graph data sets
    testset = models.gvp.data.ProteinGraphData(data)
    letter_to_num = testset.letter_to_num

    # Load MSA Transformer
    _, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    model_msa = MSATransformer(msa_alphabet)
    model_path = f"../output/train/models/msa_transformer/{run_name}_msa_transformer_{epoch}.pt"
    print(f"Loading MSA model from: {model_path}")

    # 检查文件是否存在及大小
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(f"Model file is empty: {model_path} ({file_size} bytes)")

    # 安全加载
    try:
        state_dict = torch.load(model_path, map_location=f"cuda:{device}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    if state_dict is None:
        raise RuntimeError(f"Loaded state_dict is None for {model_path}")

    # 创建新状态字典并去除分布式训练前缀
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("module.", "")  # 去除分布式前缀
        new_state_dict[key] = value

    model_msa.load_state_dict(new_state_dict)
    model_msa = model_msa.to(device)
    model_msa.eval()

    feat_processor = FeatureProcessor(
        esm_dim=1280,
        msa_dim=768,
        embed_dim=256,
        num_heads=8
    ).to(device)
    # Load GVP
    node_dim = (256, 64)
    edge_dim = (32, 1)
    model_gvp = SSEmbGNN((6, 3), node_dim, (32, 1), edge_dim, feature_processor=feat_processor)
    model_gvp = model_gvp.to(device)

    # 加载 GVP 模型权重
    gvp_path = f"../output/train/models/gvp/{run_name}_gvp_{epoch}.pt"

    # 检查文件是否存在
    if not os.path.exists(gvp_path):
        raise FileNotFoundError(f"GVP model file not found: {gvp_path}")

    state_dict_gvp = torch.load(gvp_path, map_location=f"cuda:{device}")

    # 创建新状态字典并统一去除分布式训练前缀
    new_state_dict_gvp = OrderedDict()
    for k, v in state_dict_gvp.items():
        # 统一处理所有前缀
        name = k
        if name.startswith("module."):
            name = name[7:]
        # 专门处理特征处理器前缀
        name = name.replace("feature_processor.module.", "feature_processor.")
        new_state_dict_gvp[name] = v

    # 加载状态字典前先移动模型到设备
    model_gvp = model_gvp.to(device)

    # 加载状态字典
    try:
        model_gvp.load_state_dict(new_state_dict_gvp, strict=False)
    except RuntimeError as e:
        print(f"加载GVP模型时遇到部分问题: {e}")
        print("尝试仅加载匹配的参数...")
        # 仅加载匹配的参数
        model_gvp.load_state_dict(new_state_dict_gvp, strict=False)

    # 初始化数据加载器
    test_loader = torch_geometric.loader.DataLoader(
        testset, batch_size=1, shuffle=False
    )

    # Call test
    model_msa.eval()
    model_gvp.eval()

    with torch.no_grad():
        pred_list, acc_mean = loop_pred(
            model_msa,
            model_gvp,
            msa_batch_converter,
            test_loader,
            variant_pos_dict,
            data,
            letter_to_num,
            feat_processor,
            msa_row_attn_mask=msa_row_attn_mask,
            device=device,
        )

    # Transform results into df
    df_ml = pd.DataFrame(pred_list, columns=["dms_id", "variant_pos", "score_ml_pos"])

    # Save
    df_ml.to_csv(f"../output/mave_val/df_ml_{run_name}.csv", index=False)

    # Load
    df_ml = pd.read_csv(
        f"../output/mave_val/df_ml_{run_name}.csv",
        converters=dict(score_ml_pos=literal_eval),
    )

    # Compute score_ml from nlls
    pred_list_scores = []
    mt_list = [x for x in sorted(letter_to_num, key=letter_to_num.get)][:-1]

    for entry in data:
        dms_id = entry["name"]
        df_dms_id = df_ml[df_ml["dms_id"] == dms_id]

        wt = [[wt] * 20 for wt in entry["seq"]]
        pos = [[pos] * 20 for pos in list(df_dms_id["variant_pos"])]
        pos = [item for sublist in pos for item in sublist]
        mt = mt_list * len(wt)
        wt = [item for sublist in wt for item in sublist]
        score_ml = [
            item for sublist in list(df_dms_id["score_ml_pos"]) for item in sublist
        ]

        rows = [
            [dms_id, wt[i] + str(pos[i]) + mt[i], score_ml[i]] for i in range(len(mt))
        ]
        pred_list_scores += rows

    # Transform results into df
    df_ml_scores = pd.DataFrame(
        pred_list_scores, columns=["dms_id", "variant", "score_ml"]
    )

    # Save
    df_ml_scores.to_csv(f"../output/mave_val/df_ml_scores_{run_name}.csv", index=False)

    # Load
    df_ml_scores = pd.read_csv(f"../output/mave_val/df_ml_scores_{run_name}.csv")

    # Save results to PRISM format
    for dms_id in df_ml_scores["dms_id"].unique():
        df_dms = df_ml_scores[df_ml_scores["dms_id"] == dms_id]
        save_df_to_prism(df_dms, run_name, dms_id)

    # Compute metrics
    if get_only_ssemb_metrics == True:
        corrs = []
        for dms_id in df_ml_scores["dms_id"].unique():
            corr = get_prism_corr(dms_id, run_name)
            corrs.append(corr)
        return mean(corrs), acc_mean
    else:
        corrs_ssemb = []
        corrs_gemme = []
        corrs_rosetta = []
        for dms_id in df_ml_scores["dms_id"].unique():
            corrs = get_prism_corr_all(dms_id, run_name)
            corrs_ssemb.append(corrs[0])
            corrs_gemme.append(corrs[1])
            corrs_rosetta.append(corrs[2])
        print(f"SSEmb: Mean MAVE spearman correlation: {mean(corrs_ssemb):.3f}")
        print(f"GEMME: Mean MAVE spearman correlation: {mean(corrs_gemme):.3f}")
        print(f"Rosetta: Mean MAVE spearman correlation: {mean(corrs_rosetta):.3f}")