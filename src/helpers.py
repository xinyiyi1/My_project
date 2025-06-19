import torch
import torch.nn as nn
import torch.nn.functional as F
import models.gvp.data, models.gvp.models
from datetime import datetime

import tqdm.autonotebook as tqdm
import copy
from functools import partial
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import random
import esm
import glob
import pickle
import pandas as pd
import subprocess
import numpy as np
import json
from collections import OrderedDict
from feature_processor import FeatureProcessor
from Bio.Align import substitution_matrices

print = partial(print, flush=True)
import torch.multiprocessing
import torch.distributed as dist

torch.multiprocessing.set_sharing_strategy("file_system")
import pdb_parser_scripts.parse_pdbs as parse_pdbs
from PrismData import PrismParser, VariantData
from scipy import stats
from sklearn.metrics import precision_recall_curve
import pytz
from visualization import plot_scatter, plot_hist

mave_val_pdb_to_prot = {
    "5BON": "NUD15",
    "4QO1": "P53",
    "1CVJ": "PABP",
    "1WYW": "SUMO1",
    "3OLM": "RL401",
    "6NYO": "RL401",
    "3SO6": "LDLRAP1",
    "4QTA": "MAPK",
    "1D5R": "PTEN",
    "2H11": "TPMT",
    "1R9O": "CP2C9",
}


def remove_insertions(sequence: str):
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]

import re
import os
import csv

def prepare_mave_val(num_ensemble=5, device='cpu'):
    # Load data
    dms_filenames = sorted(glob.glob(f"/home/zhaom/my_proj/data/test/mave_val/raw/*.txt"))
    print("找到的DMS文件：", dms_filenames)

    # 使用列表收集数据，最后统一处理
    data_records = []

    for dms_filename in dms_filenames:
        print(f"\n处理文件：{dms_filename}")
        dms_id = dms_filename.split("/")[-1].split("_")[1]

        try:
            with open(dms_filename, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    try:
                        variant = str(parts[0])
                        score = float(parts[1])
                        data_records.append((dms_id, variant, score))
                    except (ValueError, IndexError) as e:
                        print(f"跳过无法解析的行: {line} (错误: {e})")
                        continue

        except UnicodeDecodeError:
            with open(dms_filename, 'r', encoding='latin-1') as f:
                next(f)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    try:
                        variant = str(parts[0])
                        score = float(parts[1])
                        data_records.append((dms_id, variant, score))
                    except (ValueError, IndexError) as e:
                        print(f"跳过无法解析的行: {line} (错误: {e})")
                        continue

    # 检查是否有有效数据
    if not data_records:
        raise ValueError("没有有效数据可以创建DataFrame")

    # 使用csv模块直接写入文件，完全绕过Pandas
    os.makedirs("../data/test/mave_val/exp", exist_ok=True)
    output_file = "../data/test/mave_val/exp/dms.csv"

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dms_id', 'variant', 'score_dms'])
        writer.writerows(data_records)

    print(f"\n数据已成功保存到 {output_file}")
    print("前5条记录示例:")
    for i, record in enumerate(data_records[:5], 1):
        print(f"{i}. {record}")

    ## Pre-process PDBs
    pdb_dir = "../data/test/mave_val/structure"
    subprocess.run(
        [
            "pdb_parser_scripts/clean_pdbs.sh",
            str(pdb_dir),
        ]
    )
    parse_pdbs.parse(pdb_dir)

    # Load structure data
    print("Loading models and data...")
    with open(f"{pdb_dir}/coords.json") as json_file:
        data = json.load(json_file)
    json_file.close()

    ## Compute MSAs
    #sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    #subprocess.run(
    #    [
    #        "colabfold_search",
    #        f"{pdb_dir}/seqs.fasta",
    #        "/projects/prism/people/skr526/databases",
    #        "../data/test/mave_val/msa/",
    #    ]
    #)
    #subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/mave_val/msa"])


    print("Loading ESM2 model for embeddings...")
    model_esm2, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm2 = model_esm2.to(device)
    model_esm2.eval()
    esm_batch_converter = esm_alphabet.get_batch_converter()

    # 遍历每个蛋白质数据
    for entry in tqdm.tqdm(data, desc="Computing ESM2 embeddings"):
        seq = entry["seq"]

        # 清洗序列
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        seq_clean = "".join([aa if aa in valid_aa else "X" for aa in seq])

        # 截断序列长度，确保不超过模型支持的最大长度
        max_length = 1024
        seq_clean = seq_clean[:max_length]

        # 打印序列清洗后的长度和内容
        print(f"Protein {entry['name']} - Cleaned sequence: {seq_clean}")
        print(f"Protein {entry['name']} - Cleaned sequence length: {len(seq_clean)}")

        # 生成 ESM2 嵌入
        with torch.no_grad():
            # 转换序列为模型输入
            batch_labels, batch_strs, batch_tokens = esm_batch_converter(
                [(entry['name'], seq_clean)]  # 将清洗后的序列传入
            )

            # 打印 batch_tokens 的形状
            print(f"Batch tokens shape: {batch_tokens.shape}")

            batch_tokens = batch_tokens.to(device)
            # 获取 ESM2 嵌入
            results = model_esm2(batch_tokens, repr_layers=[33])  # 获取 ESM2 的嵌入
            print(f"Generated embedding shape before slicing: {results['representations'][-1].shape}")
            embeddings = results["representations"][33].cpu()

        # 打印 ESM2 嵌入的形状
        print(f"Generated embedding shape: {embeddings.shape}")
        print(f"ESM2 embedding shape for {entry['name']}: {embeddings.shape}")

        # 保存特征（去除 CLS 和 SEP 标记）
        entry["esm2_embeddings"] = embeddings[:, 1:-1, :].squeeze(0).numpy()

    # Load MSA data
    msa_filenames = sorted(glob.glob(f"../data/test/mave_val/msa/*.a3m"))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        name = f.split("/")[-1].split(".")[0]
        mave_msa_sub[name] = []
        for j in range(num_ensemble):
            msa = read_msa(f)
            msa_sub = [msa[0]]
            k = min(len(msa) - 1, 16 - 1)
            msa_sub += [msa[i] for i in sorted(random.sample(range(1, len(msa)), k))]
            mave_msa_sub[name].append(msa_sub)

    # Add MSAs to data
    for entry in data:
        entry["msa"] = mave_msa_sub[entry["name"]]

    # Change data names
    for entry in data:
        entry["name"] = mave_val_pdb_to_prot[entry["name"]]

    # Make variant pos dict
    variant_pos_dict = {}
    for entry in data:
        seq = entry["seq"]
        pos = [str(x + 1) for x in range(len(seq))]
        variant_wtpos_list = [[seq[i] + pos[i]] for i in range(len(seq))]
        variant_wtpos_list = [x for sublist in variant_wtpos_list for x in sublist]
        variant_pos_dict[entry["name"]] = variant_wtpos_list

    # Save data and dict of variant positions
    with open(f"../data/test/mave_val/data_with_msas.pkl","wb") as fp:
        pickle.dump(data, fp)

    with open(f"../data/test/mave_val/variant_pos_dict.pkl","wb") as fp:
        pickle.dump(variant_pos_dict, fp)

def prepare_proteingym(num_ensemble=5):
    # 定义并保存验证集列表
    val_list = [
        "NUD15_HUMAN_Suiter_2020",
        "TPMT_HUMAN_Matreyek_2018",
        "CP2C9_HUMAN_Amorosi_abundance_2021",
        "P53_HUMAN_Kotler_2018",
        "PABP_YEAST_Melamed_2013",
        "SUMO1_HUMAN_Weile_2017",
        "RL40A_YEAST_Roscoe_2014",
        "PTEN_HUMAN_Mighell_2018",
        "MK01_HUMAN_Brenan_2016",
    ]

    with open(f"../data/test/proteingym/val_list.pkl", "wb") as fp:
        pickle.dump(val_list, fp)

    # 获取所有 CSV 文件路径
    dms_dir = "../data/test/proteingym/raw/ProteinGym_substitutions/"
    dms_filenames = sorted(glob.glob(os.path.join(dms_dir, "*.csv")))
    if not dms_filenames:
        raise FileNotFoundError(f"未找到 CSV 文件于 {dms_dir}")

    # 初始化数据列表
    df_dms_list = []

    # 处理每个文件
    for dms_filename in dms_filenames:
        dms_id = os.path.basename(dms_filename).split(".")[0]
        if dms_id in val_list:
            continue

        try:
            with open(dms_filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # 强制转换为Python原生类型
                        dms_id_py = str(dms_id)
                        mutant_py = str(row["mutant"]).strip()
                        dms_score_py = float(row["DMS_score"].strip())
                        df_dms_list.append((dms_id_py, mutant_py, dms_score_py))
                    except (ValueError, TypeError) as e:
                        print(f"跳过无效行: 文件 {dms_filename}, 错误: {e}")
                        continue
        except Exception as e:
            print(f"处理文件 {dms_filename} 时出错: {repr(e)}")
            continue

    # 检查是否有数据
    if not df_dms_list:
        raise ValueError("未处理任何数据，请检查文件路径和内容。")

    # 保存数据到CSV文件
    os.makedirs("../data/test/proteingym/exp", exist_ok=True)
    output_csv_path = "../data/test/proteingym/exp/dms.csv"

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(["dms_id", "variant", "score_dms"])
        # 逐行写入数据
        for row in df_dms_list:
            writer.writerow(row)

    print(f"数据已直接写入CSV文件: {output_csv_path}")

    # Pre-process PDBs
    pdb_dir = "../data/test/proteingym/structure"
    subprocess.run(
        [
            "pdb_parser_scripts/clean_pdbs.sh",
            str(pdb_dir),
        ]
    )
    # 假设 parse_pdbs 是一个函数，用于解析PDB结构
    # parse_pdbs.parse(pdb_dir)

    # Load structure data
    print("Loading models and data...")
    try:
        with open(f"{pdb_dir}/coords.json") as json_file:
            data_all = json.load(json_file)
    except FileNotFoundError:
        print(f"未找到 coords.json 文件，请检查路径 {pdb_dir}")
        data_all = []

    # Remove assays from validation set
    data = [x for x in data_all if x.get("name") not in val_list]

    # Load MSA data
    msa_dir = "../data/test/proteingym/msa/"
    msa_filenames = sorted(glob.glob(os.path.join(msa_dir, "*.a3m")))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        name = os.path.basename(f).split(".")[0]
        mave_msa_sub[name] = []
        for j in range(num_ensemble):
            try:
                with open(f, "r") as msa_file:
                    msa = msa_file.readlines()
                msa_sub = [msa[0]]
                k = min(len(msa) - 1, 16 - 1)
                if len(msa) > 1:
                    sampled_indices = sorted(random.sample(range(1, len(msa)), k))
                    msa_sub += [msa[i] for i in sampled_indices]
                mave_msa_sub[name].append(msa_sub)
            except Exception as e:
                print(f"处理 MSA 文件 {f} 时出错: {repr(e)}")
                continue

    # Add MSAs to data
    for entry in data:
        entry_name = entry.get("name")
        if entry_name in mave_msa_sub:
            entry["msa"] = mave_msa_sub[entry_name]

    # Make variant pos dict
    variant_pos_dict = {}
    # 从 df_dms_list 中提取数据
    dms_ids = list({row[0] for row in df_dms_list})
    for dms_id in dms_ids:
        variants = [row[1] for row in df_dms_list if row[0] == dms_id]
        variant_wtpos_list = []
        for variant in variants:
            parts = variant.split(":")
            for part in parts:
                if part:
                    variant_wtpos_list.append(part[:-1])
        variant_wtpos_list = list(OrderedDict.fromkeys(variant_wtpos_list))
        variant_pos_dict[dms_id] = variant_wtpos_list

    # Save data and dict of variant positions
    with open(f"../data/test/proteingym/data_with_msas.pkl", "wb") as fp:
        pickle.dump(data, fp)

    with open(f"../data/test/proteingym/variant_pos_dict.pkl", "wb") as fp:
        pickle.dump(variant_pos_dict, fp)

    print("数据处理完成！")

def save_df_to_prism(df, run_name, dms_id):
    # Initialize
    parser = PrismParser()

    # Print to PRISM format
    df_prism = df[["variant", "score_ml"]].reset_index(drop=True)

    # Get current CPH time
    timestamp = datetime.now(pytz.timezone("Europe/Copenhagen")).strftime(
        "%Y-%m-%d %H:%M"
    )

    metadata = {
        "version": 1,
        "protein": {
            "name": "Unknown",
            "organism": "Unknown",
            "uniprot": "Unknown",
            "sequence": "".join([x for x in list(df_prism["variant"].str[0])[::20]]),
            "first_residue_number": 1,
            "pdb": "Unknown",
            "chain": "Unknown",
        },
        "columns": {"score_ml": "SSEmb prediction"},
        "created": {f"{timestamp} (CPH time) - lasse.blaabjerg@bio.ku.dk"},
    }

    # Write data
    dataset = VariantData(metadata, df_prism)
    dataset.add_index_columns()
    prism_filename = f"../output/mave_val/prism_{dms_id}_ssemb_{run_name}.txt"
    parser.write(prism_filename, dataset, comment_lines="")


def get_prism_corr(dms_id, run_name):
    # Initialize
    parser = PrismParser()

    # Load PRISM file - Experimental
    prism_pred = f"../data/test/mave_val/exp/prism_{dms_id}_dms.txt"
    df_prism_dms = parser.read(prism_pred)
    len_dms = len(df_prism_dms.dataframe)

    # Load PRISM file - SSEmb prediction
    prism_pred = f"../output/mave_val/prism_{dms_id}_ssemb_{run_name}.txt"
    df_prism_ssemb = parser.read(prism_pred)

    # Merge PRISM files
    df_total = df_prism_dms.merge([df_prism_ssemb], merge="inner").dataframe
    len_total = len(df_total)
    print(
        f"{dms_id} number of MAVE data points lost during merging: {len_dms-len_total}"
    )

    # Get correlation
    x = df_total["score_dms_00"].values
    y = df_total["score_ml_01"].values
    spearman_r_ssemb = stats.spearmanr(x, y)[0]
    print(f"{dms_id}: SSEmb spearman correlation vs. MAVE is: {spearman_r_ssemb:.3f}")
    return spearman_r_ssemb


def get_prism_corr_all(dms_id, run_name):
    # Initialize
    parser = PrismParser()

    # Load PRISM file - Experimental
    prism_pred = f"../data/test/mave_val/exp/prism_{dms_id}_dms.txt"
    df_prism_dms = parser.read(prism_pred)
    len_dms = len(df_prism_dms.dataframe)

    # Load PRISM file - SSEmb prediction
    prism_pred = f"../output/mave_val/prism_{dms_id}_ssemb_{run_name}.txt"
    df_prism_ssemb = parser.read(prism_pred)

    # Load PRISM file - GEMME
    prism_pred = f"../data/test/mave_val/gemme/prism_gemme_{dms_id}.txt"
    df_prism_gemme = parser.read(prism_pred)

    # Load PRISM file - Rosetta
    prism_pred = f"../data/test/mave_val/rosetta/prism_rosetta_{dms_id}.txt"
    df_prism_rosetta = parser.read(prism_pred)

    # Merge PRISM files
    df_total = df_prism_dms.merge(
        [df_prism_ssemb, df_prism_gemme, df_prism_rosetta], merge="inner"
    ).dataframe
    len_total = len(df_total)
    print(
        f"{dms_id} number of MAVE data points lost during merging: {len_dms-len_total}"
    )

    # Get correlation
    x = df_total["score_dms_00"].values
    y = df_total["score_ml_01"].values
    spearman_r_ssemb = stats.spearmanr(x, y)[0]
    print(f"{dms_id}: SSEmb spearman correlation vs. MAVE is: {spearman_r_ssemb:.3f}")

    x = df_total["score_dms_00"].values
    y = df_total["gemme_score_02"].values
    spearman_r_gemme = stats.spearmanr(x, y)[0]
    print(f"{dms_id}: GEMME spearman correlation vs. MAVE is: {spearman_r_gemme:.3f}")

    x = df_total["score_dms_00"].values
    y = df_total["mean_ddG_03"].values
    spearman_r_rosetta = stats.spearmanr(x, y)[0]
    print(
        f"{dms_id}: Rosetta spearman correlation vs. MAVE is: {spearman_r_rosetta:.3f}"
    )

    # Make scatter plot
    plot_scatter(df_total, dms_id, run_name, "mave_val")

    # Make histogram plot
    plot_hist(df_total, dms_id, run_name, "mave_val")
    return spearman_r_ssemb, spearman_r_gemme, spearman_r_rosetta


def scannet_collate_fn(batch):
    emb = batch[0]["emb"].unsqueeze(0)
    label = batch[0]["label"].unsqueeze(0)
    prot_weight = batch[0]["prot_weight"]
    return emb, label, prot_weight


def mask_seq_and_msa(
    seq,
    msa_batch_tokens,
    coord_mask,
    device,
    mask_size=0.15,
    mask_tok=0.60,
    mask_col=0.20,
    mask_rand=0.10,
    mask_same=0.10,
):
    # Get masked positions
    assert len(seq) == len(coord_mask), f"序列长度 {len(seq)} 与坐标掩码长度 {len(coord_mask)} 不匹配"
    assert mask_tok + mask_col + mask_rand + mask_same == 1.00
    indices = torch.arange(len(seq), device=device)
    indices_mask = indices[coord_mask]  # Only consider indices within coord mask
    if len(indices_mask) == 0:
        return seq, msa_batch_tokens, torch.tensor([], device=device)  # 无有效掩码位置时返回空
    indices_mask = indices_mask[torch.randperm(indices_mask.size(0))]
    mask_pos_all = indices_mask[: int(len(indices_mask) * mask_size)]



    mask_pos_all = mask_pos_all[mask_pos_all < len(seq)]




    mask_pos_tok = mask_pos_all[: int(len(mask_pos_all) * mask_tok)]
    mask_pos_col = mask_pos_all[
        int(len(mask_pos_all) * (mask_tok)) : int(
            len(mask_pos_all) * (mask_tok + mask_col)
        )
    ]
    mask_pos_rand = mask_pos_all[
        int(len(mask_pos_all) * (mask_tok + mask_col)) : int(
            len(mask_pos_all) * (mask_tok + mask_col + mask_rand)
        )
    ]

    # ESM2单序列独立掩码（
    seq_mask_prob = 0.15
    seq_mask_pos = torch.rand(len(seq), device=device) < seq_mask_prob
    seq_masked = seq.clone()
    seq_masked[seq_mask_pos] = 20  # 假设mask_id=20

    # Do masking - MSA level
    msa_batch_tokens_masked = msa_batch_tokens.clone()
    msa_batch_tokens_masked[:, 0, mask_pos_tok + 1] = 32  # Correct for <cls> token
    msa_batch_tokens_masked[:, :, mask_pos_col + 1] = 32  # Correct for <cls> token
    msa_batch_tokens_masked[:, 0, mask_pos_rand + 1] = torch.randint(
        low=4, high=24, size=(len(mask_pos_rand),), device=device
    )  # Correct for <cls> token, draw random standard amino acids

    # Do masking - seq level
    seq_masked = seq.clone()
    mask_pos_tok_all = torch.cat((mask_pos_tok, mask_pos_col))
    seq_masked[mask_pos_tok_all] = 20
    seq_masked[mask_pos_rand] = torch.randint(
        low=0, high=20, size=(len(mask_pos_rand),), device=device
    )

    return seq_masked, msa_batch_tokens_masked, mask_pos_all


def forward(
        model_msa,
        model_gvp,
        msa_batch_tokens_masked,
        seq_masked,
        batch,
        feat_processor,
        msa_row_attn_mask=True,
        mask_pos=None,
        loss_fn=None,
        batch_prots=None,
        get_logits_only=False,

):
    # MSA特征提取增强
    with torch.no_grad():
        msa_transformer_pred = model_msa(
            msa_batch_tokens_masked,
            repr_layers=[12]
        )
    msa_rep = msa_transformer_pred["representations"][12]  # [1, num_msa, seq_len+1, dim]
    attn_scores = msa_rep[0, :, 0, 0]  # 取CLS令牌的首元素 [num_msa]
    attn_weights = torch.softmax(attn_scores, dim=0)  # [num_msa]
    msa_emb = (msa_rep[0, :, 1:, :] * attn_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
    esm2_feats = batch.esm_features  # [batch_size, seq_len, 1280]
    esm2_feats = esm2_feats.squeeze(0)  # 如果是单样本，去除 batch 维度
    print("esm2_feats shape:", esm2_feats.shape)


    # 通过 FeatureProcessor 融合特征
    fused_feats = feat_processor(esm2_feats, msa_emb)



    # 维度对齐校验
    assert esm2_feats.size(0) == msa_emb.size(0), \
        f"ESM2 seq_len {esm2_feats.shape[0]} != MSA seq_len {msa_emb.shape[0]}"
    assert esm2_feats.size(1) == 1280, "ESM2 feature dim should be 1280"
    assert msa_emb.size(1) == 768, "MSA feature dim should be 768"

    # GVP前向传播
    h_V = (batch.node_s, batch.node_v)
    h_E = (batch.edge_s, batch.edge_v)
    logits = model_gvp(
        h_V=h_V,
        edge_index=batch.edge_index,
        h_E=h_E,
        esm2_feats=esm2_feats,
        msa_feats=msa_emb,
        seq=seq_masked,
        fused_feats=fused_feats
    )

    if get_logits_only:
        return logits
    else:
        # 掩码有效性检查
        if mask_pos is None:
            mask_pos = torch.where(seq_masked == 20)[0]  # 假设mask_id=20
        targets = batch.seq[mask_pos]

        # 损失计算
        loss = loss_fn(logits[mask_pos], targets)
        return loss / batch_prots, logits, targets


def loop_trainval(
    model_msa,
    model_gvp,
    msa_batch_converter,
    dataloader,
    batch_prots,
    epoch,
    rank,
    epoch_finetune_msa,
    feat_processor,
    msa_row_attn_mask=True,
    optimizer=None,
    scaler=None,
):


    # Initialize
    t = tqdm.tqdm(dataloader)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_count = 0, 0, 0

    # Initialize models and optimizer
    if optimizer == None:
        model_gvp = model_gvp.module
        if epoch >= epoch_finetune_msa:
            model_msa = model_msa.module
    else:
        optimizer.zero_grad(set_to_none=True)

    # Loop over proteins
    for i, batch in enumerate(t):
        print(f"Rank {rank} - Computing predictions for protein: {batch.name[0]}")


        with torch.cuda.amp.autocast(
            enabled=True
        ):  # OBS: This seems to be necessary for GVPLarge model close to convergence?
            # Move data to device
            batch = batch.to(rank)

            # Subsample MSA
            msa_sub = [batch.msa[0][0]]  # Always get query
            k = min(len(batch.msa[0]) - 1, 16 - 1)
            msa_sub += [
                batch.msa[0][j]
                for j in sorted(random.sample(range(1, len(batch.msa[0])), k))
            ]
            feat_processor = FeatureProcessor().to(rank)

            # Tokenize MSA
            msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(
                msa_sub
            )
            msa_batch_tokens = msa_batch_tokens.to(rank)

            # Mask sequence
            seq_masked, msa_batch_tokens_masked, mask_pos = mask_seq_and_msa(
                batch.seq, msa_batch_tokens, batch.mask, rank
            )

            if optimizer:
                if (i + 1) % batch_prots == 0 or (i + 1) == len(
                    t
                ):  # Accumulate gradients and update every n'th protein or last iteration
                    # Forward pass
                    loss_value, logits, seq = forward(
                        model_msa,
                        model_gvp,
                        msa_batch_tokens_masked,
                        seq_masked,
                        batch,
                        feat_processor,
                        msa_row_attn_mask=msa_row_attn_mask,
                        mask_pos=mask_pos,
                        loss_fn=loss_fn,
                        batch_prots=batch_prots,
                    )

                    # Backprop
                    scaler.scale(loss_value).backward()

                    # Optimizer step
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            model_msa.parameters(),
                            model_gvp.parameters(),
                        ),
                        1.0,
                    )  # Clip gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Create barrier
                    dist.barrier()
                else:
                    if epoch < epoch_finetune_msa:
                        with model_gvp.no_sync():
                            # Forward pass
                            loss_value, logits, seq = forward(
                                model_msa,
                                model_gvp,
                                msa_batch_tokens_masked,
                                seq_masked,
                                batch,
                                feat_processor,
                                msa_row_attn_mask=msa_row_attn_mask,
                                mask_pos=mask_pos,
                                loss_fn=loss_fn,
                                batch_prots=batch_prots,
                            )

                            # Backprop
                            scaler.scale(loss_value).backward()
                    else:
                        with model_gvp.no_sync():
                            with model_msa.no_sync():
                                # Forward pass
                                loss_value, logits, seq = forward(
                                    model_msa,
                                    model_gvp,
                                    msa_batch_tokens_masked,
                                    seq_masked,
                                    batch,
                                    feat_processor,
                                    msa_row_attn_mask=msa_row_attn_mask,
                                    mask_pos=mask_pos,
                                    loss_fn=loss_fn,
                                    batch_prots=batch_prots,
                                )

                                # Backprop
                                scaler.scale(loss_value).backward()

            else:
                # Forward pass
                loss_value, logits, seq = forward(
                    model_msa,
                    model_gvp,
                    msa_batch_tokens_masked,
                    seq_masked,
                    batch,
                    feat_processor,
                    msa_row_attn_mask=msa_row_attn_mask,
                    mask_pos=mask_pos,
                    loss_fn=loss_fn,
                    batch_prots=batch_prots,
                )

        # Update loss etc.

        if len(mask_pos) == 0:
            # 如果没有掩码位置，直接跳过当前批次
            continue

        num_nodes = len(mask_pos)
        total_loss += loss_value.detach()
        total_count += num_nodes
        pred = torch.argmax(logits[mask_pos], dim=-1).detach().cpu().numpy()  # [num_masked]
        true = batch.seq[mask_pos].detach().cpu().numpy()
        # 新增维度校验
        if pred.shape != true.shape:
            raise ValueError(f"预测维度 ({pred.shape}) 与标签维度 ({true.shape}) 不匹配")

        total_correct += (pred == true).sum()

    return total_loss / total_count, total_correct / total_count


def loop_pred(
        model_msa,
        model_gvp,
        msa_batch_converter,
        dataloader,
        variant_pos_dict,
        data,
        letter_to_num,
        feat_processor,
        msa_row_attn_mask=True,
        device=None,
):
    # Initialize
    t = tqdm.tqdm(dataloader)
    pred_list = []
    total_correct, total_count = 0, 0
    current_header = None

    # Loop over proteins
    for i, batch in enumerate(t):
        with torch.cuda.amp.autocast(enabled=True):
            # Move data to device
            batch = batch.to(device)

            # Initialize for each protein
            variant_wtpos_list = variant_pos_dict[batch.name[0]]
            seq_len = len(batch.seq)
            original_seq_len = seq_len  # 保存原始序列长度

            # Make masked marginal predictions
            for k, variant_wtpos in enumerate(variant_wtpos_list):
                print(
                    f"Computing logits for protein {batch.name[0]} ({i + 1}/{len(dataloader)}) at position: {k + 1}/{len(variant_wtpos_list)}"
                )

                # Extract variant info and initialize
                wt = letter_to_num[variant_wtpos[0]]
                pos = int(variant_wtpos[1:]) - 1  # Shift from DMS pos to seq idx
                score_ml_pos_ensemble = torch.zeros((len(batch.msa[0]), 20))

                # If protein too long; redo data loading with fragment
                if seq_len > 1024 - 1:
                    # Get sliding window
                    window_size = 1024 - 1
                    lower_side = max(pos - window_size // 2, 0)
                    upper_side = min(pos + window_size // 2 + 1, seq_len)
                    lower_bound = max(0, lower_side - (pos + window_size // 2 + 1 - upper_side))
                    upper_bound = min(seq_len, upper_side + (lower_side - (pos - window_size // 2)))

                    # 确保切片有效
                    if lower_bound >= upper_bound:
                        lower_bound = max(0, pos - window_size // 2)
                        upper_bound = min(seq_len, pos + window_size // 2 + 1)

                    # Get fragment
                    data_frag = copy.deepcopy(data[i])
                    data_frag["seq"] = data[i]["seq"][lower_bound:upper_bound]
                    data_frag["coords"] = data[i]["coords"][lower_bound:upper_bound]

                    # 更新MSA数据
                    sliced_msa = []
                    for header, full_seq in data[i]["msa"]:
                        # 确保序列长度匹配
                        if len(full_seq) != original_seq_len:
                            # 截断或填充以匹配
                            if len(full_seq) > original_seq_len:
                                full_seq = full_seq[:original_seq_len]
                            else:
                                full_seq = full_seq + 'X' * (original_seq_len - len(full_seq))

                        # 切片序列
                        sliced_seq = full_seq[lower_bound:upper_bound]
                        sliced_msa.append((header, sliced_seq))

                    data_frag["msa"] = sliced_msa

                    # 更新ESM嵌入
                    if "esm2_embeddings" in data[i]:
                        # 确保长度匹配
                        esm_embeddings = data[i]["esm2_embeddings"]
                        if len(esm_embeddings) != original_seq_len:
                            # 截断或填充
                            if len(esm_embeddings) > original_seq_len:
                                esm_embeddings = esm_embeddings[:original_seq_len]
                            else:
                                padding = np.zeros((original_seq_len - len(esm_embeddings), esm_embeddings.shape[1]))
                                esm_embeddings = np.vstack([esm_embeddings, padding])

                        data_frag["esm2_embeddings"] = esm_embeddings[lower_bound:upper_bound]

                    batch = models.gvp.data.ProteinGraphData([data_frag])[0]
                    batch = batch.to(device)
                    batch.msa = [batch.msa]
                    batch.name = [batch.name]

                    # Re-map position
                    pos = pos - lower_bound
                    seq_len = len(batch.seq)  # 更新序列长度

                # 获取原始MSA数据
                raw_msa = batch.msa[0] if hasattr(batch, 'msa') and batch.msa else []
                msa_sub = []
                current_header = None  # 重置当前header

                # 深度处理各种可能的格式
                for item in raw_msa:
                    if isinstance(item, tuple) and len(item) == 2:
                        msa_sub.append(item)

                    elif isinstance(item, list) and len(item) >= 2:
                        header = item[0] if isinstance(item[0], str) else str(item[0])
                        sequence = item[1] if isinstance(item[1], str) else ""
                        msa_sub.append((header, sequence))

                    elif isinstance(item, str):
                        if item.startswith('>'):
                            current_header = item[1:].strip()
                        else:
                            if current_header:
                                msa_sub.append((current_header, item.strip()))
                                current_header = None
                            else:
                                # 没有header的sequence
                                msa_sub.append(("unknown", item.strip()))
                    else:
                        print(f"跳过无法处理的MSA条目: {type(item)} - {item}")

                if not msa_sub:
                    print(f"警告: 蛋白质 {batch.name[0]} 无有效MSA，使用虚拟数据")
                    # 从batch.seq创建序列字符串
                    seq_array = batch.seq.cpu().numpy()
                    if hasattr(batch, 'dataset') and hasattr(batch.dataset, 'num_to_letter'):
                        seq_str = ''.join([batch.dataset.num_to_letter.get(x, 'X') for x in seq_array])
                    else:
                        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                        seq_str = ''.join([amino_acids[x] if x < len(amino_acids) else 'X' for x in seq_array])
                    msa_sub = [(f"dummy_{batch.name[0]}", seq_str)]

                # 确保所有序列长度一致
                seq_lens = set(len(seq) for _, seq in msa_sub)
                if len(seq_lens) > 1:
                    print(f"警告: MSA序列长度不一致: {seq_lens}")
                    # 取最小长度
                    min_len = min(seq_lens)
                    print(f"统一截断为最小长度: {min_len}")
                    msa_sub = [(h, s[:min_len]) for h, s in msa_sub]

                # 打印MSA信息
                print(f"处理后MSA条目数: {len(msa_sub)}")
                for j in range(min(3, len(msa_sub))):
                    header, seq = msa_sub[j]
                    print(f"MSA记录 {j}: header='{header[:30]}...', seq_len={len(seq)}")

                try:
                    # 尝试转换MSA
                    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_sub)
                except Exception as e:
                    print(f"MSA转换失败: {str(e)}")

                    # 详细记录错误信息
                    print(f"MSA内容类型: {type(msa_sub)}")
                    print(f"条目数: {len(msa_sub)}")
                    if msa_sub:
                        print(f"第一条记录类型: {type(msa_sub[0])}")
                        if isinstance(msa_sub[0], tuple):
                            print(f"第一条记录内容: header={msa_sub[0][0]}, sequence={msa_sub[0][1][:50]}...")

                    # 尝试使用仅查询序列
                    if msa_sub:
                        print("尝试仅使用查询序列...")
                        query_sequence = msa_sub[0][1]  # 假设第一个是查询序列
                        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(
                            [("query", query_sequence)]
                        )
                    else:
                        print("创建虚拟MSA tokens...")
                        seq_len = len(batch.seq)
                        msa_batch_tokens = torch.ones((1, 1, seq_len + 2), dtype=torch.int64,
                                                      device=device) * 32  # <unk> token
                        msa_batch_tokens[:, :, 0] = 0  # <cls>
                        msa_batch_tokens[:, :, -1] = 2  # <eos>
                        msa_batch_labels = ["dummy"]
                        msa_batch_strs = ["X" * seq_len]

                msa_batch_tokens = msa_batch_tokens.to(device)

                # Mask position
                msa_batch_tokens_masked = msa_batch_tokens.detach().clone()

                # 确保位置在有效范围内
                if pos < msa_batch_tokens_masked.shape[2] - 1:
                    msa_batch_tokens_masked[:, 0, pos + 1] = 32  # Account for appended <cls> token
                else:
                    print(f"警告: 位置 {pos} 超出MSA tokens范围 {msa_batch_tokens_masked.shape}")
                    pos = min(pos, msa_batch_tokens_masked.shape[2] - 2)

                seq_masked = batch.seq.detach().clone()
                if pos < len(seq_masked):
                    seq_masked[pos] = 20
                else:
                    print(f"警告: 位置 {pos} 超出序列长度 {len(seq_masked)}")
                    continue

                # Forward pass
                logits = forward(
                    model_msa,
                    model_gvp,
                    msa_batch_tokens_masked,
                    seq_masked,
                    batch,
                    feat_processor,
                    msa_row_attn_mask=msa_row_attn_mask,
                    get_logits_only=True,
                )

                # 确保位置在logits范围内
                if pos < logits.shape[0]:
                    logits_pos = logits[pos, :]
                else:
                    print(f"错误: 位置 {pos} 超出logits范围 {logits.shape}")
                    continue

                # Compute accuracy
                pred = (
                    torch.argmax(logits_pos, dim=-1).detach().cpu().numpy().item()
                )
                true = batch.seq[pos].detach().cpu().numpy().item()
                if pred == true:
                    total_correct += 1 / len(batch.msa[0])

                # Compute all possible nlls at this position based on known wt
                nlls_pos = -torch.log(F.softmax(logits_pos, dim=-1))
                nlls_pos_repeat = nlls_pos.repeat(20, 1)
                score_ml_pos_ensemble[j, :] = torch.diagonal(
                    nlls_pos_repeat[:, wt] - nlls_pos_repeat[:, torch.arange(20)]
                )

            # Append to total
            score_ml_pos = torch.mean(score_ml_pos_ensemble[: j + 1, :], axis=0)
            pred_list.append(
                [
                    batch.name[0],
                    int(variant_wtpos[1:]),
                    score_ml_pos.detach().cpu().tolist(),
                ]
            )
            total_count += 1

    return pred_list, total_correct / total_count


def loop_getemb(model_msa, model_gvp, msa_batch_converter, dataloader, device=None):
    # Initialize
    t = tqdm.tqdm(dataloader)
    emb_dict = {}

    for i, batch in enumerate(t):
        batch = batch.to(device)

        with torch.cuda.amp.autocast(enabled=True):
            # Get frozen embeddings
            msa_sub = batch.msa[0]
            msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(
                msa_sub
            )
            msa_batch_tokens = msa_batch_tokens.to(device)

            # Make MSA Transformer predictions
            msa_batch_tokens_masked = msa_batch_tokens.detach().clone()
            seq = batch.seq.clone()
            msa_transformer_pred = model_msa(
                msa_batch_tokens_masked,
                repr_layers=[12],
                self_row_attn_mask=batch.dist_mask
            )
            msa_emb = msa_transformer_pred["representations"][12][0, 0, 1:, :]

            # Make GVP predictions
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            emb_allpos = model_gvp(
                h_V, batch.edge_index, h_E, msa_emb, seq, get_emb=True
            )

            # Update dict
            emb_dict[batch.name[0]] = emb_allpos

    return emb_dict


def loop_scannet_trainval(
    model_transformer, dataloader, device=None, optimizer=None, batch_prots=None
):
    # Initialize
    t = tqdm.tqdm(dataloader)
    total_loss, total_correct, total_count = 0, 0, 0
    acc_prots = []
    matthews_prots = []
    perplexity_prots = []
    pred_list = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for i, (emb, label, prot_weight) in enumerate(t):
        # print(f"{i+1}/{len(t)}")

        # Transfer to GPU
        emb, label = emb.to(device), label.to(device)

        # Make Transformer prediction
        emb = emb.transpose(1, 0)  # B x N x H --> N x B x H
        label_pred = model_transformer(emb)
        weights = torch.ones(label.reshape(-1).size(), device=label.device)
        weights[label.reshape(-1) == 1] = 4
        loss_value = loss_fn(label_pred.reshape(-1), label.reshape(-1).float())
        loss_value = torch.mean(loss_value * weights)
        loss_value = loss_value * prot_weight

        if optimizer:
            loss_value.backward()

            # Gradient accumulation
            if ((i + 1) % batch_prots == 0) or (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

        # Update loss etc.
        num_nodes = int(label.reshape(-1).size()[0])
        total_loss += loss_value.detach() * num_nodes
        total_count += num_nodes
        pred = torch.round(torch.sigmoid(label_pred.reshape(-1))).detach().cpu().numpy()
        true = label.reshape(-1).detach().cpu().numpy()
        total_correct += (pred == true).sum()
        t.set_description("%.4f" % float((pred == true).sum() / num_nodes))
        torch.cuda.empty_cache()

    return total_loss / total_count, total_correct / total_count


def loop_scannet_test(model_transformer, dataloader, device=None):
    # Initialize
    t = tqdm.tqdm(dataloader)
    total_loss, total_correct, total_count = 0, 0, 0
    acc_prots = []
    matthews_prots = []
    perplexity_prots = []
    pred_list = []

    total_label = torch.empty(0).to(device)
    total_label_pred = torch.empty(0).to(device)

    for i, (emb, label, _) in enumerate(t):
        # print(f"{i+1}/{len(t)}")

        # Transfer to GPU
        emb, label = emb.to(device), label.to(device)

        # Make Transformer prediction
        emb = emb.transpose(1, 0)  # B x N x H --> N x B x H
        label_pred = model_transformer(emb)

        # Apply sigmoid and concat
        total_label = torch.cat((total_label, label.reshape(-1)))
        total_label_pred = torch.cat((total_label_pred, label_pred.reshape(-1)))

    # Compute AUCPR
    precision, recall, thresholds = precision_recall_curve(
        total_label.detach().cpu().numpy(), total_label_pred.detach().cpu().numpy()
    )
    return precision, recall


def some_function_using_substitution_matrix():
    matrix = substitution_matrices.load("BLOSUM62")  # 新方式加载矩阵
    return matrix



def extract_single_msa(combined_a3m: str, target_name: str):
    """从合并的MSA文件中提取单个蛋白的MSA"""
    with open(combined_a3m) as f:
        keep = False
        result = []
        for line in f:
            if line.startswith('>'):
                keep = (line[1:].strip() == target_name)
            if keep:
                result.append(line)
        return [ (result[i][1:].strip(), result[i+1].strip())
               for i in range(0, len(result), 2) ]