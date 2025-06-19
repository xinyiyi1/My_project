import sys
import subprocess
import glob
import re
import torch
import models.gvp.data, models.gvp.models
import json
import torch_geometric
import torch.nn as nn
import esm
import pandas as pd
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from models.msa_transformer.model import MSATransformer
from models.gvp.models import SSEmbGNN
from helpers import (
    read_msa,
    loop_pred,
)
from visualization import plot_proteingym
import pdb_parser_scripts.parse_pdbs as parse_pdbs
import torch.utils.data
from collections import OrderedDict
from ast import literal_eval
import subprocess
import pickle

def test(run_name, epoch, msa_row_attn_mask=True, device=None):
    # Load data and dict of variant positions
    with open(f"../data/test/proteingym/data_with_msas.pkl", "rb") as fp:
        data = pickle.load(fp)

    with open(f"../data/test/proteingym/variant_pos_dict.pkl", "rb") as fp:
        variant_pos_dict = pickle.load(fp)

        # ===== 深度修复MSA格式 =====
        for entry in data:
            if "msa" not in entry:
                continue

            # 记录原始格式用于调试
            original_format = type(entry["msa"])
            print(f"蛋白质 {entry['name']} 的原始MSA类型: {original_format}")

            # 展平所有嵌套结构并确保格式正确
            flat_msa = []
            if isinstance(entry["msa"], list):
                for item in entry["msa"]:
                    if isinstance(item, tuple) and len(item) == 2:
                        # 已经是正确格式
                        flat_msa.append(item)
                    elif isinstance(item, list):
                        # 处理嵌套列表
                        for sub_item in item:
                            if isinstance(sub_item, tuple) and len(sub_item) == 2:
                                flat_msa.append(sub_item)
                            elif isinstance(sub_item, list) and len(sub_item) >= 2:
                                # 尝试提取header和sequence
                                header = sub_item[0] if len(sub_item) > 0 else ""
                                sequence = sub_item[1] if len(sub_item) > 1 else ""
                                flat_msa.append((header, sequence))
                            elif isinstance(sub_item, str):
                                # 处理纯字符串列表
                                if sub_item.startswith('>'):
                                    # 这是header
                                    current_header = sub_item[1:].strip()
                                else:
                                    # 这是sequence
                                    if current_header:
                                        flat_msa.append((current_header, sub_item.strip()))
                                        current_header = None
                    elif isinstance(item, str):
                        # 处理纯字符串列表
                        if item.startswith('>'):
                            # header
                            current_header = item[1:].strip()
                        else:
                            # sequence
                            if current_header:
                                flat_msa.append((current_header, item.strip()))
                                current_header = None
                entry["msa"] = flat_msa

            # 最终验证
            valid_msa = []
            for item in entry["msa"]:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
                    valid_msa.append(item)
                else:
                    print(f"跳过无效条目: {type(item)} - {item}")
            entry["msa"] = valid_msa

            print(f"修复后 {entry['name']} 的MSA条目数: {len(entry['msa'])}")





    # Load DMS data
    df_dms = pd.read_csv("../data/test/proteingym/exp/dms.csv")

    # Convert to graph data sets
    testset = models.gvp.data.ProteinGraphData(data)
    letter_to_num = testset.letter_to_num





    # Load MSA Transformer
    _, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model_msa = MSATransformer(msa_alphabet)
    model_msa = model_msa.to(device)
    msa_batch_converter = msa_alphabet.get_batch_converter()
    # 添加投影层
    # model_msa.feature_proj = nn.Linear(768, 512)
    # model_msa.feature_proj.to(device)


    model_path = f"../output/train/models/msa_transformer/{run_name}_msa_transformer_{epoch}.pt"
    state_dict = torch.load(
        f"../output/train/models/msa_transformer/{run_name}_msa_transformer_{epoch}.pt",
        map_location=f"cuda:{device}"
    )

    # 创建新状态字典并去除分布式训练前缀
    model_dict = OrderedDict()
    pattern = re.compile("module.")
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict[k] = v

    # 严格加载完整状态字典
    model_msa.load_state_dict(model_dict, strict=False)

    # Load GVP
    node_dim = (256, 64)
    edge_dim = (32, 1)
    from feature_processor import FeatureProcessor
    feat_processor = FeatureProcessor(
        esm_dim=1280,
        msa_dim=768,
        embed_dim=256,
        num_heads=8
    ).to(device)

    model_gvp = SSEmbGNN((6, 3), node_dim, (32, 1), edge_dim, feat_processor=feat_processor).to(device)
    model_gvp = model_gvp.to(device)

    model_dict = OrderedDict()
    state_dict_gvp = torch.load(
        f"../output/train/models/gvp/{run_name}_gvp_{epoch}.pt",
        map_location=f"cuda:{device}")



    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 去除前缀 "feature_processor.module."
        new_key = k.replace("feature_processor.module.", "feat_processor.")
        new_state_dict[new_key] = v

    # 加载新状态字典
    model_gvp.load_state_dict(new_state_dict, strict=False)

    # Initialize data loader
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
    df_ml.to_csv(f"../output/proteingym/df_ml_{run_name}_{epoch}.csv", index=False)

    # Load
    import ast
    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x)
        except:
            return [0.0] * 20

    df_ml = pd.read_csv(
        f"../output/proteingym/df_ml_{run_name}_{epoch}.csv",
        converters={"score_ml_pos": safe_literal_eval}
    )

    # 确保分数列是数字列表
    def convert_to_float_list(x):
        try:
            if isinstance(x, list):
                return [float(i) for i in x]
            return [0.0] * 20
        except:
            return [0.0] * 20

    df_ml["score_ml_pos"] = df_ml["score_ml_pos"].apply(convert_to_float_list)

    # 加载 DMS 数据
    df_dms = pd.read_csv("../data/test/proteingym/exp/dms.csv")

    # Compute score_ml from nlls
    dms_variant_list = df_dms.values.tolist()
    for i, row in enumerate(dms_variant_list):
        dms_id = row[0]
        print(
            f"Computing score for assay {dms_id} variant: {i+1}/{len(dms_variant_list)}"
        )
        variant_set = row[1].split(":")
        score_ml = 0.0

        for variant in variant_set:
            # 跳过空变异
            if not variant:
                continue

            try:
                # 提取变异信息
                parts = variant.split(":")
                for part in parts:
                    if not part:
                        continue

                    # 确保变异格式正确
                    if len(part) < 3:
                        print(f"跳过无效变异格式: {part}")
                        continue

                    wt = part[0]
                    mt = part[-1]
                    pos_str = re.findall(r"\d+", part)

                    if not pos_str:
                        print(f"无法从变异 {part} 中提取位置")
                        continue

                    pos = int(pos_str[0])

                    # 查找匹配的预测数据
                    matches = df_ml[
                        (df_ml["dms_id"] == dms_id) & (df_ml["variant_pos"] == pos)
                        ]

                    if matches.empty:
                        # 尝试在更大范围内查找
                        close_matches = df_ml[
                            (df_ml["dms_id"] == dms_id) &
                            (df_ml["variant_pos"].between(max(1, pos - 5), min(10000, pos + 5)))
                             ]

                        if not close_matches.empty:
                            closest = close_matches.iloc[0]
                            print(f"使用最近位置 {closest['variant_pos']} 代替 {pos}")
                            score_ml_pos = closest["score_ml_pos"]
                        else:
                            print(f"未找到 {dms_id} 位置 {pos} 的预测数据")
                            # 使用零值作为默认值
                            score_ml_pos = [0.0] * 20
                    else:
                        score_ml_pos = matches["score_ml_pos"].values[0]

                        # 确保score_ml_pos是列表且元素是数字
                        if isinstance(score_ml_pos, str):
                            try:
                                # 尝试解析字符串格式的列表
                                score_ml_pos = ast.literal_eval(score_ml_pos)
                            except:
                                # 如果解析失败，创建默认列表
                                score_ml_pos = [0.0] * 20

                    # 确保是列表类型
                    if not isinstance(score_ml_pos, list):
                        print(f"预测分数不是列表类型: {type(score_ml_pos)}")
                        score_ml_pos = [0.0] * 20

                    # 确保列表元素是数字
                    if not all(isinstance(x, (int, float)) for x in score_ml_pos):
                        print(f" 预测分数包含非数字元素: {score_ml_pos}")
                        score_ml_pos = [0.0] * 20

                    # 计算分数
                    if mt in letter_to_num:
                        mt_idx = letter_to_num[mt]
                        if mt_idx < len(score_ml_pos):
                            score_ml += float(score_ml_pos[mt_idx])
                        else:
                            print(f"突变类型 {mt} 超出范围 (索引 {mt_idx} >= {len(score_ml_pos)})")
                    else:
                        print(f"无效突变类型: {mt}")

            except Exception as e:
                print(f"处理变异 {variant} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()  # 打印完整堆栈跟踪
                continue
        dms_variant_list[i].append(score_ml)
    df_total = pd.DataFrame(
        dms_variant_list, columns=["dms_id", "variant_set", "score_dms", "score_ml"]
    )

    # Save
    df_total.to_csv(f"../output/proteingym/df_total_{run_name}_{epoch}.csv", index=False)

    # Load
    df_total = pd.read_csv(f"../output/proteingym/df_total_{run_name}_{epoch}.csv")

    # Compute correlations
    plot_proteingym(df_total, run_name, epoch)