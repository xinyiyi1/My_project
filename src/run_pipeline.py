import itertools
import os
import shutil
import sys
import subprocess
import tempfile

import torch
import models.gvp.data, models.gvp.models
import json
import os
import numpy as np
import torch_geometric
from functools import partial
import esm
import random
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import psutil
import glob
from esm import pretrained

from feature_processor import FeatureProcessor
from models.msa_transformer.model import MSATransformer


print = partial(print, flush=True)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from models.gvp.models import SSEmbGNN
import pickle
from visualization import (
    plot_mave_corr_vs_depth,
)
from helpers import (read_msa, loop_trainval, prepare_mave_val, prepare_proteingym)
import run_test_mave, run_test_proteingym, run_test_rocklin, run_test_clinvar

import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

#DEVICES = [3, 4, 5, 6, 7, 8, 9]
DEVICES = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(x) for x in DEVICES])




CHECKPOINT_DIR = "../output/train/checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_model(epoch, model, model_type, run_name):
    model_dir = f"../output/train/models/{model_type}/"
    os.makedirs(model_dir, exist_ok=True)
    path = f"{model_dir}{run_name}_{model_type}_{epoch}.pt"
    # 确保保存的是单GPU模型状态
    if isinstance(model, DDP):
        model = model.module
    torch.save(model.state_dict(), path)
    print(f"Saved {model_type} model to: {path}")

def save_checkpoint(epoch, model_gvp, model_msa, model_esm2, optimizer, best_corr_mave, run_name):
    if epoch % 10 != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'model_gvp_state_dict': model_gvp.state_dict(),
        'model_msa_state_dict': model_msa.state_dict(),
        'model_esm2_state_dict': model_esm2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_corr_mave': best_corr_mave,
    }
    torch.save(checkpoint, f"{CHECKPOINT_DIR}{run_name}_checkpoint_{epoch}.pt")


def load_checkpoint(run_name, device):
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}{run_name}_checkpoint_*.pt"))
    if not checkpoints:
        print(f"No checkpoints found in {CHECKPOINT_DIR}")
        return None
    latest_checkpoint = checkpoints[-1]
    print(f"Loading checkpoint from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=f"cuda:{device}")
    return checkpoint



def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2222"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(
    dataset,
    rank,
    world_size,
    batch_size=1,
    pin_memory=False,
    num_workers=0,
    train=False,
):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    if train == True:
        dataloader = torch_geometric.loader.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
        )
    else:
        dataloader = torch_geometric.loader.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            sampler=None,
        )
    return dataloader


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):

    # Setup the process groups
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    setup(rank, world_size)

    feat_processor = FeatureProcessor(
        esm_dim=1280,
        msa_dim=768,
        embed_dim=256,
        num_heads=8
    ).to(rank)
    feat_processor = DDP(
        feat_processor,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True
    )



    # Set fixed seed
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # Print name of run
    run_name = "final_cath"

    # Set initial parameters
    EPOCHS = 200
    EPOCH_FINETUNE_MSA = 100
    VAL_INTERVAL = 10
    BATCH_PROTS = 128 // len(DEVICES)
    LR_LIST = [1e-3, 1e-6]
    PATIENCE = 5

    EPOCH_FINETUNE_MSA = 100
    EPOCH_FINETUNE_ESM2 = 150




    ## Load CATH data
    print("Preparing CATH data")
    pdb_dir_cath = "/home/zhaom/my_proj/data/train/cath"
    msa_dir = "/home/zhaom/my_proj/data/train/cath/msa"
    pdb_dir_cath = "/home/zhaom/my_proj/data/train/cath"
    chain_set_jsonl_path = f"{pdb_dir_cath}/chain_set.jsonl"
    chain_set_splits_json_path = f"{pdb_dir_cath}/chain_set_splits.json"

    # Check if chain_set.jsonl already exists
    if not os.path.exists(chain_set_jsonl_path):
        subprocess.run([f"bash {pdb_dir_cath}/getCATH.sh"], shell=True)
        subprocess.run([f"mv chain_set.jsonl {chain_set_jsonl_path}"], shell=True)
        subprocess.run([f"mv chain_set_splits.json {chain_set_splits_json_path}"], shell=True)
    else:
        print(f"Skipping download as {chain_set_jsonl_path} already exists.")

    cath = models.gvp.data.CATHDataset(
        path=chain_set_jsonl_path,
        splits_path=chain_set_splits_json_path,
    )

    from esm import pretrained
    print("Loading ESM2 model...")
    model_esm2, esm_alphabet = pretrained.esm2_t33_650M_UR50D()  # 选择与硬件匹配的模型版本
    model_esm2 = model_esm2.to(rank)
    model_esm2.eval()
    esm_batch_converter = esm_alphabet.get_batch_converter()


    print("Precomputing ESM2 embeddings...")
    for entry in tqdm(cath.total):
        seq = entry["seq"]

        # 清洗序列中的非法字符
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        seq_clean = "".join([aa if aa in valid_aa else "" for aa in seq])
        if len(seq_clean) == 0:
            raise ValueError(f"序列 {entry['name']} 清洗后为空")

        # 生成 ESM2 嵌入
        with torch.no_grad():
            # 转换序列格式
            batch_labels, batch_strs, batch_tokens = esm_batch_converter([(entry['name'], seq_clean)])
            batch_tokens = batch_tokens.to(rank)

            # 提取特征
            results = model_esm2(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33].cpu()

        # 保存特征（去除CLS/SEP标记）
        entry["esm2_embeddings"] = embeddings[:, 1:-1, :].squeeze(0).numpy()  # [seq_len, 1280]
        if "esm2_embeddings" not in entry:
            raise RuntimeError(f"预处理失败: {entry['name']} 未生成嵌入")


    # 添加校验
    print("最终数据校验...")
    missing_final = [entry["name"] for entry in cath.total if "esm2_embeddings" not in entry]
    if missing_final:
        print(f"错误: 仍有 {len(missing_final)} 条数据缺失嵌入")
        print("示例:", missing_final[:5])
        raise ValueError("数据预处理不完整")
    else:
        print("所有数据预处理完成，保存缓存...")



    # 确保序列文件存在
    seqs_fasta_path = f"{pdb_dir_cath}/seqs.fasta"
    if not os.path.exists(seqs_fasta_path):
        print(f"Creating {seqs_fasta_path}...")
        with open(seqs_fasta_path, "w") as f:
            for entry in cath.total:
                f.write(f">{entry['name']}\n")
                f.write(f"{entry['seq']}\n")
    else:
        print(f"{seqs_fasta_path} already exists")

    # 确保MSA目录存在
    os.makedirs(f"{pdb_dir_cath}/msa", exist_ok=True)


    # Compute MSAs
    # TO DO: Add code example to extract sequences from CATH data set
    # to file: f"{pdb_dir_cath}/seqs.fasta"





    #sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    #colabfold_path = "/home/zhaom/anaconda3/envs/sse/bin/colabfold_search"  # 实际路径
    ##subprocess.run([
        #colabfold_path,
        #f"{pdb_dir_cath}/seqs.fasta",
        #os.path.expanduser("~/colabfold_databases"),  # 新路径
        #f"{pdb_dir_cath}/msa/"
    #])

    #subprocess.run(["python", "merge_and_sort_msas.py", "../data/train/cath/msa"])

    # 检查MSA文件是否存在
    missing = []
    for entry in cath.total:
        msa_file = f"{msa_dir}/{entry['name']}.a3m"
        if not os.path.exists(msa_file):
            missing.append(entry['name'])
    if missing:
        raise RuntimeError(f"Missing {len(missing)} MSA files (e.g. {missing[:3]})")

    # 加载MSA数据
    cache_file = f"{pdb_dir_cath}/data_with_msas.pkl"
    if os.path.exists(cache_file):
        print("Loading cached data with MSAs...")
        with open(cache_file, "rb") as fp:
            cath.total = pickle.load(fp)
    else:
        print("Adding pre-computed MSAs...")
        for i, entry in enumerate(tqdm(cath.total, desc="Loading MSAs")):
            msa_file = f"{msa_dir}/{entry['name']}.a3m"
            entry["msa"] = read_msa(msa_file)
            if not entry["msa"]:
                raise ValueError(f"空MSA文件: {msa_file}")
            if len(entry["msa"][0][1]) != len(entry["seq"]):
                raise ValueError(f"MSA序列长度不匹配: {entry['name']}")

        # 保存缓存
        with open(cache_file, "wb") as fp:
            pickle.dump(cath.total, fp)

    #Add MSAs
    #for i, entry in enumerate(cath,total):
        #print(f"Adding CATH MSAs: {i+1}/{len(cath.total)}")
        #entry["msa"] = read_msa(f"{pdb_dir_cath}/msa/{entry['name']}.a3m")

    # Checkpoint - save and load
    with open(f"{pdb_dir_cath}/data_with_msas.pkl", "wb") as fp:  # Pickling
        pickle.dump(cath.total, fp)

    with open(f"{pdb_dir_cath}/data_with_msas.pkl", "rb") as fp:  # Unpickling
        cath.total = pickle.load(fp)

    ## Filter data
    # Only keep entries where MSA and structucture sequence lengths match
    data = [
        entry for entry in cath.total
        if "msa" in entry and len(entry["seq"]) == len(entry["msa"][0][1])
    ]

    # Filter: Only keep entries without X in sequence
    data = [entry for entry in cath.total if "X" not in entry["seq"]]

    # Save all training and validation sequences in a fasta file to check homology
    cath.split()
    with open(f"../data/test/mave_val/structure/coords.json") as json_file:
        data_mave_val = json.load(json_file)

    with open(f"../data/test/proteingym/structure/coords.json") as json_file:
        data_proteingym = json.load(json_file)

    fh = open(f"../data/train/cath/seqs_cath.fasta", "w")
    for entry in cath.train:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")

    for entry in cath.val:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")

    for entry in data_mave_val:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")

    for entry in data_proteingym:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")
    fh.close()

    # Compute clusters of 95% sequence similarities between all training, validation and test proteins
    subprocess.run(
        [
            "cd-hit",
            "-i",
            "../data/train/cath/seqs_cath.fasta",
            "-o",
            "../data/train/cath/seqs_cath_homology.fasta",
            "-c",
            "0.95",
            "-n",
            "5",
            "-d",
            "999",
        ]
    )

    # Remove proteins from training data that has high sequence similarity with validation or test proteins
    val_prot_names = [entry["name"] for entry in cath.val]
    val_mave_prot_names = [entry["name"] for entry in data_mave_val]
    test_prot_names = [entry["name"] for entry in data_proteingym]
    valtest_prot_names = val_prot_names + val_mave_prot_names + test_prot_names

    fh = open("../data/train/cath/seqs_cath_homology.fasta.clstr", "r")
    cluster_dict = {}
    remove_list = []
    for line in fh.readlines():
        if line.startswith(">Cluster"):
            cluster_name = line
            cluster_dict[cluster_name] = []
        else:
            cluster_dict[cluster_name].append(line.split(">")[1].split("...")[0])

    for cluster_name, prot_names in cluster_dict.items():
        if len(prot_names) > 1 and any(
            valtest_prot_name in prot_names for valtest_prot_name in valtest_prot_names
        ):
            remove_list += prot_names
    remove_list = [
        prot_name for prot_name in remove_list if prot_name not in valtest_prot_names
    ]
    cath.train = [entry for entry in cath.train if entry["name"] not in remove_list]

    # Checkpoint - save and load
    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_train.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(cath.train, fp)
    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_val.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(cath.val, fp)

    # Prepare MAVE validation and ProteinGym test data
    prepare_mave_val()
    prepare_proteingym()
    #prepare_proteingym_bad()
    #prepare_proteingym_dafault()

    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_train.pkl", "rb"
    ) as fp:  # Unpickling
        cath.train = pickle.load(fp)
    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_val.pkl", "rb"
    ) as fp:  # Unpickling
        cath.val = pickle.load(fp)

    # Convert to graph data sets
    trainset = models.gvp.data.ProteinGraphData(cath.train)

    valset = models.gvp.data.ProteinGraphData(cath.val)

    ## Load and initialize MSA Transformer
    model_msa_pre, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    model_msa = model_msa_pre.to(rank)


    for param in model_msa.parameters():
        param.requires_grad = False

    # Load and initialize GVP
    node_dim = (256, 64)
    edge_dim = (32, 1)
    print("FeatureProcessor类型:", type(feat_processor))
    model_gvp = SSEmbGNN(
        node_in_dim=(6, 3),
        node_h_dim=node_dim,
        edge_in_dim=(32, 1),
        edge_h_dim=edge_dim,
        feature_processor=feat_processor
    )
    print("模型初始化成功!")
    model_gvp.to(rank)
    model_gvp = DDP(
        model_gvp,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
        static_graph=False,
    )

    # Initialize training modules
    train_loader = prepare(trainset, rank, world_size, train=True)

    val_loader = prepare(valset, rank, world_size)
    # 初始化优化器
    optimizer_groups = [
        # GVP参数
        {"params": model_gvp.parameters(), "lr": 1e-3}
    ]

    optimizer = torch.optim.AdamW(optimizer_groups)

    scaler = torch.cuda.amp.GradScaler()
    best_epoch, best_corr_mave = None, 0

    def add_param_group_safely(optimizer, new_params):
        """安全添加参数组，避免重复"""
        existing_params = set()
        for group in optimizer.param_groups:
            existing_params.update(set(group["params"]))
        new_params = [p for p in new_params if p not in existing_params]
        if new_params:
            optimizer.add_param_group({"params": new_params, "lr": 1e-5})

    # Initialize lists for monitoring loss
    epoch_list = []
    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []
    corr_mave_list, acc_mave_list = [], []










    checkpoint = load_checkpoint(run_name, rank)
    if checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        model_gvp.load_state_dict(checkpoint['model_gvp_state_dict'])
        model_msa.load_state_dict(checkpoint['model_msa_state_dict'])
        model_esm2.load_state_dict(checkpoint['model_esm2_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_corr_mave = checkpoint['best_corr_mave']
    else:
        start_epoch = 0
        best_corr_mave = 0
    best_epoch = 0
    patience_counter = 0
    best_corr_mave = 0
    for epoch in range(start_epoch, EPOCHS):
        # Check if we should fine-tune MSA Transformer row attention
        if epoch == EPOCH_FINETUNE_MSA:
            # 更新patience_counter的逻辑
            if epoch >= EPOCH_FINETUNE_MSA:
                if corr_mave > best_corr_mave:
                    best_corr_mave = corr_mave
                    patience_counter = 0
                    best_epoch = epoch
                    # 保存最佳 epoch 到文件
                    with open(f"../output/train/models/best_epoch.txt", "w") as f:
                        f.write(str(best_epoch))
                else:
                    patience_counter += 1
            if patience_counter == PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
            # 解冻MSA行注意力层参数
            msa_train_params = []
            for name, param in model_msa.named_parameters():
                if "row_self_attention" in name:
                    param.requires_grad = True
                    msa_train_params.append(param)

            model_msa = DDP(
                model_msa,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
                static_graph=False,
            )
            # 添加新参数组
            add_param_group_safely(optimizer, msa_train_params)

            if epoch >= EPOCH_FINETUNE_ESM2:
                # 解冻ESM-2最后2层
                esm2_train_params = []
                for layer in model_esm2.layers[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        esm2_train_params.append(param)

                # 添加新参数组
                add_param_group_safely(optimizer, esm2_train_params, lr=1e-6)
            BATCH_PROTS = 2048 // len(DEVICES)

        # If we are using DistributedSampler, we need to tell it which epoch this is
        train_loader.sampler.set_epoch(epoch)

        # Train loop
        model_msa.train()
        model_gvp.train()

        loss_train, acc_train = loop_trainval(
            model_msa,
            model_gvp,
            msa_batch_converter,
            train_loader,
            BATCH_PROTS,
            epoch,
            rank,
            EPOCH_FINETUNE_MSA,
            feat_processor,
            optimizer=optimizer,
            scaler=scaler,
        )
        ## Gather and save training metrics for epoch
        # OBS: This cannot be placed within validation loop or we get hangs
        loss_train = loss_train.type(torch.float32)
        loss_train_all_gather = [torch.zeros(1, device=rank) for _ in range(world_size)]
        dist.all_gather(loss_train_all_gather, loss_train)

        # Validation loop
        if rank == 0:
            # 初始化当前epoch的指标
            corr_mave = 0.0
            acc_mave = 0.0
            if epoch % VAL_INTERVAL == 0:

                os.makedirs("../output/train/models/esm2", exist_ok=True)
                # Save model
                os.makedirs("../output/train/models/esm2", exist_ok=True)
                path_msa = f"../output/train/models/msa_transformer/{run_name}_msa_transformer_{epoch}.pt"
                path_gvp = f"../output/train/models/gvp/{run_name}_gvp_{epoch}.pt"
                path_esm2 = f"../output/train/models/esm2/{run_name}_esm2_{epoch}.pt"
                path_optimizer = f"../output/train/models/optimizer/{run_name}_adam_{epoch}.pt"

                # 保存检查点
                save_checkpoint(epoch, model_gvp, model_msa, model_esm2, optimizer, best_corr_mave, run_name)

                # 保存 MSA Transformer 模型
                def get_full_state_dict(model):
                    """安全获取完整状态字典"""
                    if isinstance(model, (nn.parallel.DistributedDataParallel, DDP)):
                        return model.module.state_dict()
                    return model.state_dict()

                # 在保存模型的地方添加：
                state_dict_msa = get_full_state_dict(model_msa)
                if state_dict_msa is None:
                    print(f"epoch {epoch}的MSA模型state_dict为空，跳过保存")
                else:
                    # 使用临时文件确保写入完整
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        torch.save(state_dict_msa, tmp_file.name)
                        shutil.move(tmp_file.name, path_msa)
                    print(f"已保存MSA模型到: {path_msa} (大小: {os.path.getsize(path_msa) / 1024:.1f}KB)")


                # 保存模型
                torch.save(get_full_state_dict(model_msa), path_msa)

                # 正常保存其他模型
                torch.save(model_gvp.state_dict(), path_gvp)
                torch.save(model_esm2.state_dict(), path_esm2)
                torch.save(optimizer.state_dict(), path_optimizer)

                print(f"[Epoch {epoch}] Saved models to: {path_gvp}, {path_msa}, {path_esm2}")

                with torch.no_grad():
                    # Do training validation
                    model_msa.eval()
                    model_gvp.eval()
                    feat_processor.eval()


                    loss_val, acc_val = loop_trainval(
                        model_msa,
                        model_gvp,
                        msa_batch_converter,
                        val_loader,
                        BATCH_PROTS,
                        epoch,
                        rank,
                        EPOCH_FINETUNE_MSA,
                        feat_processor,
                    )

                    if epoch >= EPOCH_FINETUNE_MSA:
                        # Do validation on MAVE set
                        corr_mave, acc_mave = run_test_mave.test(
                            run_name,
                            epoch,
                            device=rank,
                        )
                    else:
                        corr_mave, acc_mave = 0.0, 0.0

                # 更新最佳结果
                if corr_mave > best_corr_mave:
                    best_corr_mave = corr_mave
                    best_epoch = epoch
                    patience_counter = 0
                    # 保存最佳epoch
                    with open(f"../output/train/models/best_epoch.txt", "w") as f:
                        f.write(str(best_epoch))
                else:
                    patience_counter += 1

                # Save validation results
                epoch_list.append(epoch)
                loss_train_list.append(
                    torch.mean(torch.stack(loss_train_all_gather)).to("cpu").item()
                )
                loss_val_list.append(loss_val.to("cpu").item())
                acc_val_list.append(acc_val)
                corr_mave_list.append(corr_mave)
                acc_mave_list.append(corr_mave)

                metrics = {
                    "epoch": epoch_list,
                    "loss_train": loss_train_list,
                    "loss_val": loss_val_list,
                    "acc_val": acc_val_list,
                    "corr_mave": corr_mave_list,
                    "acc_mave": acc_mave_list,
                }
                with open(f"../output/train/metrics/{run_name}_metrics", "wb") as f:
                    pickle.dump(metrics, f)
        patience_counter_tensor = torch.tensor([patience_counter], device=rank)
        dist.all_reduce(patience_counter_tensor, op=dist.ReduceOp.MAX)
        patience_counter = patience_counter_tensor.item()

        should_stop = (patience_counter >= PATIENCE)
        should_stop_tensor = torch.tensor([should_stop], device=rank)
        dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)

        dist.barrier()

        if should_stop_tensor.item():
            if rank == 0:
                print(f"Early stopped at epoch {epoch}")
            cleanup()
            break
    #     # 每epoch结束同步
    #     dist.barrier()
    #     # if patience_counter == PATIENCE:
    #     #     break
    #     #
    #     # # Create barrier after each epoch
    #     # dist.barrier()
    #
    # # Clean up
    # cleanup()

    # MAVE val set
    print("Starting MAVE val predictions")
    run_test_mave.test(run_name, best_epoch, get_only_ssemb_metrics=False, device=rank)
    plot_mave_corr_vs_depth()
    print("Finished MAVE val predictions")

    # ProteinGym test set
    print("Starting ProteinGym test")
    run_test_proteingym.test(run_name, best_epoch, device=rank)
    print("Finished ProteinGym test")

    # Rocklin test set
    print("Starting Rocklin test")
    run_test_rocklin.test(run_name, best_epoch, num_ensemble=5, device=rank)
    print("Finished Rocklin test")

    # ClinVar test set
    print("Starting ClinVar test")
    run_test_clinvar.test(run_name, best_epoch, num_ensemble=5, device=rank)
    print("Finished ClinVar test")

if __name__ == "__main__":
    world_size = len(DEVICES)
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )