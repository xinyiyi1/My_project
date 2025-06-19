import torch
from run_test_mave import test as run_test_mave_test
from helpers import prepare_mave_val
from visualization import plot_mave_corr_vs_depth
import pickle

if __name__ == "__main__":
    # 重新生成包含ESM2嵌入的MAVE验证数据
    print("=" * 80)
    print("Preparing MAVE validation data with ESM2 embeddings...")
    prepare_mave_val()

    # 验证数据是否包含ESM2嵌入
    with open(f"../data/test/mave_val/data_with_msas.pkl", "rb") as fp:
        data = pickle.load(fp)

    missing_emb = [e['name'] for e in data if 'esm2_embeddings' not in e]
    if missing_emb:
        print(f"警告: {len(missing_emb)}个蛋白质缺少ESM2嵌入")
        print("示例:", missing_emb[:3])
    else:
        print("所有蛋白质都已成功生成ESM2嵌入")

    print("=" * 80)

    # 运行测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = "final_cath"
    best_epoch = 90

    print("=" * 80)
    print("Starting MAVE val predictions")
    run_test_mave_test(run_name, best_epoch, get_only_ssemb_metrics=False, device=device)
    print("=" * 80)

    print("Generating correlation plots...")
    plot_mave_corr_vs_depth()
    print("=" * 80)
    print("Finished MAVE val predictions")
