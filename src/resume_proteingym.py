import torch
from run_test_proteingym import test as test_proteingym

if  __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = "final_cath"
    best_epoch = 130  # 替换为实际最佳epoch
    print('Resuming ProteinGym test')
    test_proteingym(run_name, best_epoch, device=0)