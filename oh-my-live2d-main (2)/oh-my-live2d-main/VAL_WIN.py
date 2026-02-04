import torch
from CAM_S import CAMPPlus
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 定义超参数
data_dir = r"D:\比赛视频Videos\sopran_cutted"
val_batch_size = 16
num_workers = 0  # 在Windows上设置为0避免多进程问题
num_classes = 50
pretrained_weights = "MODEL_WEIGHT/logs_ddnet_sopran/2025-08-28_17-36-59/best_model.pth"
output_dir = r"D:\比赛视频Videos\sopranres"

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)


class CustomDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = Path(data_dir)
        self.MFCC_data_dir = self.data_dir / 'MFCC_Output'

        # 使用更高效的文件查找方式
        self.mfcc_files = list(self.MFCC_data_dir.glob('*_MFCC.xlsx'))
        self.transforms = transforms

        print(f"Found {len(self.mfcc_files)} MFCC files for inference")

        # 预加载文件名映射，避免重复处理
        self.sample_ids = [f.stem.replace('_MFCC', '') for f in self.mfcc_files]

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
        mfcc_file = self.mfcc_files[idx]
        sample_id = self.sample_ids[idx]

        try:
            # 使用更高效的pandas读取方式
            MFCC_data = pd.read_excel(mfcc_file, header=None, engine="openpyxl").values
            MFCC_tensor = torch.tensor(MFCC_data, dtype=torch.float32).unsqueeze(0)

            if self.transforms is not None:
                MFCC_tensor = self.transforms(MFCC_tensor)

            return MFCC_tensor, sample_id

        except Exception as e:
            print(f"Error loading {mfcc_file}: {e}")
            # 返回空数据或跳过
            return torch.zeros((1, 13, 100)), sample_id  # 假设MFCC维度


def load_model(checkpoint_path, num_classes, device):
    """加载模型的独立函数"""
    model = CAMPPlus(
        num_class=num_classes,
        input_size=1,
        embd_dim=8192,
        growth_rate=64,
        bn_size=4,
        init_channels=128,
        config_str='batchnorm-relu'
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained weights not found at {checkpoint_path}")

    # 更健壮的权重加载
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    print(f"✅ Loaded pretrained weights from {checkpoint_path}")
    return model


def save_predictions_to_excel(net, val_loader, device, output_path):
    """保存预测结果到Excel文件"""
    net.eval()
    all_preds = []
    all_filenames = []

    # 技巧名称列表
    tech_names = ["vibrato", "throat", "position", "open", "clean",
                  "resonate", "unify", "falsetto", "chset", "nasal"]

    with torch.no_grad():
        for batch_idx, (im, sample_ids) in enumerate(val_loader):
            im = im.to(device)

            try:
                # 添加模型推理的异常处理
                output, _, _ = net(im)
                output = output.view(output.shape[0], 5, 10)  # (B, 5, 10)
                preds = output.argmax(dim=1).cpu().numpy() + 1  # 转为 1-5

                all_preds.append(preds)
                all_filenames.extend(sample_ids)

                # 进度显示
                print(f"Processed batch {batch_idx + 1}/{len(val_loader)} - {len(sample_ids)} samples")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

    # 检查是否有预测结果
    if not all_preds:
        print("⚠️ 没有获取到预测数据")
        return

    # 合并所有batch的预测结果
    all_preds = np.concatenate(all_preds, axis=0)

    # 创建DataFrame
    df_results = pd.DataFrame({
        "Filename": all_filenames
    })

    # 添加每个技巧的预测结果
    for i, tech_name in enumerate(tech_names):
        df_results[tech_name] = all_preds[:, i]

    # 按文件名排序并保存
    df_results = df_results.sort_values(by="Filename")
    df_results.to_excel(output_path, index=False)

    print(f"✅ 推理完成！共处理 {len(df_results)} 个样本")
    print(f"✅ 结果已保存至：{output_path}")

    # 显示统计信息
    print("\n📊 预测结果统计：")
    for tech in tech_names:
        print(f"{tech}: 均值={df_results[tech].mean():.2f}, 标准差={df_results[tech].std():.2f}")


def main():
    """主函数"""
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # 创建数据集和加载器
    print("创建数据加载器...")
    val_dataset = CustomDataset(data_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,  # 在Windows上设置为0
        pin_memory=False  # 当num_workers=0时，pin_memory应该为False
    )

    # 加载模型
    print("加载模型...")
    model = load_model(pretrained_weights, num_classes, device)

    # 执行推理
    print("开始推理...")
    output_path = os.path.join(output_dir, "competition_tenor.xlsx")
    save_predictions_to_excel(model, val_loader, device, output_path)


if __name__ == "__main__":
    # 在Windows上必须使用这个保护
    import multiprocessing

    multiprocessing.freeze_support()  # 对于打包成exe的情况

    main()
    print("程序执行完毕！")