# segPANDA200数据集训练代码（修正版）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from models import unet_fpn

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数设置
BATCH_SIZE = 32
NUM_CLASSES_SEG = 3  # 分割类别数（背景、前景1、前景2）
NUM_CLASSES_CLS = 6  # 分类类别数（0-5共6个类别）
EPOCHS = 200
LEARNING_RATE = 0.001
DATA_ROOT = 'D:/病理图像/segPANDA200'  # 数据根目录


# 自定义segPANDA200数据集类
class SegPANDA200Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None):
        """
        Args:
            root_dir: 数据根目录
            split: 'train' 或 'test'
            transform: 图像预处理
            mask_transform: 掩码预处理
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # 存储类别标签

        # 主文件夹下是train和test
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"目录不存在: {split_dir}")

        # train/test下分别是0,1,2,3,4,5代表不同的类
        class_dirs = sorted([d for d in os.listdir(split_dir)
                             if os.path.isdir(os.path.join(split_dir, d)) and d.isdigit()])

        print(f"在{split}集中找到类别文件夹: {class_dirs}")

        for class_idx_str in class_dirs:
            class_idx = int(class_idx_str)
            class_path = os.path.join(split_dir, class_idx_str)

            # 每个类别下有很多文件夹，每个文件夹代表一个病例/样本
            sample_dirs = [d for d in os.listdir(class_path)
                           if os.path.isdir(os.path.join(class_path, d))]

            print(f"  类别 {class_idx_str} 中有 {len(sample_dirs)} 个病例文件夹")

            for sample_dir in sample_dirs:
                sample_path = os.path.join(class_path, sample_dir)

                # 直接在该文件夹下查找图像和mask文件
                all_files = os.listdir(sample_path)

                # 找出所有图像文件（不以_mask.png结尾的.jpg/.png文件）
                image_files = []
                for f in all_files:
                    if f.endswith(('.jpg', '.jpeg', '.png')):
                        # 检查是否为mask文件
                        if not f.endswith('_mask.png') and '_mask.' not in f:
                            image_files.append(f)

                for img_file in image_files:
                    img_path = os.path.join(sample_path, img_file)

                    # 对应的mask文件命名规则：原文件名 + '_mask.png'
                    # 例如: "0ac7f7d..._level0_x3288_y9864.jpg" -> "0ac7f7d..._level0_x3288_y9864_mask.png"
                    file_name_without_ext = os.path.splitext(img_file)[0]
                    mask_file = file_name_without_ext + '_mask.png'
                    mask_path = os.path.join(sample_path, mask_file)

                    # 检查mask文件是否存在
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        self.labels.append(class_idx)
                    else:
                        # 如果找不到标准命名的mask，尝试其他可能的命名方式
                        mask_file = img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png').replace('.png',
                                                                                                                '_mask.png')
                        mask_path = os.path.join(sample_path, mask_file)

                        if os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                            self.labels.append(class_idx)
                        else:
                            print(f"警告: 找不到对应的mask文件 {img_file} 在 {sample_path}")

        print(f"\n{split}集加载完成，共 {len(self.image_paths)} 个样本")
        if len(self.image_paths) > 0:
            # 统计类别分布
            unique_labels = set(self.labels)
            print(f"类别分布:")
            for i in sorted(unique_labels):
                count = self.labels.count(i)
                print(f"  类别 {i}: {count} 个样本 ({count / len(self.labels) * 100:.1f}%)")

            # 检查mask文件的像素值分布（随机检查几个）
            print(f"\n检查mask文件像素值分布:")
            for i in range(min(3, len(self.mask_paths))):
                try:
                    mask_sample = Image.open(self.mask_paths[i]).convert('L')
                    mask_array = np.array(mask_sample)
                    unique_vals = np.unique(mask_array)
                    print(f"  mask示例 {i + 1}: 唯一像素值 {unique_vals}")
                except:
                    pass
        else:
            print(f"警告: {split}集中没有加载到任何样本！")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # 加载分割标签（灰度图）
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 获取类别标签
        label = self.labels[idx]

        # 应用变换
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 确保mask是long类型，像素值应为0,1,2
        mask = mask.squeeze(0).long()  # [H, W]

        return image, mask, label


# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 掩码的变换（不需要归一化，使用最近邻插值保持边界清晰）
mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()  # 会将像素值归一化到[0,1]，但因为是整数，所以没问题
])


# 训练函数（同时训练分割和分类）
def train_epoch_multi_task(model, loader, seg_criterion, cls_criterion, optimizer, cls_weight=0.3):
    model.train()
    total_seg_loss = 0
    total_cls_loss = 0
    total_loss = 0

    total_seg_correct = 0
    total_pixels = 0
    total_cls_correct = 0
    total_samples = 0

    for images, masks, labels in tqdm(loader, desc='训练'):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)

        # 前向传播
        seg_output, cls_output, _, _ = model(images)

        # 计算分割损失
        seg_loss = seg_criterion(seg_output, masks)  # masks已经是long类型

        # 计算分类损失
        cls_loss = cls_criterion(cls_output, labels)

        # 总损失
        loss = seg_loss + cls_weight * cls_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item()
        total_loss += loss.item()

        # 计算分割准确率（逐像素）
        seg_pred = torch.argmax(seg_output, dim=1)
        total_seg_correct += (seg_pred == masks).sum().item()
        total_pixels += masks.numel()

        # 计算分类准确率
        cls_pred = torch.argmax(cls_output, dim=1)
        total_cls_correct += (cls_pred == labels).sum().item()
        total_samples += labels.size(0)

    avg_seg_loss = total_seg_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_total_loss = total_loss / len(loader)
    seg_acc = total_seg_correct / total_pixels
    cls_acc = total_cls_correct / total_samples

    return avg_total_loss, avg_seg_loss, avg_cls_loss, seg_acc, cls_acc


# 验证函数（同时验证分割和分类）
def validate_multi_task(model, loader, seg_criterion, cls_criterion, cls_weight=0.3):
    model.eval()
    total_seg_loss = 0
    total_cls_loss = 0
    total_loss = 0

    total_seg_correct = 0
    total_pixels = 0
    total_dice = 0
    total_iou = 0

    total_cls_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, masks, labels in tqdm(loader, desc='验证'):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)

            # 前向传播
            seg_output, cls_output = model(images)

            # 计算损失
            seg_loss = seg_criterion(seg_output, masks)
            cls_loss = cls_criterion(cls_output, labels)
            loss = seg_loss + cls_weight * cls_loss

            # 统计损失
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()

            # 计算分割准确率
            seg_pred = torch.argmax(seg_output, dim=1)
            total_seg_correct += (seg_pred == masks).sum().item()
            total_pixels += masks.numel()

            # 计算每个类别的Dice系数
            for class_idx in range(1, NUM_CLASSES_SEG):  # 跳过背景类(0)
                pred_class = (seg_pred == class_idx).float()
                target_class = (masks == class_idx).float()

                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum() - intersection

                if target_class.sum() == 0 and pred_class.sum() == 0:
                    dice = 1.0
                elif target_class.sum() == 0:
                    dice = 0.0
                else:
                    dice = (2. * intersection) / (pred_class.sum() + target_class.sum() + 1e-6)

                total_dice += dice.item()

                # 计算IoU
                iou = intersection / (union + 1e-6) if union > 0 else 0.0
                total_iou += iou.item()

            # 计算分类准确率
            cls_pred = torch.argmax(cls_output, dim=1)
            total_cls_correct += (cls_pred == labels).sum().item()
            total_samples += labels.size(0)

    avg_seg_loss = total_seg_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_total_loss = total_loss / len(loader)
    seg_acc = total_seg_correct / total_pixels
    avg_dice = total_dice / (len(loader.dataset) * (NUM_CLASSES_SEG - 1))  # 平均每个类别
    avg_iou = total_iou / (len(loader.dataset) * (NUM_CLASSES_SEG - 1))
    cls_acc = total_cls_correct / total_samples

    return avg_total_loss, avg_seg_loss, avg_cls_loss, seg_acc, avg_dice, avg_iou, cls_acc


# 多类别Dice损失函数
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] 模型输出的logits
            target: [B, H, W] 真实标签 (0, 1, 2)
        """
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        # 计算每个类别的Dice损失
        dice_loss = 0
        for class_idx in range(pred.shape[1]):  # 包括背景类
            pred_class = pred[:, class_idx, :, :]
            target_class = target_onehot[:, class_idx, :, :]

            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice.mean())

        return dice_loss / pred.shape[1]  # 返回平均Dice损失


# 组合分割损失函数
class CombinedSegLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = MultiClassDiceLoss()

    def forward(self, pred, target):
        return (self.ce_weight * self.ce_loss(pred, target) +
                self.dice_weight * self.dice_loss(pred, target))


if __name__ == '__main__':
    # 创建训练集和验证集
    print("=" * 60)
    print("正在加载训练集...")
    train_dataset = SegPANDA200Dataset(
        root_dir=DATA_ROOT,
        split='train',
        transform=train_transform,
        mask_transform=mask_transform
    )

    print("\n" + "=" * 60)
    print("正在加载测试集（作为验证集）...")
    val_dataset = SegPANDA200Dataset(
        root_dir=DATA_ROOT,
        split='test',
        transform=val_transform,
        mask_transform=mask_transform
    )

    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError("训练集为空！请检查数据路径和文件夹结构。")
    if len(val_dataset) == 0:
        raise ValueError("验证集为空！请检查数据路径和文件夹结构。")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n数据集统计:")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"分割类别数: {NUM_CLASSES_SEG}")
    print(f"分类类别数: {NUM_CLASSES_CLS}")

    # 初始化模型
    model = unet_fpn.UNet_fpn(num_classes=NUM_CLASSES_SEG,
                              in_channels=3).to(device)

    # 损失函数
    seg_criterion = CombinedSegLoss(ce_weight=1.0, dice_weight=1.0)
    cls_criterion = nn.CrossEntropyLoss()

    # 分类损失的权重（可以调整）
    CLS_WEIGHT = 0.3

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练循环
    best_combined_score = 0
    best_model_filename = None

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print("-" * 60)

        # 训练
        train_total_loss, train_seg_loss, train_cls_loss, train_seg_acc, train_cls_acc = \
            train_epoch_multi_task(model, train_loader, seg_criterion, cls_criterion, optimizer, CLS_WEIGHT)

        print(f"训练 - 总损失: {train_total_loss:.4f}, 分割损失: {train_seg_loss:.4f}, 分类损失: {train_cls_loss:.4f}")
        print(f"训练 - 分割准确率: {train_seg_acc:.4f}, 分类准确率: {train_cls_acc:.4f}")

        # 验证
        val_total_loss, val_seg_loss, val_cls_loss, val_seg_acc, val_dice, val_iou, val_cls_acc = \
            validate_multi_task(model, val_loader, seg_criterion, cls_criterion, CLS_WEIGHT)

        print(f"验证 - 总损失: {val_total_loss:.4f}, 分割损失: {val_seg_loss:.4f}, 分类损失: {val_cls_loss:.4f}")
        print(f"验证 - 分割准确率: {val_seg_acc:.4f}, 平均Dice: {val_dice:.4f}, 平均IoU: {val_iou:.4f}, 分类准确率: {val_cls_acc:.4f}")

        scheduler.step()

        # 计算综合得分（Dice和分类准确率的加权平均）
        combined_score = val_dice * 0.5 + val_cls_acc * 0.5

        # 每轮都保存模型权重
        model_filename = f'segpanda200_epoch_{epoch + 1:03d}_dice_{val_dice:.4f}_iou_{val_iou:.4f}_clsacc_{val_cls_acc:.4f}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'combined_score': combined_score,
            'dice': val_dice,
            'iou': val_iou,
            'cls_acc': val_cls_acc,
            'seg_acc': val_seg_acc,
            'train_loss': train_total_loss,
            'val_loss': val_total_loss,
        }, model_filename)
        print(f"✓ 已保存模型: {model_filename}")

        # 保存最佳模型
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_model_filename = f'segpanda200_best_model_dice_{val_dice:.4f}_clsacc_{val_cls_acc:.4f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_combined_score': best_combined_score,
                'best_dice': val_dice,
                'best_iou': val_iou,
                'best_cls_acc': val_cls_acc,
            }, best_model_filename)
            print(f"✓ 保存最佳模型: {best_model_filename} (Dice: {val_dice:.4f}, 分类准确率: {val_cls_acc:.4f})")

    print(f"\n训练完成！最佳综合得分: {best_combined_score:.4f}")
    if best_model_filename:
        print(f"最佳模型: {best_model_filename}")


    # 测试输出形状
    def test_output_shape():
        model.eval()
        sample_input = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            seg_output, cls_output = model(sample_input)
        print(f"\n输出形状验证:")
        print(f"输入形状: {sample_input.shape}")
        print(f"分割输出形状: {seg_output.shape}  # 应为 [4, {NUM_CLASSES_SEG}, 224, 224]")
        print(f"分类输出形状: {cls_output.shape}  # 应为 [4, {NUM_CLASSES_CLS}]")
        return seg_output, cls_output


    test_output_shape()