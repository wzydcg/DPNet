#改进后的病理图像分割和分类模型
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
BATCH_SIZE = 4
NUM_CLASSES_SEG = 2  # 分割类别数（前景/背景）
NUM_CLASSES_CLS = 6  # 分类类别数（6个类别）
EPOCHS = 200
LEARNING_RATE = 0.001
DATA_ROOT = 'D:/EBHI-SEG'  # 数据根目录


# 自定义分割数据集类
class EBHISegDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        """
        Args:
            root_dir: 数据根目录，包含六个类别子文件夹
            transform: 图像预处理
            mask_transform: 掩码预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # 存储类别标签

        # 遍历六个类别文件夹
        class_names = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])

        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            image_dir = os.path.join(class_path, 'image')
            label_dir = os.path.join(class_path, 'label')

            # 检查image和label文件夹是否存在
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                print(f"警告: {class_name} 文件夹下缺少 image 或 label 子文件夹")
                continue

            # 获取所有图像文件
            image_files = sorted([f for f in os.listdir(image_dir)
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])

            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(label_dir, img_file)  # 假设标签文件名与图像相同

                # 如果标签文件存在
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(class_idx)
                else:
                    print(f"警告: 找不到对应的标签文件 {mask_path}")

        print(f"数据集加载完成，共 {len(self.image_paths)} 个样本")
        print(f"类别: {class_names}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # 加载分割标签
        mask = Image.open(self.mask_paths[idx]).convert('L')  # 灰度图

        # 获取类别标签
        label = self.labels[idx]

        # 应用变换
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 将掩码转换为二值图（0/1），假设前景像素值>0
        mask = (mask > 0).float()

        return image, mask, label


# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
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
    transforms.ToTensor()
])


# 训练函数（同时训练分割和分类）
def train_epoch_multi_task(model, loader, seg_criterion, cls_criterion, optimizer, cls_weight=0.5):
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

        # 前向传播 - model返回 (分割结果, 分类结果)
        seg_output, cls_output, _ , _ = model(images)

        # 计算分割损失
        seg_loss = seg_criterion(seg_output, masks.squeeze(1).long())

        # 计算分类损失
        cls_loss = cls_criterion(cls_output, labels)

        # 总损失 = 分割损失 + 分类损失 * 权重
        loss = seg_loss + cls_weight * cls_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失
        total_seg_loss += seg_loss.item()
        total_cls_loss += cls_loss.item()
        total_loss += loss.item()

        # 计算分割准确率
        seg_pred = torch.argmax(seg_output, dim=1)
        total_seg_correct += (seg_pred == masks.squeeze(1)).sum().item()
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
def validate_multi_task(model, loader, seg_criterion, cls_criterion, cls_weight=0.5):
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
            seg_loss = seg_criterion(seg_output, masks.squeeze(1).long())
            cls_loss = cls_criterion(cls_output, labels)
            loss = seg_loss + cls_weight * cls_loss

            # 统计损失
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()

            # 计算分割准确率和指标
            seg_pred = torch.argmax(seg_output, dim=1)
            total_seg_correct += (seg_pred == masks.squeeze(1)).sum().item()
            total_pixels += masks.numel()

            # 计算Dice系数和IoU
            pred_binary = (seg_pred > 0).float()
            target_binary = masks.squeeze(1)

            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum() - intersection

            dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
            iou = intersection / (union + 1e-6)

            total_dice += dice.item()
            total_iou += iou.item()

            # 计算分类准确率
            cls_pred = torch.argmax(cls_output, dim=1)
            total_cls_correct += (cls_pred == labels).sum().item()
            total_samples += labels.size(0)

    avg_seg_loss = total_seg_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_total_loss = total_loss / len(loader)
    seg_acc = total_seg_correct / total_pixels
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    cls_acc = total_cls_correct / total_samples

    return avg_total_loss, avg_seg_loss, avg_cls_loss, seg_acc, avg_dice, avg_iou, cls_acc


# Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W]
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# 组合分割损失函数
class CombinedSegLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        return (self.ce_weight * self.ce_loss(pred, target) +
                self.dice_weight * self.dice_loss(pred, target))


if __name__ == '__main__':
    # 创建完整数据集
    full_dataset = EBHISegDataset(
        root_dir=DATA_ROOT,
        transform=train_transform,
        mask_transform=mask_transform
    )

    # 划分训练集和验证集 (80%训练，20%验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 为验证集设置不同的transform（无数据增强）
    val_dataset.dataset.transform = val_transform

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n数据集统计:")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    # 初始化模型 - 注意这里需要修改UNet_fpn的输出
    # 假设UNet_fpn返回两个输出：(分割输出, 分类输出)
    model = unet_fpn.UNet_fpn(num_classes=NUM_CLASSES_SEG,
                              in_channels=3).to(device)

    # 损失函数
    seg_criterion = CombinedSegLoss(ce_weight=1.0, dice_weight=1.0)
    cls_criterion = nn.CrossEntropyLoss()

    # 分类损失的权重
    CLS_WEIGHT = 0.3  # 可以根据需要调整

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练循环
    best_combined_score = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print("-" * 50)

        # 训练
        train_total_loss, train_seg_loss, train_cls_loss, train_seg_acc, train_cls_acc = \
            train_epoch_multi_task(model, train_loader, seg_criterion, cls_criterion, optimizer, CLS_WEIGHT)

        print(f"训练 - 总损失: {train_total_loss:.4f}, 分割损失: {train_seg_loss:.4f}, 分类损失: {train_cls_loss:.4f}")
        print(f"训练 - 分割准确率: {train_seg_acc:.4f}, 分类准确率: {train_cls_acc:.4f}")

        # 验证
        val_total_loss, val_seg_loss, val_cls_loss, val_seg_acc, val_dice, val_iou, val_cls_acc = \
            validate_multi_task(model, val_loader, seg_criterion, cls_criterion, CLS_WEIGHT)

        print(f"验证 - 总损失: {val_total_loss:.4f}, 分割损失: {val_seg_loss:.4f}, 分类损失: {val_cls_loss:.4f}")
        print(f"验证 - 分割准确率: {val_seg_acc:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, 分类准确率: {val_cls_acc:.4f}")

        scheduler.step()

        # 计算综合得分（可以自定义权重）
        combined_score = val_dice * 0.5 + val_cls_acc * 0.5

        # 每轮都保存模型权重，文件名包含分类准确率和分割Dice系数
        model_filename = f'epoch_{epoch + 1:03d}_dice_{val_dice:.4f}_clsacc_{val_cls_acc:.4f}.pth'
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

        # # 可选：同时保存最佳模型（基于综合得分）
        # if combined_score > best_combined_score:
        #     best_combined_score = combined_score
        #     best_model_filename = f'best_model_dice_{val_dice:.4f}_clsacc_{val_cls_acc:.4f}.pth'
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'best_combined_score': best_combined_score,
        #         'best_dice': val_dice,
        #         'best_iou': val_iou,
        #         'best_cls_acc': val_cls_acc,
        #     }, best_model_filename)
        #     print(f"✓ 保存最佳模型: {best_model_filename} (Dice: {val_dice:.4f}, 分类准确率: {val_cls_acc:.4f})")

    print(f"\n 训练完成！最佳综合得分: {best_combined_score:.4f}")


    # 测试输出形状
    def test_output_shape():
        model.eval()
        sample_input = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            seg_output, cls_output = model(sample_input)
        print(f"\n输出形状验证:")
        print(f"输入形状: {sample_input.shape}")
        print(f"分割输出形状: {seg_output.shape}")  # [4, NUM_CLASSES_SEG, 224, 224]
        print(f"分类输出形状: {cls_output.shape}")  # [4, NUM_CLASSES_CLS]
        return seg_output, cls_output


    test_output_shape()