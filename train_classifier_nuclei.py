# nuclei_segmentation_datasets训练代码（修正版）
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
NUM_CLASSES_CLS = 4  # 分类类别数（cpm15, cpm17, kumar等4个类别）
EPOCHS = 200
LEARNING_RATE = 0.001
DATA_ROOT = 'D:/病理图像/nuclei_segmentation_datasets'  # 数据根目录


# 自定义细胞核分割数据集类
class NucleiSegDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None):
        """
        Args:
            root_dir: 数据根目录，包含四个类别子文件夹（cpm15, cpm17等）
            split: 'train', 'test', 或 None（如果直接使用Images/Masks结构）
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

        # 获取所有类别文件夹
        class_dirs = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])

        print(f"找到类别文件夹: {class_dirs}")

        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(root_dir, class_name)

            # 探索该类下的结构
            self._explore_class_folder(class_path, class_idx, class_name)

        print(f"\n{split}集加载完成，共 {len(self.image_paths)} 个样本")
        if len(self.image_paths) > 0:
            # 统计类别分布
            unique_labels = set(self.labels)
            print(f"类别分布:")
            for i in sorted(unique_labels):
                count = self.labels.count(i)
                class_name = class_dirs[i] if i < len(class_dirs) else f"类别{i}"
                print(f"  {class_name}: {count} 个样本 ({count / len(self.labels) * 100:.1f}%)")
        else:
            print(f"警告: {split}集中没有加载到任何样本！")

    def _explore_class_folder(self, class_path, class_idx, class_name, current_depth=0):
        """
        递归探索类别文件夹，寻找Images和Masks文件夹

        Args:
            class_path: 当前探索的路径
            class_idx: 类别索引
            class_name: 类别名称
            current_depth: 当前递归深度（防止无限递归）
        """
        if current_depth > 5:  # 防止递归过深
            return

        try:
            items = os.listdir(class_path)
        except PermissionError:
            return

        # 情况1：当前文件夹下直接有Images和Masks
        if 'Images' in items and 'Masks' in items:
            images_dir = os.path.join(class_path, 'Images')
            masks_dir = os.path.join(class_path, 'Masks')

            self._load_images_masks(images_dir, masks_dir, class_idx, class_name)
            return

        # 情况2：检查是否有train/test等子文件夹
        split_dirs = []
        for item in items:
            item_path = os.path.join(class_path, item)
            if os.path.isdir(item_path):
                # 如果当前有split指定的文件夹
                if self.split and item == self.split:
                    self._explore_class_folder(item_path, class_idx, class_name, current_depth + 1)
                    return
                # 记录所有可能的子文件夹
                split_dirs.append(item_path)

        # 情况3：如果没找到split指定的文件夹，但找到了其他子文件夹
        if not self.split or self.split == 'all':
            for sub_dir in split_dirs:
                self._explore_class_folder(sub_dir, class_idx, class_name, current_depth + 1)

    def _load_images_masks(self, images_dir, masks_dir, class_idx, class_name):
        """从Images和Masks文件夹加载图像和掩码对"""
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            return

        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]

        if not image_files:
            # 尝试查找子文件夹中的图像
            sub_dirs = [d for d in os.listdir(images_dir)
                        if os.path.isdir(os.path.join(images_dir, d))]
            for sub_dir in sub_dirs:
                sub_images_dir = os.path.join(images_dir, sub_dir)
                sub_masks_dir = os.path.join(masks_dir, sub_dir)
                if os.path.exists(sub_masks_dir):
                    self._load_images_masks(sub_images_dir, sub_masks_dir, class_idx, class_name)
            return

        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)

            # 尝试不同的mask文件命名方式
            mask_found = False

            # 获取文件名和扩展名
            file_name, file_ext = os.path.splitext(img_file)
            file_ext = file_ext.lower()

            # 可能的mask文件名列表
            possible_mask_names = []

            # 1. 直接同名（但扩展名可能是.png）
            if file_ext != '.png':
                possible_mask_names.append(file_name + '.png')

            # 2. 常见的mask命名模式
            possible_mask_names.extend([
                img_file,  # 同名（适用于图像和mask扩展名相同的情况）
                file_name + '_mask.png',
                'mask_' + file_name + '.png',
                file_name + '_mask' + file_ext,
                file_name.replace('_image', '') + '_mask.png',
                file_name.replace('_img', '') + '_mask.png',
            ])

            # 3. 针对特定数据集的命名模式
            # 对于kumar数据集：TCGA-XXX.tif 对应 TCGA-XXX.png
            if 'TCGA' in file_name:
                possible_mask_names.append(file_name + '.png')
                possible_mask_names.append(file_name.replace('TCGA', '') + '.png')

            # 4. 移除可能的前缀/后缀
            for prefix in ['image_', 'img_', 'im_']:
                if file_name.startswith(prefix):
                    base_name = file_name[len(prefix):]
                    possible_mask_names.append(base_name + '.png')

            for suffix in ['_image', '_img', '_im']:
                if file_name.endswith(suffix):
                    base_name = file_name[:-len(suffix)]
                    possible_mask_names.append(base_name + '.png')

            # 去重
            possible_mask_names = list(dict.fromkeys(possible_mask_names))

            # 尝试所有可能的mask文件名
            for mask_name in possible_mask_names:
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(class_idx)
                    mask_found = True
                    break

            if not mask_found:
                # 如果没找到mask，但图像文件本身可能包含"mask"字样，跳过
                if 'mask' not in img_file.lower():
                    print(f"警告: 在 {class_name} 中找不到 {img_file} 对应的mask文件")
                    # 调试信息：显示尝试过的mask文件名
                    # print(f"尝试过的mask文件名: {possible_mask_names[:5]}")

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
    transforms.ToTensor()
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

            # 计算每个batch的Dice系数和IoU
            for i in range(images.size(0)):
                pred_binary = (seg_pred[i] > 0).float()
                target_binary = masks[i].squeeze(0)

                if target_binary.sum() == 0 and pred_binary.sum() == 0:
                    dice = 1.0
                    iou = 1.0
                elif target_binary.sum() == 0:
                    dice = 0.0
                    iou = 0.0
                else:
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
    avg_dice = total_dice / len(loader.dataset)  # 除以样本数
    avg_iou = total_iou / len(loader.dataset)
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

        # 只计算前景类（索引1）的Dice
        pred_foreground = pred[:, 1, :, :]
        target_foreground = target_onehot[:, 1, :, :]

        intersection = (pred_foreground * target_foreground).sum(dim=(1, 2))
        union = pred_foreground.sum(dim=(1, 2)) + target_foreground.sum(dim=(1, 2))

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


# 调试函数：检查特定类别的文件结构
def debug_dataset_structure():
    """调试函数：检查数据集的具体结构"""
    class_dirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]

    for class_name in class_dirs:
        class_path = os.path.join(DATA_ROOT, class_name)
        print(f"\n=== 检查类别: {class_name} ===")

        for root, dirs, files in os.walk(class_path):
            level = root.replace(class_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")

            # 显示前5个文件作为示例
            image_files = [f for f in files if f.endswith(('.tif', '.tiff', '.png', '.jpg'))]
            if image_files:
                print(f"{indent}  文件示例: {image_files[:3]}")


if __name__ == '__main__':
    # 可选：先调试查看数据结构
    # debug_dataset_structure()
    # exit()

    print("=" * 60)
    print("正在加载训练集...")

    # 创建训练集（使用split='train'）
    train_dataset = NucleiSegDataset(
        root_dir=DATA_ROOT,
        split='train',  # 会查找train文件夹
        transform=train_transform,
        mask_transform=mask_transform
    )

    print("\n" + "=" * 60)
    print("正在加载测试集（作为验证集）...")

    # 创建验证集（使用split='test'）
    val_dataset = NucleiSegDataset(
        root_dir=DATA_ROOT,
        split='test',  # 会查找test文件夹
        transform=val_transform,
        mask_transform=mask_transform
    )

    # 如果上面加载的数据集为空，尝试使用其他split名称
    if len(train_dataset) == 0:
        print("\n尝试使用其他split名称加载训练集...")
        for split_name in ['training', 'Training', 'train_data', 'Train']:
            train_dataset = NucleiSegDataset(
                root_dir=DATA_ROOT,
                split=split_name,
                transform=train_transform,
                mask_transform=mask_transform
            )
            if len(train_dataset) > 0:
                print(f"成功使用 '{split_name}' 加载训练集")
                break

    if len(val_dataset) == 0:
        print("\n尝试使用其他split名称加载验证集...")
        for split_name in ['testing', 'Testing', 'test_data', 'Test', 'val', 'validation']:
            val_dataset = NucleiSegDataset(
                root_dir=DATA_ROOT,
                split=split_name,
                transform=val_transform,
                mask_transform=mask_transform
            )
            if len(val_dataset) > 0:
                print(f"成功使用 '{split_name}' 加载验证集")
                break

    # 如果还是为空，尝试加载所有数据并手动划分
    if len(train_dataset) == 0 and len(val_dataset) == 0:
        print("\n未找到train/test文件夹，尝试加载所有数据...")
        full_dataset = NucleiSegDataset(
            root_dir=DATA_ROOT,
            split='all',  # 加载所有数据
            transform=train_transform,
            mask_transform=mask_transform
        )

        if len(full_dataset) > 0:
            # 手动划分训练集和验证集
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            print(f"手动划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
        else:
            raise ValueError("无法加载任何数据！请检查数据路径和文件夹结构。")

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

    # 分类损失的权重
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
        print(f"验证 - 分割准确率: {val_seg_acc:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, 分类准确率: {val_cls_acc:.4f}")

        scheduler.step()

        # 计算综合得分
        combined_score = val_dice * 0.5 + val_cls_acc * 0.5

        # 保存模型
        model_filename = f'nuclei_seg_epoch_{epoch + 1:03d}_dice_{val_dice:.4f}_clsacc_{val_cls_acc:.4f}.pth'
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
            best_model_filename = f'nuclei_seg_best_model_dice_{val_dice:.4f}_clsacc_{val_cls_acc:.4f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_combined_score': best_combined_score,
                'best_dice': val_dice,
                'best_iou': val_iou,
                'best_cls_acc': val_cls_acc,
            }, best_model_filename)
            print(f"✓ 保存最佳模型: {best_model_filename}")

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
        print(f"分割输出形状: {seg_output.shape}")
        print(f"分类输出形状: {cls_output.shape}")
        return seg_output, cls_output


    test_output_shape()