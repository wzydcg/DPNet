import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import torch


def process_image_with_equations_tensor(image_tensor, alpha=30, save_visualization=True):
    """
    按照论文公式完整实现频域纹理提取过程，输入为图像张量

    公式对应:
    (1) Xf = F(Xd) - 傅里叶变换
    (2) Xh = F^{-1}(H(Xf, alpha)) - 高通滤波并逆变换

    Args:
        image_tensor: 输入图像张量，支持以下格式：
                      - torch.Tensor: [C, H, W] 或 [B, C, H, W]，值范围 [0,1] 或 [0,255]
                      - numpy.ndarray: [H, W, C] 或 [C, H, W]，值范围 [0,255] 或 [0,1]
        alpha: 高通滤波器阈值，控制滤除的低频分量半径
        save_visualization: 是否保存可视化结果

    Returns:
        Xd: 原始图像张量 (torch.Tensor, [H, W, C] 或 [B, C, H, W], 范围 [0,1])
        Xh: 高频纹理图像张量 (torch.Tensor, [H, W, C] 或 [B, C, H, W], 范围 [0,1])
    """

    # 记录输入类型和维度，用于后续恢复
    input_is_torch = isinstance(image_tensor, torch.Tensor)
    input_is_batch = False
    original_shape = None

    if input_is_torch:
        original_shape = image_tensor.shape
        if image_tensor.dim() == 4:  # [B, C, H, W]
            input_is_batch = True
            batch_size, channels, height, width = image_tensor.shape
            # print(f"输入为批量数据: {image_tensor.shape}")

            # 批量处理
            Xd_batch = []
            Xh_batch = []

            for i in range(batch_size):
                Xd, Xh = process_single_image(
                    image_tensor[i], alpha, save_visualization=(save_visualization and i == 0)
                )
                Xd_batch.append(torch.from_numpy(Xd).permute(2, 0, 1))  # [H,W,C] -> [C,H,W]
                Xh_batch.append(torch.from_numpy(Xh).permute(2, 0, 1))

            # 堆叠为 [B, C, H, W]
            Xd = torch.stack(Xd_batch, dim=0)
            Xh = torch.stack(Xh_batch, dim=0)

            return Xd, Xh
        else:
            # 单张图像 [C, H, W]
            Xd, Xh = process_single_image(image_tensor, alpha, save_visualization)
            # 转换为 [C, H, W] 的 torch.Tensor
            Xd = torch.from_numpy(Xd).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            Xh = torch.from_numpy(Xh).permute(2, 0, 1)
            return Xd, Xh
    else:
        # numpy输入
        if image_tensor.ndim == 4:  # [B, H, W, C] 或 [B, C, H, W]
            input_is_batch = True
            # 转换为 torch 处理
            Xd, Xh = process_image_with_equations_tensor(
                torch.from_numpy(image_tensor).float(), alpha, save_visualization
            )
            return Xd.numpy(), Xh.numpy() if not input_is_torch else (Xd, Xh)
        else:
            Xd, Xh = process_single_image(image_tensor, alpha, save_visualization, input_is_torch=False)
            return Xd, Xh


def process_single_image(image_input, alpha=30, save_visualization=True, input_is_torch=True):
    """
    处理单张图像的核心函数
    """
    # 将输入转换为numpy数组 [H, W, C]，范围 [0,1]
    if input_is_torch:
        # 处理 torch.Tensor
        img_tensor = image_input.detach().cpu()

        # 转换为 [H, W, C] 格式
        if img_tensor.dim() == 3:
            if img_tensor.shape[0] in [1, 3]:  # [C, H, W]
                img_tensor = img_tensor.permute(1, 2, 0)

        # 转换为numpy并确保范围 [0,1]
        img_np = img_tensor.numpy()
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
    else:
        # 处理 numpy.ndarray
        img_np = image_input.copy()

        # 处理 [C, H, W] 格式
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3] and img_np.shape[2] not in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))

        # 确保范围 [0,1]
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

    # 确保是3通道
    if img_np.ndim == 2:
        # 灰度图转RGB
        img_np = np.stack([img_np, img_np, img_np], axis=2)
    elif img_np.shape[2] == 1:
        # 单通道转RGB
        img_np = np.concatenate([img_np, img_np, img_np], axis=2)
    elif img_np.shape[2] > 3:
        # 多通道，取前3个通道
        img_np = img_np[:, :, :3]

    # 公式中的Xd
    Xd = img_np.astype(np.float32)

    # 公式(1): Xf = F(Xd) - 转换到频域
    Xf_list = []
    magnitude_list = []

    for c in range(3):
        # 傅里叶变换
        f_transform = fft2(Xd[:, :, c])
        f_shifted = fftshift(f_transform)
        Xf_list.append(f_shifted)

        # 计算幅度谱用于可视化
        magnitude = np.log(np.abs(f_shifted) + 1)
        magnitude_list.append(magnitude)

    # 公式(2): Xh = F^{-1}(H(Xf, α))
    Xh_list = []

    h, w = Xd.shape[:2]
    center_h, center_w = h // 2, w // 2
    y, x = np.ogrid[:h, :w]

    # 创建高通滤波器掩码
    mask = np.ones((h, w), dtype=np.float32)
    mask_area = (x - center_w) ** 2 + (y - center_h) ** 2 <= alpha ** 2
    mask[mask_area] = 0.0  # 低频区域设为0

    for c in range(3):
        # 应用高通滤波器 H(Xf, alpha)
        f_filtered = Xf_list[c] * mask

        # 逆傅里叶变换回空间域
        f_ishifted = ifftshift(f_filtered)
        img_high = np.real(ifft2(f_ishifted))
        img_high = np.clip(img_high, 0, 1)
        Xh_list.append(img_high)

    # 合并通道得到最终的高频纹理图像 Xh
    Xh = np.stack(Xh_list, axis=2)

    # # 可视化
    # if save_visualization:
    #     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    #
    #     # 原始图像
    #     axes[0, 0].imshow(Xd)
    #     axes[0, 0].set_title(f'Original RGB Image Xd')
    #     axes[0, 0].axis('off')
    #
    #     # 频域幅度谱
    #     magnitude_combined = np.mean(np.array(magnitude_list), axis=0)
    #     axes[0, 1].imshow(magnitude_combined, cmap='hot')
    #     axes[0, 1].set_title(f'Frequency Domain Xf (α={alpha})')
    #     axes[0, 1].axis('off')
    #
    #     # 高通滤波器掩码
    #     axes[0, 2].imshow(mask, cmap='gray')
    #     axes[0, 2].set_title('High-pass Filter Mask H')
    #     axes[0, 2].axis('off')
    #
    #     # 高频纹理图像
    #     axes[1, 0].imshow(Xh)
    #     axes[1, 0].set_title(f'High-frequency Texture Xh')
    #     axes[1, 0].axis('off')
    #
    #     # 原始图像与纹理叠加
    #     overlay = np.clip(Xd + Xh * 0.5, 0, 1)
    #     axes[1, 1].imshow(overlay)
    #     axes[1, 1].set_title('Original + Texture Overlay')
    #     axes[1, 1].axis('off')
    #
    #     # 边缘检测对比
    #     edges_rgb = cv2.Canny((Xd * 255).astype(np.uint8), 50, 150)
    #     edges_texture = cv2.Canny((Xh * 255).astype(np.uint8), 30, 100)
    #     axes[1, 2].imshow(edges_texture, cmap='gray')
    #     axes[1, 2].set_title('Texture Edges')
    #     axes[1, 2].axis('off')
    #
    #     plt.tight_layout()
    #     plt.savefig('frequency_texture_extraction.png', dpi=150, bbox_inches='tight')
    #     plt.show()

    return Xd, Xh


def batch_process_tensors(image_tensors, alpha=30):
    """
    批量处理多个图像张量，返回堆叠的张量

    Args:
        image_tensors: 可以是:
                       - torch.Tensor: [B, C, H, W]
                       - list of torch.Tensor: 每个 [C, H, W]
                       - numpy.ndarray: [B, H, W, C] 或 [B, C, H, W]
        alpha: 高通滤波器阈值

    Returns:
        original_batch: 原始图像批量 (torch.Tensor, [B, C, H, W], 范围 [0,1])
        texture_batch: 纹理图像批量 (torch.Tensor, [B, C, H, W], 范围 [0,1])
    """
    # 直接调用主函数，它已经支持批量处理并返回 [B, C, H, W] 格式
    return process_image_with_equations_tensor(image_tensors, alpha=alpha, save_visualization=False)


# 示例用法
if __name__ == "__main__":
    # 创建测试张量
    test_tensor = torch.randn(3, 28, 28)  # [C, H, W]
    test_tensor = torch.clamp(test_tensor, 0, 1)  # 确保在 [0,1] 范围内

    # 处理单张图像 - 返回 [C, H, W]
    Xd, Xh = process_image_with_equations_tensor(test_tensor, alpha=10)
    print(f"单张图像 - 原始形状: {Xd.shape}")  # [3, 28, 28]
    print(f"单张图像 - 纹理形状: {Xh.shape}")  # [3, 28, 28]

    # 处理批量图像 - 返回 [B, C, H, W]
    batch_tensor = torch.randn(4, 3, 28, 28)  # [B, C, H, W]
    batch_tensor = torch.clamp(batch_tensor, 0, 1)
    Xd_batch, Xh_batch = process_image_with_equations_tensor(batch_tensor, alpha=10)
    print(f"\n批量图像 - 原始形状: {Xd_batch.shape}")  # [4, 3, 28, 28]
    print(f"批量图像 - 纹理形状: {Xh_batch.shape}")  # [4, 3, 28, 28]