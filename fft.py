import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def extract_high_frequency_texture(image_path, alpha=30, downsample_factor=1):
    """
    从图像中提取高频纹理细节，基于频域处理

    Args:
        image_path: 图像文件路径
        alpha: 高通滤波器阈值，控制滤除的低频分量（频率半径）
        downsample_factor: 下采样因子（默认为1，表示不下采样）

    Returns:
        original_img: 原始图像
        high_freq_img: 高频纹理图像
        freq_domain: 频域表示（用于可视化）
    """
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]

    # 下采样（如果需要）
    if downsample_factor > 1:
        h, w = img.shape[:2]
        new_h, new_w = h // downsample_factor, w // downsample_factor
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]

    # 对每个通道进行傅里叶变换
    freq_channels = []
    for c in range(3):
        # 2D傅里叶变换
        f = fft2(img[:, :, c])
        f_shifted = fftshift(f)  # 将低频移到中心

        # 创建高通滤波器掩码
        mask = np.ones((h, w), dtype=np.uint8)
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        mask_area = (x - center_w) ** 2 + (y - center_h) ** 2 <= alpha ** 2
        mask[mask_area] = 0  # 低频区域设为0，保留高频

        # 应用高通滤波器
        f_high = f_shifted * mask

        # 逆傅里叶变换
        f_high_ishift = ifftshift(f_high)
        img_high = np.real(ifft2(f_high_ishift))

        # 裁剪到有效范围
        img_high = np.clip(img_high, 0, 1)
        freq_channels.append(img_high)

        # 保存频域用于可视化
        if c == 0:
            magnitude = np.log(np.abs(f_shifted) + 1)
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            freq_domain = magnitude

    # 合并通道
    high_freq_img = np.stack(freq_channels, axis=2)

    return img, high_freq_img, freq_domain


def process_image_with_equations(image_path, alpha=30, save_visualization=True):
    """
    按照论文公式完整实现频域纹理提取过程

    公式对应:
    (1) Xf = F(Xd) - 傅里叶变换
    (2) Xh = F^{-1}(H(Xf, α)) - 高通滤波并逆变换
    """

    # 读取图像
    img_rgb = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0

    # 公式中的Xd (可能的下采样)
    Xd = img_float

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
        # 应用高通滤波器 H(Xf, α)
        f_filtered = Xf_list[c] * mask

        # 逆傅里叶变换回空间域
        f_ishifted = ifftshift(f_filtered)
        img_high = np.real(ifft2(f_ishifted))
        img_high = np.clip(img_high, 0, 1)
        Xh_list.append(img_high)

    # 合并通道得到最终的高频纹理图像 Xh
    Xh = np.stack(Xh_list, axis=2)

    # 可视化
    if save_visualization:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 原始图像
        axes[0, 0].imshow(Xd)
        axes[0, 0].set_title(f'Original RGB Image Xd')
        axes[0, 0].axis('off')

        # 频域幅度谱
        magnitude_combined = np.mean(np.array(magnitude_list), axis=0)
        axes[0, 1].imshow(magnitude_combined, cmap='hot')
        axes[0, 1].set_title(f'Frequency Domain Xf (α={alpha})')
        axes[0, 1].axis('off')

        # 高通滤波器掩码
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('High-pass Filter Mask H')
        axes[0, 2].axis('off')

        # 高频纹理图像
        axes[1, 0].imshow(Xh)
        axes[1, 0].set_title(f'High-frequency Texture Xh')
        axes[1, 0].axis('off')

        # 原始图像与纹理叠加
        overlay = np.clip(Xd + Xh * 0.5, 0, 1)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Original + Texture Overlay')
        axes[1, 1].axis('off')

        # 边缘检测对比
        edges_rgb = cv2.Canny((Xd * 255).astype(np.uint8), 50, 150)
        edges_texture = cv2.Canny((Xh * 255).astype(np.uint8), 30, 100)
        axes[1, 2].imshow(edges_texture, cmap='gray')
        axes[1, 2].set_title('Texture Edges')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('frequency_texture_extraction.png', dpi=150, bbox_inches='tight')
        plt.show()

    return Xd, Xh


def batch_process_images(image_paths, alpha_values=[20, 30, 40]):
    """
    批量处理多张图像，比较不同alpha阈值的效果
    """
    fig, axes = plt.subplots(len(image_paths), len(alpha_values) + 1,
                             figsize=(15, 4 * len(image_paths)))

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img.astype(np.float32) / 255.0

        # 显示原始图像
        axes[i, 0].imshow(img_norm)
        axes[i, 0].set_title(f'Original {i + 1}')
        axes[i, 0].axis('off')

        # 对不同alpha值提取高频纹理
        for j, alpha in enumerate(alpha_values):
            _, Xh = process_image_with_equations(img_path, alpha=alpha, save_visualization=False)
            axes[i, j + 1].imshow(Xh)
            axes[i, j + 1].set_title(f'α={alpha}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig('alpha_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 单张图像处理
    image_path = "D:/GT2000000-1-400-001.png"  # 替换为您的图像路径

    # 不同alpha阈值的效果
    for alpha in [10, 20, 30, 40]:
        print(f"\n处理图像，alpha={alpha}:")
        original, texture = process_image_with_equations(image_path, alpha=alpha)
        print(texture.shape)
        print(f"原始图像形状: {original.shape}")
        print(f"高频纹理图像形状: {texture.shape}")
        print(f"纹理强度 (均值±标准差): {texture.mean():.4f} ± {texture.std():.4f}")

    # 批量处理示例（如果需要）
    # image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
    # batch_process_images(image_list)